import asyncio
import base64
import io
import os
import re
import time
import typing as T
from datetime import timedelta
import random
import json
from collections import deque
from contextlib import asynccontextmanager

import google.generativeai as genai
import PIL.Image
from anthropic import AsyncAnthropic, RateLimitError
from devtools import debug
from google.generativeai import caching as gemini_caching
from openai import AsyncAzureOpenAI, AsyncOpenAI
from xai_sdk import AsyncClient
from xai_sdk.chat import user, assistant, system, image
import httpx

from src import logfire
from src.logic import random_string
from src.models import Attempt, Model, ModelUsage

if "GEMINI_API_KEY" in os.environ:
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])


def remove_thinking(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)


# Global concurrency and rate limiting for OpenRouter
_openrouter_sem: asyncio.Semaphore | None = None
_openrouter_timestamps: deque[float] = deque()
_openrouter_lock = asyncio.Lock()


def _get_openrouter_limits() -> tuple[int, int]:
    try:
        rpm = max(1, int(os.environ.get("OPENROUTER_RPM", "20")))
    except Exception:
        rpm = 20
    try:
        conc = max(1, int(os.environ.get("OPENROUTER_CONCURRENCY", "20")))
    except Exception:
        conc = 20
    # Hard cap to avoid runaway
    conc = min(conc, 40)
    rpm = min(rpm, 120)  # allow user to raise a bit, but keep sane upper bound
    return rpm, conc


@asynccontextmanager
async def _openrouter_guard():
    global _openrouter_sem
    rpm, conc = _get_openrouter_limits()
    if _openrouter_sem is None:
        _openrouter_sem = asyncio.Semaphore(conc)
    await _openrouter_sem.acquire()
    try:
        # Simple token bucket: ensure <= rpm per 60 seconds
        async with _openrouter_lock:
            now = time.time()
            # prune old
            while _openrouter_timestamps and now - _openrouter_timestamps[0] >= 60.0:
                _openrouter_timestamps.popleft()
            # if full, wait until next slot opens
            while len(_openrouter_timestamps) >= rpm:
                earliest = _openrouter_timestamps[0]
                wait_s = max(0.0, 60.0 - (now - earliest)) + random.uniform(0.01, 0.05)
                await asyncio.sleep(wait_s)
                now = time.time()
                while _openrouter_timestamps and now - _openrouter_timestamps[0] >= 60.0:
                    _openrouter_timestamps.popleft()
            _openrouter_timestamps.append(now)
        yield
    finally:
        _openrouter_sem.release()


def text_only_messages(messages: list[dict[str, T.Any]]) -> list[dict[str, T.Any]]:
    new_messages = []
    for message in messages:
        content_strs: list[str] = []
        if isinstance(message["content"], str):
            content_strs.append(message["content"])
        else:
            for content in message["content"]:
                if content["type"] == "text":
                    content_strs.append(content["text"])
        if content_strs:
            new_messages.append(
                {
                    "role": message["role"],
                    "content": "\n".join(content_strs),
                }
            )
    return new_messages


async def _openrouter_completion_with_adaptive_max_tokens(
    *,
    client: AsyncOpenAI,
    model_name: str,
    messages: list[dict[str, T.Any]],
    temperature: float,
    extra_body: dict | None = None,
    start_cap: int | None = None,
    max_attempts: int = 1,
):
    """
    Call OpenRouter with an adaptive max_tokens strategy.

    - Starts from an initial cap (env OPENROUTER_MAX_TOKENS or 400_000)
    - On size/limit/context errors: backs off (halves or tightens) and retries
    - On JSON decode / HTML responses: treats as transient, logs a preview, backs off, and retries
    - On other transient errors (rate/timeout/gateway): randomized exponential backoff and retry
    """
    cap = start_cap or int(os.environ.get("OPENROUTER_MAX_TOKENS", "400000"))
    attempt = 0
    last_err: Exception | None = None

    MIN_CAP = 256  # Conservative floor to avoid infinite loops

    while attempt < max_attempts and cap >= MIN_CAP:
        try:
            return await client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=cap,
                extra_body=extra_body or {},
            )
        except Exception as e:
            s = str(e).lower()
            last_err = e

            # Try to preview a snippet of the response body for diagnostics
            body_preview = ""
            try:
                resp = getattr(e, "response", None)
                if resp is not None:
                    # Some SDKs expose .text, others require reading .body
                    if hasattr(resp, "text") and resp.text is not None:
                        body_preview = (resp.text or "")[:256]
                    elif hasattr(resp, "body") and resp.body is not None:
                        try:
                            body_preview = resp.body.decode("utf-8", errors="ignore")[:256]
                        except Exception:
                            body_preview = str(resp.body)[:256]
            except Exception:
                pass

            # Log the error with a short body preview (helps spot HTML or gateway pages)
            try:
                logfire.debug(
                    f"OpenRouter error (cap={cap}, attempt={attempt}): {e} | body_preview={body_preview!r}"
                )
            except Exception:
                pass
            # Also print to console so it isn't missed when debug logs are not visible
            try:
                print(
                    f"OpenRouter error (cap={cap}, attempt={attempt}): {e} | body_preview={body_preview!r}"
                )
            except Exception:
                pass

            # Optional diagnostic raw retry to capture status/headers/body
            if os.environ.get("OPENROUTER_DIAG_RAW", "0") == "1":
                try:
                    async with _openrouter_guard():
                        async with httpx.AsyncClient(timeout=60) as _client:
                            headers = {
                                "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
                                "Content-Type": "application/json",
                                "Accept": "application/json",
                                "Connection": "close",
                                "HTTP-Referer": os.environ.get("OPENROUTER_SITE_URL", "http://localhost"),
                                "X-Title": os.environ.get("OPENROUTER_APP_NAME", "arc_agi"),
                            }
                            payload = {
                                "model": model_name,
                                "messages": messages,
                                "temperature": temperature,
                                "max_tokens": cap,
                            }
                            if extra_body:
                                payload["extra_body"] = extra_body
                            _resp = await _client.post(
                                "https://openrouter.ai/api/v1/chat/completions",
                                json=payload,
                                headers=headers,
                            )
                            _preview = (_resp.text or "")[:2048]
                            print(
                                f"[DIAG_RAW] status={_resp.status_code} ct={_resp.headers.get('content-type')} cl={_resp.headers.get('content-length')} body_preview={_preview!r}"
                            )
                            logfire.debug(
                                f"[DIAG_RAW] status={_resp.status_code} ct={_resp.headers.get('content-type')} cl={_resp.headers.get('content-length')} body_preview={_preview!r}"
                            )
                except Exception as de:
                    try:
                        print(f"[DIAG_RAW] failed to fetch raw response: {de}")
                        logfire.debug(f"[DIAG_RAW] failed to fetch raw response: {de}")
                    except Exception:
                        pass

            # 1) Token/context-limit hints => reduce cap and retry
            limit_hints = (
                "413",
                "payload too large",
                "too large",
                "max tokens",
                "maximum tokens",
                "max_output_tokens",
                "exceeds",
                "length",
                "context",
                "token limit",
            )
            if any(h in s for h in limit_hints) or any(h in (body_preview or "").lower() for h in limit_hints):
                cap = max(MIN_CAP, cap // 2)
                attempt += 1
                continue

            # 2) JSON decode / non-JSON body (proxy/HTML/gateway) => transient retry with backoff
            json_hints = ("expecting value", "json", "<!doctype", "<html")
            if any(h in s for h in json_hints) or any(h in (body_preview or "").lower() for h in json_hints):
                # Fail fast on JSON decode/non-JSON body
                raise

            # 3) Other transient errors (rate limits / timeouts / gateway)
            transient_hints = ("429", "rate", "timeout", "temporar", "retry", "busy", "unavailable", "gateway", "bad gateway", "502", "504")
            if any(h in s for h in transient_hints) or any(h in (body_preview or "").lower() for h in transient_hints):
                attempt += 1
                # Try to respect Retry-After if present
                delay = None
                try:
                    resp = getattr(e, "response", None)
                    if resp is not None and hasattr(resp, "headers"):
                        ra = resp.headers.get("retry-after")
                        if ra:
                            delay = float(ra)
                except Exception:
                    delay = None
                if delay is None:
                    base = min(180.0, 30.0 * (attempt + 1))
                    delay = base * random.uniform(0.8, 1.3)
                await asyncio.sleep(delay)
                continue

            # Unknown error: re-raise
            raise

    # Exhausted retries; bubble up last error
    try:
        print(f"OpenRouter adaptive call exhausted retries; last error: {last_err}")
        logfire.debug(f"OpenRouter adaptive call exhausted retries; last error: {last_err}")
    except Exception:
        pass
    raise last_err if last_err else RuntimeError("OpenRouter adaptive call failed with no exception")


async def get_next_message_anthropic(
    anthropic_client: AsyncAnthropic,
    system_messages: list[dict[str, T.Any]],
    messages: list[dict[str, T.Any]],
    model: Model,
    temperature: float,
    retry_secs: int = 15,
    max_retries: int = 200,
) -> tuple[str, ModelUsage] | None:
    retry_count = 0
    while True:
        try:
            request_id = random_string()
            start = time.time()
            logfire.debug(f"[{request_id}] calling anthropic")
            message = await anthropic_client.beta.prompt_caching.messages.create(
                system=system_messages,
                temperature=temperature,
                max_tokens=8_192,
                messages=messages,
                model=model.value,
                extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"},
                timeout=120,
            )
            took_ms = (time.time() - start) * 1000
            usage = ModelUsage(
                cache_creation_input_tokens=message.usage.cache_creation_input_tokens,
                cache_read_input_tokens=message.usage.cache_read_input_tokens,
                input_tokens=message.usage.input_tokens,
                output_tokens=message.usage.output_tokens,
            )
            logfire.debug(
                f"[{request_id}] got back anthropic, took {took_ms:.2f}, {usage}, cost_cents={Attempt.cost_cents_from_usage(model=model, usage=usage)}"
            )
            break  # Success, exit the loop
        except RateLimitError:
            logfire.debug(
                f"Rate limit error, retrying in 15 seconds ({retry_count}/{max_retries})..."
            )
            retry_count += 1
            if retry_count >= max_retries:
                # raise  # Re-raise the exception after max retries
                return None
            await asyncio.sleep(retry_secs)
        except Exception as e:
            if "invalid x-api-key" in str(e):
                return None
            logfire.debug(
                f"Other anthropic error: {str(e)}, retrying in {retry_secs} seconds ({retry_count}/{max_retries})..."
            )
            retry_count += 1
            if retry_count >= max_retries:
                # raise  # Re-raise the exception after max retries
                return None
            await asyncio.sleep(retry_secs)
    return message.content[-1].text, usage


async def get_next_message_deepseek(
    *,
    deepseek_client: AsyncOpenAI,
    messages: list[dict[str, T.Any]],
    model: Model,
    temperature: float,
    retry_secs: int = 15,
    max_retries: int = 50,
    use_baseten: bool,
) -> tuple[str, ModelUsage] | None:
    retry_count = 0
    MAX_CONTEXT_LENGTH = 65536
    params = {
        "temperature": temperature,
        "max_tokens": 8192,
        "messages": messages,
        "model": model.value,
        "timeout": 600,
        # "stream": False,
    }
    b10_str = " b10" if use_baseten else ""
    if use_baseten:
        params["model"] = "deepseek"
        params["extra_body"] = {
            "baseten": {
                "model_id": os.environ["BASETEN_R1_MODEL_ID"],
            }
        }
        params["max_tokens"] = 30_000
        params["stream"] = True
        params["stream_options"] = {"include_usage": True}
    while True:
        try:
            request_id = random_string()
            start = time.time()
            logfire.debug(f"[{request_id}] calling deepseek{b10_str}...")
            if not params.get("stream", None):
                print("calling")
                message = await deepseek_client.chat.completions.create(**params)
                cached_tokens = message.usage.prompt_tokens_details.cached_tokens
                usage = ModelUsage(
                    cache_creation_input_tokens=0,
                    cache_read_input_tokens=cached_tokens,
                    input_tokens=message.usage.prompt_tokens - cached_tokens,
                    output_tokens=message.usage.completion_tokens,
                )
                final_content = message.choices[0].message.content
            else:
                response = await deepseek_client.chat.completions.create(**params)
                final_content = ""
                usage = None
                count = 0
                async for chunk in response:
                    # print(chunk)
                    count += 1
                    if count % 100 == 0:
                        logfire.debug(f"[{request_id}] got chunk {count}")
                    if len(chunk.choices):
                        if chunk.choices[0].delta.content:
                            final_content += chunk.choices[0].delta.content
                            # print(final_content)
                    else:
                        if details := chunk.usage.prompt_tokens_details:
                            cached_tokens = details.cached_tokens or 0
                        else:
                            cached_tokens = 0
                        usage = ModelUsage(
                            cache_creation_input_tokens=0,
                            cache_read_input_tokens=cached_tokens,
                            input_tokens=chunk.usage.prompt_tokens - cached_tokens,
                            output_tokens=chunk.usage.completion_tokens,
                        )
                final_content = remove_thinking(text=final_content).strip()
                print(final_content)
                # TODO should i parse out thinking tags? probably

            took_ms = (time.time() - start) * 1000

            logfire.debug(
                f"[{request_id}] got back deepseek{b10_str}, took {took_ms:.2f}, {usage}, cost_cents={Attempt.cost_cents_from_usage(model=model, usage=usage)}"
            )
            break  # Success, exit the loop
        except Exception as e:
            error_msg = str(e)
            # Try to extract prompt tokens from error message
            if "tokens (" in error_msg:
                try:
                    prompt_tokens = int(
                        error_msg.split("(")[1].split(" in the messages")[0]
                    )
                    max_completion_tokens = MAX_CONTEXT_LENGTH - prompt_tokens
                    if max_completion_tokens <= 0:
                        return None
                    params["max_tokens"] = min(8192, max_completion_tokens)
                except (IndexError, ValueError):
                    pass  # If parsing fails, continue with normal retry logic
                    # raise e

            logfire.debug(
                f"Other deepseek{b10_str} error: {error_msg}, retrying in {retry_count} seconds ({retry_count}/{max_retries})..."
            )
            retry_count += 1
            if retry_count >= max_retries:
                return None
            await asyncio.sleep(retry_secs)
    return final_content, usage


async def get_next_message_openai(
    openai_client: AsyncOpenAI,
    messages: list[dict[str, T.Any]],
    model: Model,
    temperature: float,
    retry_secs: int = 15,
    max_retries: int = 3,
    name: str = "openai",
) -> tuple[str, ModelUsage] | None:
    retry_count = 0
    extra_params = {}
    if model not in [Model.o3_mini, Model.o1_mini, Model.o1_preview]:
        extra_params["temperature"] = temperature
    while True:
        try:
            request_id = random_string()
            start = time.time()
            logfire.debug(f"[{request_id}] calling openai")
            print(f"[{request_id}] calling openai with model {model.value}")
            message = await openai_client.chat.completions.create(
                **extra_params,
                max_completion_tokens=16384,
                messages=messages,
                model=model.value,
            )
            took_ms = (time.time() - start) * 1000
            cached_tokens = message.usage.prompt_tokens_details.cached_tokens
            usage = ModelUsage(
                cache_creation_input_tokens=0,
                cache_read_input_tokens=cached_tokens,
                input_tokens=message.usage.prompt_tokens - cached_tokens,
                output_tokens=message.usage.completion_tokens,
            )
            logfire.debug(
                f"[{request_id}] got back {name}, took {took_ms:.2f}, {usage}, cost_cents={Attempt.cost_cents_from_usage(model=model, usage=usage)}"
            )
            print(
                f"[{request_id}] got back {name}, took {took_ms:.2f}, {usage}, cost_cents={Attempt.cost_cents_from_usage(model=model, usage=usage)}"
            )
            break  # Success, exit the loop
        except Exception as e:
            logfire.debug(
                f"Other {name} error: {str(e)}, retrying in {retry_count} seconds ({retry_count}/{max_retries})..."
            )
            print(
                f"Other {name} error: {str(e)}, retrying in {retry_count} seconds ({retry_count}/{max_retries})..."
            )
            retry_count += 1
            if retry_count >= max_retries:
                # raise  # Re-raise the exception after max retries
                return None
            await asyncio.sleep(retry_secs)
    return message.choices[0].message.content, usage

async def get_next_message_xai(
    xai_client: AsyncClient,
    messages: list[dict[str, T.Any]],
    model: Model,
    temperature: float,
    retry_secs: int = 15,
    max_retries: int = 0,
    name: str = "xai",
) -> tuple[str, ModelUsage] | None:
    retry_count = 0
    extra_params = {}
    extra_params["temperature"] = temperature
    while True:
        try:
            request_id = random_string()
            start = time.time()
            logfire.debug(f"[{request_id}] calling {name}")
            print(f"[{request_id}] calling {name} with model {model.value}")
            chat = xai_client.chat.create(model=model.value, max_tokens=120000)

            print(f"[{request_id}] chat successfully created")
            
            # Convert messages to XAI format
            for msg in messages:
                if msg["role"] == "system":
                    role = system
                elif msg["role"] == "user":
                    role = user
                elif msg["role"] == "assistant":
                    role = assistant
                else:
                    raise ValueError(f"Invalid role: {msg['role']}")

                for content in msg["content"]:
                    if content["type"] == "text":
                        chat.append(role(content["text"]))
                    elif content["type"] == "image_url":
                        chat.append(role(image(content["image_url"]["url"])))
                    else:
                        raise ValueError(f"Invalid content type: {content['type']}")

            logfire.debug(f"[{request_id}] chat: {chat}")
            
            message = await chat.sample()

            print(f"[{request_id}] message: {message.content}")
            logfire.debug(f"[{request_id}] message: {message.content}")
            took_ms = (time.time() - start) * 1000
            cached_tokens = message.usage.cached_prompt_text_tokens
            usage = ModelUsage(
                cache_creation_input_tokens=0,
                cache_read_input_tokens=cached_tokens,
                input_tokens=message.usage.prompt_tokens - cached_tokens,
                output_tokens=message.usage.completion_tokens,
            )
            logfire.debug(
                f"[{request_id}] got back {name}, took {took_ms:.2f}, {usage}, reasoning tokens={message.usage.reasoning_tokens}, cost_cents={Attempt.cost_cents_from_usage(model=model, usage=usage)}"
            )
            print(
                f"[{request_id}] got back {name}, took {took_ms:.2f}, {usage}, reasoning tokens={message.usage.reasoning_tokens}, cost_cents={Attempt.cost_cents_from_usage(model=model, usage=usage)}"
            )
            break  # Success, exit the loop
        except Exception as e:
            logfire.debug(
                f"Other {name} error: {str(e)}, retrying in {retry_count} seconds ({retry_count}/{max_retries})..."
            )
            print(
                f"Other {name} error: {str(e)}, retrying in {retry_count} seconds ({retry_count}/{max_retries})..."
            )
            retry_count += 1
            if retry_count >= max_retries:
                # raise  # Re-raise the exception after max retries
                return None
            await asyncio.sleep(retry_secs)
    return message.content, usage

async def get_next_message_gemini(
    cache: gemini_caching.CachedContent,
    model: Model,
    temperature: float,
    retry_secs: int = 15,
    max_retries: int = 200,
) -> tuple[str, ModelUsage] | None:
    retry_count = 0
    while True:
        try:
            request_id = random_string()
            start = time.time()
            logfire.debug(f"[{request_id}] calling gemini")

            genai_model = genai.GenerativeModel.from_cached_content(
                cached_content=cache
            )

            response = await genai_model.generate_content_async(
                contents=[
                    genai.types.ContentDict(
                        role="user", parts=[genai.types.PartDict(text="Please answer.")]
                    )
                ],
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                    # max_output_tokens=10_000,
                ),
            )

            took_ms = (time.time() - start) * 1000
            usage = ModelUsage(
                cache_creation_input_tokens=0,
                cache_read_input_tokens=response.usage_metadata.cached_content_token_count,
                input_tokens=response.usage_metadata.prompt_token_count
                - response.usage_metadata.cached_content_token_count,
                output_tokens=response.usage_metadata.candidates_token_count,
            )
            logfire.debug(
                f"[{request_id}] got back gemini, took {took_ms:.2f}, {usage}, cost_cents={Attempt.cost_cents_from_usage(model=model, usage=usage)}"
            )
            break  # Success, exit the loop
        except Exception as e:
            if "invalid x-api-key" in str(e):
                return None
            logfire.debug(
                f"Other gemini error: {str(e)}, retrying in {retry_secs} seconds ({retry_count}/{max_retries})..."
            )
            retry_count += 1
            if retry_count >= max_retries:
                # raise  # Re-raise the exception after max retries
                return None
            await asyncio.sleep(retry_secs)
    return response.text, usage


async def get_next_messages(
    *, messages: list[dict[str, T.Any]], model: Model, temperature: float, n_times: int
) -> list[tuple[str, ModelUsage]] | None:
    if n_times <= 0:
        return []
    if model in [Model.claude_3_5_sonnet, Model.claude_3_5_haiku]:
        if model == Model.claude_3_5_haiku:
            messages = text_only_messages(messages)
        anthropic_client = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        if messages[0]["role"] == "system":
            system_messages = messages[0]["content"]
            messages = messages[1:]
        else:
            system_messages = []
        cache_control_count = 0
        for message in messages:
            content = message["content"]
            if isinstance(content, list):
                for content in message["content"]:
                    if content["type"] == "image_url":
                        content["type"] = "image"
                        content["source"] = {
                            "data": content["image_url"]["url"].replace(
                                "data:image/png;base64,", ""
                            ),
                            "media_type": "image/png",
                            "type": "base64",
                        }
                        del content["image_url"]
                    if "cache_control" in content:
                        cache_control_count = cache_control_count + 1
                        if cache_control_count >= 3:
                            del content["cache_control"]

        # remove all the caches except for on the last one
        if isinstance(messages[-1]["content"], str):
            messages[-1]["content"] = [
                {"type": "text", "text": messages[-1]["content"]}
            ]
        messages[-1]["content"][-1]["cache_control"] = {"type": "ephemeral"}

        n_messages = [
            await get_next_message_anthropic(
                anthropic_client=anthropic_client,
                system_messages=system_messages,
                messages=messages,
                model=model,
                temperature=temperature,
            ),
            *await asyncio.gather(
                *[
                    get_next_message_anthropic(
                        anthropic_client=anthropic_client,
                        system_messages=system_messages,
                        messages=messages,
                        model=model,
                        temperature=temperature,
                    )
                    for _ in range(n_times - 1)
                ]
            ),
        ]
        # filter out the Nones
        return [m for m in n_messages if m]
    elif model in [
        Model.gpt_4o,
        Model.gpt_4o_mini,
        Model.gpt_5,
        Model.o1_mini,
        Model.o1_preview,
        Model.o3_mini,
    ]:
        openai_client = AsyncOpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            timeout=1200, # 1200 seconds = 20 minutes
            max_retries=10,
        )
        if messages[0]["role"] == "system":
            messages[0]["role"] = "developer"
        if model in [Model.o1_mini, Model.o1_preview, Model.o3_mini]:
            messages = text_only_messages(messages=messages)

        n_messages = [
            await get_next_message_openai(
                openai_client=openai_client,
                messages=messages,
                model=model,
                temperature=temperature,
            ),
            *await asyncio.gather(
                *[
                    get_next_message_openai(
                        openai_client=openai_client,
                        messages=messages,
                        model=model,
                        temperature=temperature,
                    )
                    for _ in range(n_times - 1)
                ]
            ),
        ]
        return [m for m in n_messages if m]
    elif model in [Model.deep_seek_r1, Model.baseten_deepseek_r1]:
        if model == Model.deep_seek_r1:
            deepseek_client = AsyncOpenAI(
                api_key=os.environ["DEEPSEEK_API_KEY"],
                base_url="https://api.deepseek.com",
            )
            use_baseten = False
        elif model == Model.baseten_deepseek_r1:
            deepseek_client = AsyncOpenAI(
                api_key=os.environ["BASETEN_API_KEY"],
                base_url="https://bridge.baseten.co/v1/direct",
            )
            use_baseten = True
        else:
            raise ValueError(f"Invalid model: {model}")
        messages = text_only_messages(messages)

        if model == Model.deep_seek_r1:
            n_messages = [
                await get_next_message_deepseek(
                    deepseek_client=deepseek_client,
                    messages=messages,
                    model=model,
                    temperature=temperature,
                    use_baseten=use_baseten,
                ),
                *await asyncio.gather(
                    *[
                        get_next_message_deepseek(
                            deepseek_client=deepseek_client,
                            messages=messages,
                            model=model,
                            temperature=temperature,
                            use_baseten=use_baseten,
                        )
                        for _ in range(n_times - 1)
                    ]
                ),
            ]
        elif model == Model.baseten_deepseek_r1:
            n_messages = await asyncio.gather(
                *[
                    get_next_message_deepseek(
                        deepseek_client=deepseek_client,
                        messages=messages,
                        model=model,
                        temperature=temperature,
                        use_baseten=use_baseten,
                    )
                    for _ in range(n_times)
                ]
            )
        else:
            raise ValueError(f"Invalid model: {model}")
        # filter out the Nones
        return [m for m in n_messages if m]
    elif model == Model.openrouter_grok_4_fast_free:
        openrouter_client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
            timeout=1200,
            max_retries=10,
            default_headers={
                "HTTP-Referer": os.environ.get("OPENROUTER_SITE_URL", "http://localhost"),
                "X-Title": os.environ.get("OPENROUTER_APP_NAME", "arc_agi"),
                "Accept": "application/json",
                "Connection": "close",
            },
        )
        allow_images = os.environ.get("OPENROUTER_ALLOW_IMAGES", "0") == "1"
        cleaned_messages = messages if allow_images else text_only_messages(messages)
        async def _one_call():
            try:
                async with _openrouter_guard():
                    message = await _openrouter_completion_with_adaptive_max_tokens(
                        client=openrouter_client,
                        model_name=model.value,
                        messages=cleaned_messages,
                        temperature=temperature,
                        extra_body={"reasoning": {"effort": os.environ.get("OPENROUTER_REASONING_EFFORT", "high")}},
                        start_cap=int(os.environ.get("OPENROUTER_MAX_TOKENS", "200000")),
                    )
                if message.usage.prompt_tokens_details:
                    cached_tokens = message.usage.prompt_tokens_details.cached_tokens
                else:
                    cached_tokens = 0
                usage = ModelUsage(
                    cache_creation_input_tokens=0,
                    cache_read_input_tokens=cached_tokens,
                    input_tokens=message.usage.prompt_tokens - cached_tokens,
                    output_tokens=message.usage.completion_tokens,
                )
                return message.choices[0].message.content, usage
            except Exception as e:
                print(f"OpenRouter final failure in _one_call: {e}")
                logfire.debug(f"OpenRouter final failure in _one_call: {e}")
                return None
        raw_results = await asyncio.gather(*[_one_call() for _ in range(n_times)], return_exceptions=True)
        out: list[tuple[str, ModelUsage]] = []
        for r in raw_results:
            if isinstance(r, Exception):
                print(f"OpenRouter call failed with exception: {r}")
                logfire.debug(f"OpenRouter call failed with exception: {r}")
                continue
            if r:
                out.append(r)
        return out if out else None
    elif model in [Model.grok_3, Model.grok_4, Model.grok_4_fast_reasoning]:
        xai_client = AsyncClient(
            api_key=os.environ["XAI_API_KEY"],
            timeout=3600, # 3600 seconds = 60 minutes
        )

        print("Created xai client")

        n_messages = await asyncio.gather(
            *[
                get_next_message_xai(
                    xai_client=xai_client,
                    messages=messages,
                    model=model,
                    temperature=temperature,
                )
                for _ in range(n_times)
            ]
        )
        return [m for m in n_messages if m]
    elif model in [Model.gemini_1_5_pro]:
        if messages[0]["role"] == "system":
            system_messages = messages[0]["content"]
            messages = messages[1:]
        else:
            system_messages = []
        system_instruction = system_messages[0]["text"]
        gemini_contents: list[genai.types.ContentDict] = []
        for message in messages:
            if message["role"] == "assistant":
                role = "model"
            else:
                role = message["role"]
            # debug(message["content"])
            if type(message["content"]) is str:
                parts = [genai.types.PartDict(text=message["content"])]
            else:
                parts = []
                for c in message["content"]:
                    if c["type"] == "text":
                        parts.append(genai.types.PartDict(text=c["text"]))
                    elif c["type"] == "image_url":
                        image = PIL.Image.open(
                            io.BytesIO(
                                base64.b64decode(
                                    c["image_url"]["url"].replace(
                                        "data:image/png;base64,", ""
                                    )
                                )
                            )
                        )
                        if image.mode == "RGBA":
                            image = image.convert("RGB")
                        parts.append(image)
            gemini_contents.append(genai.types.ContentDict(role=role, parts=parts))

        cache = gemini_caching.CachedContent.create(
            model=model.value,
            display_name=f"{random_string(10)}-{n_times}",  # used to identify the cache
            system_instruction=system_instruction,
            contents=gemini_contents,
            ttl=timedelta(minutes=5),
        )

        n_messages = [
            *await asyncio.gather(
                *[
                    get_next_message_gemini(
                        cache=cache, model=model, temperature=temperature
                    )
                    for _ in range(n_times)
                ]
            ),
        ]
        # filter out the Nones
        return [m for m in n_messages if m]
    else:
        raise ValueError(f"Invalid model: {model}")


async def get_next_message(
    *, messages: list[dict[str, T.Any]], model: Model, temperature: float
) -> tuple[str, ModelUsage]:
    if int(os.environ.get("NO_WIFI", 0)) == 1:
        return "[[1, 2, 3], [4, 5, 6]]", ModelUsage(
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            input_tokens=0,
            output_tokens=0,
        )
    if model in [Model.claude_3_5_sonnet, Model.claude_3_5_haiku]:
        anthropic_client = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        if messages[0]["role"] == "system":
            system_messages = messages[0]["content"]
            messages = messages[1:]
        else:
            system_messages = []
        for message in messages:
            content = message["content"]
            if isinstance(content, list):
                for content in message["content"]:
                    if content["type"] == "image_url":
                        content["type"] = "image"
                        content["source"] = {
                            "data": content["image_url"]["url"].replace(
                                "data:image/png;base64,", ""
                            ),
                            "media_type": "image/png",
                            "type": "base64",
                        }
                        del content["image_url"]

        retry_count = 0
        max_retries = 12
        while True:
            try:
                message = await anthropic_client.beta.prompt_caching.messages.create(
                    system=system_messages,
                    temperature=temperature,
                    max_tokens=8_192,
                    messages=messages,
                    model=model.value,
                    extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"},
                    timeout=120,
                )
                break  # Success, exit the loop
            except RateLimitError:
                logfire.debug(
                    f"Rate limit error, retrying in 30 seconds ({retry_count}/{max_retries})..."
                )
                retry_count += 1
                if retry_count >= max_retries:
                    raise  # Re-raise the exception after max retries
                await asyncio.sleep(15)  # Wait for 30 seconds before retrying

        return message.content[-1].text, ModelUsage(
            cache_creation_input_tokens=message.usage.cache_creation_input_tokens,
            cache_read_input_tokens=message.usage.cache_read_input_tokens,
            input_tokens=message.usage.input_tokens,
            output_tokens=message.usage.output_tokens,
        )
    elif model in [Model.gpt_4o, Model.gpt_4o_mini]:
        openai_client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
        message = await openai_client.chat.completions.create(
            model=model.value,
            messages=messages,
            temperature=temperature,
            max_tokens=10_000,
        )
        cached_tokens = message.usage.prompt_tokens_details.cached_tokens
        return message.choices[0].message.content, ModelUsage(
            cache_creation_input_tokens=0,
            cache_read_input_tokens=cached_tokens,
            input_tokens=message.usage.prompt_tokens - cached_tokens,
            output_tokens=message.usage.completion_tokens,
        )
    elif model == Model.nvidia_llama_3_1_nemotron_70b_instruct:
        nvidia_client = AsyncOpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=os.environ["NVIDIA_API_KEY"],
        )
        message = await nvidia_client.chat.completions.create(
            model=model.value,
            messages=text_only_messages(messages),
            temperature=temperature,
            max_tokens=10_000,
        )
        if message.usage.prompt_tokens_details:
            cached_tokens = message.usage.prompt_tokens_details.cached_tokens
        else:
            cached_tokens = 0
        return message.choices[0].message.content, ModelUsage(
            cache_creation_input_tokens=0,
            cache_read_input_tokens=cached_tokens,
            input_tokens=message.usage.prompt_tokens - cached_tokens,
            output_tokens=message.usage.completion_tokens,
        )
    elif model == Model.groq_llama_3_2_90b_vision:
        groq_client = AsyncOpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=os.environ["GROQ_API_KEY"],
        )
        message = await groq_client.chat.completions.create(
            model=model.value,
            messages=text_only_messages(messages),
            temperature=temperature,
            max_tokens=8_192,
        )
        if message.usage.prompt_tokens_details:
            cached_tokens = message.usage.prompt_tokens_details.cached_tokens
        else:
            cached_tokens = 0
        return message.choices[0].message.content, ModelUsage(
            cache_creation_input_tokens=0,
            cache_read_input_tokens=cached_tokens,
            input_tokens=message.usage.prompt_tokens - cached_tokens,
            output_tokens=message.usage.completion_tokens,
        )
    elif model == Model.openrouter_claude_3_5_sonnet:
        openrouter_client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
        )
        message = await openrouter_client.chat.completions.create(
            model=model.value,
            messages=messages,
            temperature=temperature,
            max_tokens=10_000,
        )
        if message.usage.prompt_tokens_details:
            cached_tokens = message.usage.prompt_tokens_details.cached_tokens
        else:
            cached_tokens = 0
        return message.choices[0].message.content, ModelUsage(
            cache_creation_input_tokens=0,
            cache_read_input_tokens=cached_tokens,
            input_tokens=message.usage.prompt_tokens - cached_tokens,
            output_tokens=message.usage.completion_tokens,
        )
    elif model == Model.openrouter_o1:
        openrouter_client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
        )
        message = await openrouter_client.chat.completions.create(
            model=model.value,
            messages=messages,
            temperature=temperature,
            max_tokens=20_000,
        )
        if message.usage.prompt_tokens_details:
            cached_tokens = message.usage.prompt_tokens_details.cached_tokens
        else:
            cached_tokens = 0
        return message.choices[0].message.content, ModelUsage(
            cache_creation_input_tokens=0,
            cache_read_input_tokens=cached_tokens,
            input_tokens=message.usage.prompt_tokens - cached_tokens,
            output_tokens=message.usage.completion_tokens,
        )
    elif model == Model.openrouter_o1_mini:
        openrouter_client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
        )
        message = await openrouter_client.chat.completions.create(
            model=model.value,
            messages=messages,
            temperature=temperature,
            max_tokens=20_000,
        )
        if message.usage.prompt_tokens_details:
            cached_tokens = message.usage.prompt_tokens_details.cached_tokens
        else:
            cached_tokens = 0
        return message.choices[0].message.content, ModelUsage(
            cache_creation_input_tokens=0,
            cache_read_input_tokens=cached_tokens,
            input_tokens=message.usage.prompt_tokens - cached_tokens,
            output_tokens=message.usage.completion_tokens,
        )
    elif model == Model.openrouter_grok_4_fast_free:
        openrouter_client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
            default_headers={
                "HTTP-Referer": os.environ.get("OPENROUTER_SITE_URL", "http://localhost"),
                "X-Title": os.environ.get("OPENROUTER_APP_NAME", "arc_agi"),
                "Accept": "application/json",
                "Connection": "close",
            },
        )
        allow_images = os.environ.get("OPENROUTER_ALLOW_IMAGES", "0") == "1"
        cleaned_messages = messages if allow_images else text_only_messages(messages)
        async with _openrouter_guard():
            message = await _openrouter_completion_with_adaptive_max_tokens(
                client=openrouter_client,
                model_name=model.value,
                messages=cleaned_messages,
                temperature=temperature,
                extra_body={"reasoning": {"effort": os.environ.get("OPENROUTER_REASONING_EFFORT", "high")}},
                start_cap=int(os.environ.get("OPENROUTER_MAX_TOKENS", "200000")),
            )
        if message.usage.prompt_tokens_details:
            cached_tokens = message.usage.prompt_tokens_details.cached_tokens
        else:
            cached_tokens = 0
        return message.choices[0].message.content, ModelUsage(
            cache_creation_input_tokens=0,
            cache_read_input_tokens=cached_tokens,
            input_tokens=message.usage.prompt_tokens - cached_tokens,
            output_tokens=message.usage.completion_tokens,
        )
    elif model == [Model.azure_gpt_4o, Model.azure_gpt_4o_mini]:
        azure_client = AsyncAzureOpenAI(
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_version="2024-10-01-preview",
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        )
        message = await azure_client.chat.completions.create(
            model=model.value.replace("azure-", ""),
            messages=messages,
            temperature=temperature,
            max_tokens=10_000,
        )
        if message.usage.prompt_tokens_details:
            cached_tokens = message.usage.prompt_tokens_details.cached_tokens
        else:
            cached_tokens = 0
        return message.choices[0].message.content, ModelUsage(
            cache_creation_input_tokens=0,
            cache_read_input_tokens=cached_tokens,
            input_tokens=message.usage.prompt_tokens - cached_tokens,
            output_tokens=message.usage.completion_tokens,
        )
    elif model == Model.gemini_1_5_pro:
        if messages[0]["role"] == "system":
            system_messages = messages[0]["content"]
            messages = messages[1:]
        else:
            system_messages = []
        model = genai.GenerativeModel(
            model.value, system_instruction=system_messages[0]["text"]
        )
        gemini_contents = []
        for message in messages:
            if message["role"] == "assistant":
                role = "model"
            else:
                role = message["role"]
            # debug(message["content"])
            if type(message["content"]) is str:
                parts = [genai.types.PartDict(text=message["content"])]
            else:
                parts = []
                for c in message["content"]:
                    if c["type"] == "text":
                        parts.append(genai.types.PartDict(text=c["text"]))
                    elif c["type"] == "image_url":
                        image = PIL.Image.open(
                            io.BytesIO(
                                base64.b64decode(
                                    c["image_url"]["url"].replace(
                                        "data:image/png;base64,", ""
                                    )
                                )
                            )
                        )
                        if image.mode == "RGBA":
                            image = image.convert("RGB")
                        parts.append(image)
            gemini_contents.append(genai.types.ContentDict(role=role, parts=parts))
        response = await model.generate_content_async(
            contents=gemini_contents,
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=10_000,
            ),
        )
        return response.text, ModelUsage(
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            input_tokens=response.usage_metadata.prompt_token_count,
            output_tokens=response.usage_metadata.candidates_token_count,
        )
    else:
        raise ValueError(f"Invalid model: {model}")


noop_code = """
def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    raise NotImplementedError()
""".strip()


def clean_code(s: str) -> str:
    return s.replace("\t", " " * 4)


def parse_python_backticks(s: str) -> str:
    if s.count("```python") == 0:
        logfire.debug("NO CODE BLOCKS")
        out = s.partition("</reasoning>")[2]
        if out == "":
            return noop_code
        return clean_code(out)

    if s.count("```python") > 1:
        # print(f"MULTIPLE CODE BLOCKS\n=====\n\n{s}\n\n=====")
        for chunk in s.split("```python")[::-1]:
            if "def transform(" in chunk:
                s = "```python" + chunk
                break

    assert s.count("```python") == 1

    attempted_search = re.search(r"```python\n(.*)\n```", s, re.DOTALL | re.MULTILINE)
    if attempted_search is not None:
        return clean_code(attempted_search.group(1))

    attempted_search = re.search(r"```python\n(.*)\n`", s, re.DOTALL | re.MULTILINE)
    if attempted_search is not None:
        logfire.debug("PARSE ERROR CASE (1)")
        return clean_code(attempted_search.group(1))
    else:
        logfire.debug("PARSE ERROR CASE (2!)")

    return clean_code(s.partition("```python")[2])


def parse_2d_arrays_from_string(s: str) -> list[list[list[int]]]:
    # Regular expression pattern to match 2D arrays
    pattern = r"\[\s*(\[[^\[\]]*\](?:,\s*\[[^\[\]]*\])*\s*)\]"

    # Find all matches of the pattern in the output string
    matches = re.findall(pattern, s)

    # Process each match to create a list of 2D arrays
    arrays_list: list[list[list[int]]] = []

    for match in matches:
        # Find all inner arrays within the matched 2D array
        rows = re.findall(r"\[([^\]]*)\]", match)
        array_2d = []
        for row in rows:
            # Split the row by commas and convert to integers
            nums = [int(n.strip()) for n in row.split(",") if n.strip()]
            array_2d.append(nums)
        arrays_list.append(array_2d)

    return arrays_list
