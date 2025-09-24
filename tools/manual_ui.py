#!/usr/bin/env python3
# A lightweight local web UI (stdlib only) to review failure prompts and paste manual responses.
# Run: python tools/manual_ui.py --port 8765

import argparse
import json
import os
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Optional, List, Tuple
from urllib.parse import urlparse, parse_qs

FAILURES_DIR = Path("failures")
MANUAL_DIR = Path("manual_responses")

HTML_BASE = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>ARC-AGI Manual UI</title>
  <style>
    body {{ font-family: sans-serif; margin: 24px; }}
    .row {{ margin: 8px 0; }}
    .hash {{ font-family: monospace; }}
    textarea {{ width: 100%; height: 320px; font-family: monospace; }}
    pre {{ white-space: pre-wrap; }}
    .btn {{ padding: 6px 10px; margin-right: 6px; }}
    .warn {{ color: #a00; }}
    .ok {{ color: #080; }}
    .mono {{ font-family: monospace; }}
    table {{ border-collapse: collapse; width: 100%; }}
    td, th {{ border-bottom: 1px solid #ddd; padding: 6px; text-align: left; }}
  </style>
  <script>
    function copyPrompt(id) {{
      const el = document.getElementById(id);
      navigator.clipboard.writeText(el.innerText).then(() => {{
        alert('Prompt copied to clipboard');
      }}).catch(err => alert('Copy failed: ' + err));
    }}
    function lockAndCopy(hash, promptId) {{
      // Lock on server, then copy to clipboard. Keep UX in a single click.
      fetch('/lock?hash=' + encodeURIComponent(hash))
        .then(() => {{
          copyPrompt(promptId);
        }})
        .catch(() => {{
          // Even if lock fails, still attempt copy
          copyPrompt(promptId);
        }});
    }}
  </script>
</head>
<body>
  <div class="row"><a href="/">Home</a></div>
  {content}
</body>
</html>
"""


def _lock_path(prompt_hash: str) -> Path:
    return MANUAL_DIR / f"{prompt_hash}.lock"


def _is_locked(prompt_hash: str) -> Tuple[bool, Optional[float]]:
    lp = _lock_path(prompt_hash)
    if not lp.exists():
        return False, None
    try:
        data = json.loads(lp.read_text(encoding="utf-8"))
        exp = float(data.get("expires_at", 0))
        if time.time() < exp:
            return True, exp
        else:
            try:
                lp.unlink(missing_ok=True)
            except Exception:
                pass
            return False, None
    except Exception:
        return False, None


def _lock(prompt_hash: str, challenge_id: str, minutes: int) -> None:
    MANUAL_DIR.mkdir(parents=True, exist_ok=True)
    expires_at = time.time() + minutes * 60
    payload = {
        "prompt_hash": prompt_hash,
        "challenge_id": challenge_id,
        "locked_at": time.time(),
        "expires_at": expires_at,
    }
    _lock_path(prompt_hash).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _unlock(prompt_hash: str) -> None:
    try:
        _lock_path(prompt_hash).unlink(missing_ok=True)
    except Exception:
        pass


def _iter_failures() -> List[Tuple[str, str, Path]]:
    out: List[Tuple[str, str, Path]] = []
    if not FAILURES_DIR.exists():
        return out
    for cid in sorted([d for d in FAILURES_DIR.iterdir() if d.is_dir()]):
        for txt in sorted(cid.glob("*.txt")):
            out.append((cid.name, txt.stem, txt))
    return out


def _render_index() -> str:
    rows = []
    rows.append("<h1>Pending Failures</h1>")
    rows.append("<p>Click an item to open. You can work in multiple tabs; use locks to avoid overlap.</p>")
    rows.append("<table><tr><th>challenge_id</th><th>prompt_hash</th><th>status</th><th>open</th></tr>")
    any_rows = False
    for challenge_id, prompt_hash, txt_path in _iter_failures():
        manual_path = MANUAL_DIR / f"{prompt_hash}.txt"
        has_manual = manual_path.exists()
        locked, exp = _is_locked(prompt_hash)
        status = []
        status.append("manual=YES" if has_manual else "manual=NO")
        if locked:
            status.append(f"locked_until={time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(exp))}")
        if not has_manual:
            any_rows = True
        rows.append(
            f"<tr><td>{challenge_id}</td><td class='mono'>{prompt_hash}</td><td>{' '.join(status)}</td>"
            f"<td><a href='/item?hash={prompt_hash}'>open</a></td></tr>"
        )
    rows.append("</table>")
    if not any_rows:
        rows.append("<p class='ok'>No pending manual items.</p>")
    return HTML_BASE.format(content="\n".join(rows))


def _load_prompt_text(prompt_hash: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    # returns (challenge_id, prompt_text, prompt_path)
    if not FAILURES_DIR.exists():
        return None, None, None
    for cid in [d for d in FAILURES_DIR.iterdir() if d.is_dir()]:
        path = cid / f"{prompt_hash}.txt"
        if path.exists():
            try:
                return cid.name, path.read_text(encoding="utf-8"), str(path)
            except Exception:
                return cid.name, None, str(path)
    return None, None, None


def _extract_user_only(prompt_hash: str, full_text: Optional[str]) -> str:
    # Prefer JSON messages to reconstruct only the user content.
    if FAILURES_DIR.exists():
        for cid in [d for d in FAILURES_DIR.iterdir() if d.is_dir()]:
            jpath = cid / f"{prompt_hash}.json"
            if jpath.exists():
                try:
                    msgs = json.loads(jpath.read_text(encoding='utf-8'))
                    last_user = None
                    for m in msgs:
                        if isinstance(m, dict) and m.get('role') == 'user':
                            last_user = m
                    if last_user:
                        content = last_user.get('content', '')
                        if isinstance(content, str):
                            return content.strip()
                        parts: List[str] = []
                        if isinstance(content, list):
                            for c in content:
                                if isinstance(c, dict) and c.get('type') == 'text':
                                    parts.append(c.get('text', ''))
                        if parts:
                            return "\n\n".join(parts).strip()
                except Exception:
                    pass
    # Fallback: parse from full_text between [USER] and the next role marker
    if full_text:
        s = full_text
        idx = s.rfind('[USER]')
        if idx != -1:
            s2 = s[idx+len('[USER]'):]
            # find next role marker like [ASSISTANT] or [SYSTEM]
            next_idx = s2.find('\n[')
            if next_idx != -1:
                return s2[:next_idx].strip()
            return s2.strip()
    return (full_text or '').strip()


def _render_item(prompt_hash: str) -> str:
    challenge_id, text, path_str = _load_prompt_text(prompt_hash)
    if not challenge_id:
        return HTML_BASE.format(content=f"<p class='warn'>Prompt not found: {prompt_hash}</p>")

    manual_path = MANUAL_DIR / f"{prompt_hash}.txt"
    has_manual = manual_path.exists()
    locked, exp = _is_locked(prompt_hash)

    rows = []
    rows.append(f"<h2>Prompt <span class='hash'>{prompt_hash}</span></h2>")
    rows.append(f"<div class='row'>challenge_id: <b>{challenge_id}</b></div>")
    rows.append(f"<div class='row'>source: <span class='mono'>{path_str}</span></div>")
    status_bits = []
    status_bits.append("manual=YES" if has_manual else "manual=NO")
    if locked:
        status_bits.append(f"locked_until={time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(exp))}")
    rows.append(f"<div class='row'>status: {' '.join(status_bits)}</div>")

    if not text:
        rows.append("<p class='warn'>Failed to load prompt text.</p>")
    else:
        rows.append(
            "<div class='row'>"
            f"<button class='btn' onclick=\"lockAndCopy('{prompt_hash}','prompt_user')\">Lock & Copy</button>"
            f"<button class='btn' onclick=\"copyPrompt('prompt_user')\">Copy Prompt</button>"
            f"<a class='btn' href='/lock?hash={prompt_hash}'><button class='btn'>Lock (60m)</button></a>"
            f"<a class='btn' href='/unlock?hash={prompt_hash}'><button class='btn'>Unlock</button></a>"
            "</div>"
        )
        # Hidden user-only content for accurate copying
        user_only = _extract_user_only(prompt_hash, text)
        rows.append("<pre id='prompt_user' style='display:none'>" + (user_only.replace("<", "&lt;").replace(">", "&gt;").strip()) + "</pre>")
        rows.append("<pre id='prompt'>" + (text.replace("<", "&lt;").replace(">", "&gt;") ) + "</pre>")
        # If wire.json exists, add a button and hidden block for copying exact JSON payload sent (when available)
        wire_json_path = None
        for cid in [d for d in FAILURES_DIR.iterdir() if d.is_dir()]:
            p = cid / f"{prompt_hash}.wire.json"
            if p.exists():
                wire_json_path = p
                break
        if wire_json_path:
            try:
                wire_text = wire_json_path.read_text(encoding='utf-8')
                rows.append(
                    "<div class='row'>"
                    f"<button class='btn' onclick=\"copyPrompt('wire_json')\">Copy Wire JSON</button>"
                    f"<span class='mono'>{wire_json_path}</span>"
                    "</div>"
                )
                rows.append("<pre id='wire_json' style='display:none'>" + (wire_text.replace("<", "&lt;").replace(">", "&gt;") ) + "</pre>")
            except Exception:
                pass

    rows.append("<h3>Paste assistant response</h3>")
    rows.append(
        f"<form method='POST' action='/submit'>"
        f"<input type='hidden' name='hash' value='{prompt_hash}'/>"
        f"<textarea name='response' placeholder='Paste the assistant response here'></textarea>"
        f"<div class='row'><button class='btn' type='submit'>Save</button></div>"
        f"</form>"
    )

    rows.append("<div class='row'><a href='/next?hash=" + prompt_hash + "'>Next pending</a></div>")
    return HTML_BASE.format(content="\n".join(rows))


def _next_pending_after(current_hash: str) -> Optional[str]:
    seen_current = (current_hash == "")
    for cid, ph, _ in _iter_failures():
        if not seen_current:
            if ph == current_hash:
                seen_current = True
            continue
        manual = MANUAL_DIR / f"{ph}.txt"
        if not manual.exists():
            return ph
    return None


class Handler(BaseHTTPRequestHandler):
    server_version = "ManualUI/0.1"

    def do_GET(self):
        try:
            parsed = urlparse(self.path)
            if parsed.path in ("/", "/index"):
                self._ok_html(_render_index())
                return
            if parsed.path == "/item":
                q = parse_qs(parsed.query)
                h = q.get("hash", [""])[0]
                self._ok_html(_render_item(h))
                return
            if parsed.path == "/lock":
                q = parse_qs(parsed.query)
                h = q.get("hash", [""])[0]
                cid, _, _ = _load_prompt_text(h)
                if cid:
                    _lock(h, cid, 60)
                self._redirect(f"/item?hash={h}")
                return
            if parsed.path == "/unlock":
                q = parse_qs(parsed.query)
                h = q.get("hash", [""])[0]
                _unlock(h)
                self._redirect(f"/item?hash={h}")
                return
            if parsed.path == "/next":
                q = parse_qs(parsed.query)
                h = q.get("hash", [""])[0]
                # find next pending without manual file
                found_next = _next_pending_after(h)
                if not found_next:
                    self._redirect("/")
                else:
                    self._redirect(f"/item?hash={found_next}")
                return
            self._not_found()
        except Exception as e:
            self._err(500, f"Internal error: {e}")

    def do_POST(self):
        try:
            parsed = urlparse(self.path)
            length = int(self.headers.get('Content-Length', '0'))
            body = self.rfile.read(length).decode('utf-8', errors='ignore')
            fields = {}
            for kv in body.split('&'):
                if '=' in kv:
                    k, v = kv.split('=', 1)
                    fields[k] = self._urldecode(v)
            if parsed.path == "/submit":
                prompt_hash = fields.get('hash', '')
                response = fields.get('response', '').strip()
                if not prompt_hash:
                    self._err(400, 'Missing hash')
                    return
                if not response:
                    self._err(400, 'Empty response')
                    return
                MANUAL_DIR.mkdir(parents=True, exist_ok=True)
                out = MANUAL_DIR / f"{prompt_hash}.txt"
                out.write_text(response, encoding='utf-8')
                # keep lock (user might continue editing), but refresh expiry to +60m
                cid, _, _ = _load_prompt_text(prompt_hash)
                if cid:
                    _lock(prompt_hash, cid, 60)
                # Auto-advance to next pending if available
                nxt = _next_pending_after(prompt_hash)
                if nxt:
                    self._redirect(f"/item?hash={nxt}")
                else:
                    self._redirect("/")
                return
            self._not_found()
        except Exception as e:
            self._err(500, f"Internal error: {e}")

    # helpers
    def _ok_html(self, html: str):
        data = html.encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _redirect(self, to: str):
        self.send_response(302)
        self.send_header('Location', to)
        self.end_headers()

    def _not_found(self):
        self._err(404, 'Not found')

    def _err(self, code: int, msg: str):
        html = HTML_BASE.format(content=f"<p class='warn'>{code} {msg}</p>")
        data = html.encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _urldecode(self, s: str) -> str:
        return (
            s.replace('+', ' ')
             .replace('%0D', '\r')
             .replace('%0A', '\n')
             .replace('%25', '%')
             .replace('%26', '&')
             .replace('%3D', '=')
        )


def main():
    parser = argparse.ArgumentParser(description='Local manual UI for ARC-AGI failures')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()

    MANUAL_DIR.mkdir(parents=True, exist_ok=True)

    httpd = HTTPServer((args.host, args.port), Handler)
    print(f"Manual UI running at http://{args.host}:{args.port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == '__main__':
    main()