# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

Repository overview
- Purpose: ARC-AGI solver combining DreamCoder-style evolving library with LLM-generated Python programs; optional LPN (Latent Program Network) integration.
- Two main surfaces:
  - CLI/Batch: run solvers to produce submissions and evaluate.
  - HTTP API: FastAPI endpoints wrapping core solver utilities.

Environment and dependencies
- Python: Use 3.10 or 3.11.
  - pyproject.toml enforces ">=3.10,<3.11.0" and Dockerfile uses python:3.10.
  - Note: the README mentions 3.11, but the constraints pin <3.11.
- Virtual environment
  - PowerShell: python -m venv .venv; .\.venv\Scripts\Activate.ps1
  - bash/zsh: python -m venv .venv; source .venv/bin/activate
- Install dependencies (pip)
  - pip install -U -r requirements.txt
- Optional: Poetry (matches Dockerfile behavior)
  - poetry install --no-root --only main

Configuration and secrets
- Create a .env in the project root (python-dotenv is included). Common keys:
  - OPENAI_API_KEY, ANTHROPIC_API_KEY, LOGFIRE_TOKEN, XAI_API_KEY
- Optional LPN integration requires:
  - LPN_ARTIFACT_PATH (Weights & Biases artifact identifier)

Run commands
- HTTP API (development)
  - fastapi run src/app.py --host 0.0.0.0 --port 8000
  - Alternative (uvicorn): python -m uvicorn src.app:app --reload --host 0.0.0.0 --port 8000
- CLI solvers (batch)
  - ARC-AGI-1 public eval set: python -m src.submission -v1
  - ARC-AGI-2 default eval set: python -m src.submission
  - Specify dataset directory (JSON format like ARC-AGI-2 public eval):
    - python -m src.submission -p <path-to-dataset-dir>
  - Optional LPN mode (requires LPN_ARTIFACT_PATH):
    - python -m src.submission -l on -p <path>  # any string enables LPN mode
  - Experimental main loop variant: python -m src.main [ -v1 | -p <path> | -l on ]

Linting
- Ruff (configured in pyproject.toml):
  - ruff check .

Docker (optional)
- Build: docker build -t arc-agi .
- Run: docker run -p 8000:80 arc-agi
  - The image uses FastAPI’s runner to serve src/app.py on port 80.

High-level architecture
- Core solving pipeline
  - src.logic: orchestrates solving a challenge; coordinates LLM prompting, program execution, and library updates.
  - src.trees.*: strategies/trees defining search orchestration.
  - src.models: Library for learned primitives/programs; Attempt and related data structures.
  - src.run_python: sandboxed execution of generated transforms.
  - src.data: loads/builds challenge datasets (ARC-AGI-1 and 2). 
- HTTP surface
  - src/app.py exposes endpoints:
    - POST /solve_challenge: async solve for a challenge and tree
    - POST /run_python_transform: run python transforms
    - POST /llm_responses_to_grid_list: convert LLM outputs to grids
- Optional LPN integration
  - lpn/ provides Latent Program Network code; submission and main can bind an LPN from a W&B artifact when -l is set and LPN_ARTIFACT_PATH is provided.
- Concurrency
  - The CLI flows leverage asyncio to parallelize solving batches of challenges.

Notes
- Prefer Python 3.10 or 3.11 to match toolchain constraints.
- Use .env for provider keys; do not hardcode secrets.
- For LPN training/experimentation details, see lpn/README.md.
