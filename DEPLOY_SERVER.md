# MAT Thesis Demo Server Deployment

## Goal

Run the thesis-first Streamlit UI for the MAT restoration demo from a clean server checkout.

The UI entrypoint is:

```bash
streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0
```

The main UI flow is:

`Input -> Mask -> Masked Input -> Stage 1 -> Final Output`

## Recommended Branch

Use:

```bash
codex/phase3-thesis-ui-demo
```

This branch keeps the UI work isolated from the core phase branches while staying aligned with the thesis-final line.

## Server Setup

### 1. Clone or update the repository

```bash
git clone <repo-url>
cd MAT
git fetch --all
git checkout codex/phase3-thesis-ui-demo
git pull --ff-only origin codex/phase3-thesis-ui-demo
```

### 2. Create a Python environment

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Verify Streamlit is installed

```bash
python -c "import streamlit; print(streamlit.__version__)"
```

## Running the App

```bash
streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0
```

Open the app at:

```text
http://<server-host>:8501
```

## Checkpoints

For local inference mode, the server must have valid checkpoint paths for:

- `Stage 1 checkpoint`
- `Final checkpoint`

These paths are supplied in the UI under `Advanced Controls`.

If you do not want the server to run the model locally, use `Backend Mode = remote` and provide a remote inference endpoint.

## What to Verify

Before declaring the deployment successful, confirm:

- the checked out branch is `codex/phase3-thesis-ui-demo`
- `streamlit_app.py` exists
- dependencies install without error
- Streamlit starts on port `8501`
- the page loads in a browser
- if local inference is intended:
  - checkpoints exist
  - PyTorch can access the intended device

## Troubleshooting

### Streamlit starts but drawing is unavailable

Verify:

```bash
python -c "from streamlit_drawable_canvas import st_canvas; print('ok')"
```

### UI loads but local inference fails

Check:

- checkpoint file paths are correct
- the environment has a working PyTorch install
- CUDA is available if GPU inference is expected

### UI should run but model should stay elsewhere

Use remote mode:

- set `Backend Mode` to `remote`
- provide the remote endpoint in `Advanced Controls`

## Suggested Agent Prompt

```text
Deploy and run the MAT thesis demo UI from branch codex/phase3-thesis-ui-demo.

Tasks:
1. Clone or update the repo.
2. Checkout codex/phase3-thesis-ui-demo.
3. Create or reuse a Python virtual environment.
4. Install requirements from requirements.txt.
5. Verify streamlit imports successfully.
6. Run:
   streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0
7. Report:
   - current branch
   - latest commit
   - python version
   - whether install succeeded
   - whether port 8501 is serving
   - any missing checkpoint paths or inference issues
```
