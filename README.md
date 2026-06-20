# memory_proc - Processor for a personal memory base for scientific research

Convert scientific-paper PDFs into clean Markdown for downstream use by humans
and agents. Equations are preserved as LaTeX; figures are extracted into a
per-paper `figures/` folder. Built on [Marker](https://github.com/datalab-to/marker).

## Layout

```
memory_proc/
├── pdf/                      # drop input PDFs here
├── processed/<paper>/
│   ├── <paper>.md            # markdown, equations as $...$ / $$...$$
│   ├── meta.json             # source, sha256, pages, device, duration
│   └── figures/              # extracted images, referenced from the .md
├── process.py                # the processor / batch driver
└── pyproject.toml            # uv project, pinned to Python 3.12
```

## Setup

Requires [uv](https://docs.astral.sh/uv/). The project pins **Python 3.12**
(Marker's PyTorch stack does not support 3.13/3.14 yet). Works on Linux and
Windows.

```bash
uv sync                       # creates .venv and installs marker-pdf
```

By default `pyproject.toml` pins the **CPU-only** PyTorch build (small, no CUDA).
That is the right choice on machines without an NVIDIA GPU. To run on an NVIDIA
GPU you must switch to the CUDA build — see [Hardware & GPU](#hardware--gpu).

## Usage

```bash
uv run process.py                 # convert every new PDF in pdf/
uv run process.py --file NAME     # convert one PDF (name in pdf/ or a path)
uv run process.py --force         # reconvert even if already processed
uv run process.py --llm           # optional LLM cleanup pass for equations/tables
```

Re-runs are cheap: a paper is skipped when its SHA-256 matches the hash in its
`meta.json`. Use `--force` to override.

### LLM cleanup pass (optional)

`--llm` enables Marker's built-in LLM mode (via Claude) to improve tricky
equations and tables. It needs an API key:

```bash
cp .env.example .env            # then fill in ANTHROPIC_API_KEY
export ANTHROPIC_API_KEY=sk-ant-...
uv run process.py --llm
```

## Hardware & GPU

The processor auto-detects the device at startup (`detect_device()` →
`torch.cuda.is_available()`). **CPU is impractically slow**: a single 12-page
paper took ~6.2 hours on a CPU-only machine. On an NVIDIA GPU the same paper
finishes in under a minute, with no code changes. First run also downloads
~3 GB of model weights.

### Will it use my GPU?

`torch.cuda.is_available()` returns `False` — so the run falls back to CPU —
whenever **either** of these is true:

1. The installed PyTorch is the **CPU-only build** (`torch==…+cpu`). This build
   has no CUDA support compiled in and can never see a GPU, even on a GPU box.
2. **No NVIDIA driver/GPU is visible** (`nvidia-smi` fails).

Quick check on any machine:

```bash
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

`…+cpu` or `False` means you are running on CPU. Note: **NVIDIA only.** AMD
integrated/discrete GPUs (ROCm) and Apple Silicon are not a supported
acceleration path here — treat those machines as CPU-only.

### Enabling an NVIDIA GPU

Required on **both Linux and Windows**: a working NVIDIA driver (`nvidia-smi`
must list your GPU) and the **CUDA** PyTorch build instead of the default
CPU one. You do **not** install the CUDA toolkit separately — the PyTorch CUDA
wheels bundle everything they need.

Switch this project to the CUDA build by removing the CPU pin from
`pyproject.toml`:

```toml
# delete these two blocks on a GPU machine:
[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[tool.uv.sources]
torch = [{ index = "pytorch-cpu" }]
torchvision = [{ index = "pytorch-cpu" }]
```

Then re-resolve and verify:

```bash
uv sync
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# expect e.g. 2.x.y+cu124  True
```

`uv sync` will pull the CUDA wheels (~3–5 GB of `nvidia-*` packages — large, but
that is normal). After this, `detect_device()` reports `cuda` and Marker uses the
GPU automatically.

#### Linux notes

- Install the proprietary NVIDIA driver from your distro (e.g. `nvidia` /
  `nvidia-dkms` on Arch, `nvidia-driver-xxx` on Debian/Ubuntu), then reboot and
  confirm `nvidia-smi` works.
- If you keep the CPU pin and just want to avoid editing it per machine, you can
  instead force the CUDA index for one sync without changing the file — but
  editing `pyproject.toml` on the GPU host is simpler and reproducible.

#### Windows notes

- Install [uv](https://docs.astral.sh/uv/) and run the same `uv` commands in
  **PowerShell**. Paths and commands above are identical except shell syntax.
- Install the NVIDIA driver from nvidia.com (or GeForce Experience) and confirm
  `nvidia-smi` runs in PowerShell.
- `MODEL_CACHE_DIR` in `.env` should use a Windows path if you relocate the cache,
  e.g. `MODEL_CACHE_DIR=D:\datalab\models`. The default (`%LOCALAPPDATA%`) is
  usually fine on Windows.
- WSL2 also works and behaves like the Linux instructions, provided the Windows
  NVIDIA driver is installed (no separate driver inside WSL).

## Model cache location

Weights default to `~/.cache/datalab`. If the home partition is full, downloads
fail mid-move with `No space left on device` (the partial then blocks retries
with a confusing "already exists" error). Point the cache at a roomier disk via
`MODEL_CACHE_DIR` in `.env` (already set to `/var/tmp/datalab/models` here).

If a download ever stalls or is interrupted, delete the partial model dir before
retrying, e.g. `rm -rf $MODEL_CACHE_DIR/models/<model>/<version>`.
