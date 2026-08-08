#!/usr/bin/env python3
"""Convert scientific-paper PDFs to Markdown using Marker.

Scans ./pdf for PDFs, converts each to Markdown (equations as LaTeX, figures
extracted to a per-paper figures/ folder), and writes results under ./processed.

Re-runs are cheap: a paper is skipped when its SHA-256 already matches the hash
recorded in its meta.json. Use --force to reconvert anyway.

Usage:
    uv run process.py                 # convert all new PDFs in pdf/
    uv run process.py --file NAME     # convert a single PDF (name or path)
    uv run process.py --force         # reconvert even if already done
    uv run process.py --llm           # enable Marker's LLM cleanup pass
    uv run process.py -v              # extra diagnostics (GPU mem, tracebacks)
    uv run process.py --no-equations  # skip LaTeX OCR (helps on 6–8 GiB GPUs)
    uv run process.py --isolate       # one fresh process per PDF (survives CUDA aborts)
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

VERBOSE = False


def log(msg: str, *, verbose_only: bool = False) -> None:
    if verbose_only and not VERBOSE:
        return
    print(msg, flush=True)


def _configure_cpu_threads() -> None:
    """Cap BLAS/OpenMP threads so pdftext workers and PyTorch don't oversubscribe."""
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    # Surya spawns nested pools unless this is set (Marker's convert CLI does the same).
    os.environ.setdefault("IN_STREAMLIT", "true")
    os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
    os.environ.setdefault("GLOG_minloglevel", "2")
    # expandable_segments is Linux-only; on Windows it only emits a PyTorch warning.
    if sys.platform != "win32":
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    n = os.cpu_count() or 4
    if sys.platform == "win32":
        # Windows WDDM + heavy GPU load: keep CPU threads low to avoid watchdog BSODs.
        threads = str(max(2, min(4, n // 6)))
    else:
        threads = str(max(2, n // 4))
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, threads)
    os.environ.setdefault("MKL_DYNAMIC", "FALSE")
    os.environ.setdefault("OMP_DYNAMIC", "FALSE")


_configure_cpu_threads()

ROOT = Path(__file__).resolve().parent
PDF_DIR = ROOT / "pdf"
OUT_DIR = ROOT / "processed"

# Load .env (if present) before any marker/surya import so settings like
# MODEL_CACHE_DIR and ANTHROPIC_API_KEY are picked up. MODEL_CACHE_DIR matters
# on hosts where the home partition is full: point it at a roomier disk so the
# ~3 GB of model weights don't fail to download with "No space left on device".
try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

# Image references in Marker markdown look like ![alt](name.jpeg) or with a path.
_IMG_RE = re.compile(r"(!\[[^\]]*\]\()([^)]+)(\))")

# Marker leaks RSS when a PdfConverter is reused across documents; recycle it often.
RECYCLE_CONVERTER_EVERY = 1

# Shorter GPU kernels + less CPU contention reduce CLOCK_WATCHDOG_TIMEOUT / TDR on
# Windows, where the display driver shares the GPU with CUDA (WDDM).
# recognition_batch_size kept low — larger values have triggered IndexKernel OOB
# assertions in Surya on some technical PDFs.
_BATCH_KEYS = (
    "recognition_batch_size",
    "layout_batch_size",
    "detection_batch_size",
    "ocr_error_batch_size",
    "table_rec_batch_size",
    "equation_batch_size",
)

# Normal Windows CUDA tuning (9+ GiB VRAM).
_WINDOWS_CUDA_BATCH: dict[str, int] = dict(
    zip(_BATCH_KEYS, (4, 4, 4, 4, 4, 4), strict=True)
)

# Laptop GPUs (≤8 GiB): batch=1 avoids Surya equation/OCR illegal-memory-access crashes.
_LOW_VRAM_CUDA_BATCH: dict[str, int] = dict(zip(_BATCH_KEYS, (1, 1, 1, 1, 1, 1), strict=True))
LOW_VRAM_GIB_THRESHOLD = 8.0

# EquationProcessor runs Surya recognition with long tiled sequences — heaviest VRAM stage.
_EQUATION_PROCESSORS = frozenset(
    {
        "marker.processors.equation.EquationProcessor",
        "marker.processors.llm.llm_equation.LLMEquationProcessor",
        "marker.processors.llm.llm_mathblock.LLMMathBlockProcessor",
    }
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def format_size(path: Path) -> str:
    n = path.stat().st_size
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024
    return f"{n:.1f} GB"


def slugify(name: str) -> str:
    """Filesystem-friendly stem: keep it readable, drop trouble characters."""
    stem = re.sub(r"\s+", "_", name.strip())
    stem = re.sub(r"[^A-Za-z0-9._-]", "", stem)
    return stem.strip("._-") or "paper"


def detect_device() -> str:
    try:
        import torch
    except Exception:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def torch_runtime_info(device: str) -> list[str]:
    lines: list[str] = []
    try:
        import torch

        lines.append(f"PyTorch {torch.__version__}")
        if device == "cuda" and torch.cuda.is_available():
            idx = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(idx)
            lines.append(f"GPU [{idx}]: {props.name} ({props.total_memory / (1 << 30):.1f} GiB VRAM)")
            lines.append(f"CUDA driver/runtime: {torch.version.cuda}")
    except Exception as exc:
        lines.append(f"(could not query torch runtime: {exc})")
    return lines


def gpu_total_vram_gib() -> float | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        return props.total_memory / (1 << 30)
    except Exception:
        return None


def resolve_low_vram(force: bool, device: str) -> bool:
    if force or os.environ.get("MARKER_LOW_VRAM", "").lower() in ("1", "true", "yes"):
        return True
    if device != "cuda":
        return False
    vram = gpu_total_vram_gib()
    return vram is not None and vram <= LOW_VRAM_GIB_THRESHOLD


def gpu_memory_snapshot() -> str | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        idx = torch.cuda.current_device()
        alloc = torch.cuda.memory_allocated(idx) / (1 << 20)
        reserved = torch.cuda.memory_reserved(idx) / (1 << 20)
        return f"GPU mem: {alloc:.0f} MiB allocated, {reserved:.0f} MiB reserved"
    except Exception as exc:
        return f"GPU mem: unavailable ({exc})"


def is_fatal_cuda_error(exc: BaseException) -> bool:
    """True when the exception (or its cause chain) indicates a broken CUDA context."""
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        msg = str(cur).lower()
        name = type(cur).__name__.lower()
        if "cuda" in msg or "cuda" in name:
            return True
        if "index out of bounds" in msg:
            return True
        cur = cur.__cause__ or cur.__context__
    return False


def cuda_context_healthy() -> bool:
    try:
        import torch

        if not torch.cuda.is_available():
            return True
        torch.cuda.synchronize()
        return True
    except Exception:
        return False


def _pdftext_workers() -> int:
    override = os.environ.get("MARKER_PDFTEXT_WORKERS")
    if override is not None:
        return max(1, int(override))
    if sys.platform == "win32":
        return 2
    try:
        import psutil

        return max(1, psutil.cpu_count(logical=False) or 1)
    except ImportError:
        return max(1, (os.cpu_count() or 4) // 2)


def _configure_torch_threads() -> None:
    try:
        import torch

        n = os.cpu_count() or 4
        threads = max(2, min(4, n // 6)) if sys.platform == "win32" else max(2, n // 4)
        torch.set_num_threads(threads)
        log(f"  torch CPU threads: {threads}", verbose_only=True)
    except Exception:
        pass


def release_gpu_memory(*, cooldown: bool = False, device: str = "", label: str = "") -> None:
    prefix = f"  [{label}] " if label else "  "
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            snap = gpu_memory_snapshot()
            if snap:
                log(f"{prefix}released GPU memory — {snap}", verbose_only=not label)
    except Exception as exc:
        log(f"{prefix}GPU cleanup failed: {exc}")
    if cooldown and sys.platform == "win32" and device == "cuda":
        sec = float(os.environ.get("MARKER_GPU_COOLDOWN_SEC", "3"))
        if sec > 0:
            log(f"  GPU cooldown: sleeping {sec}s (MARKER_GPU_COOLDOWN_SEC)")
            time.sleep(sec)


def marker_performance_config(*, low_vram: bool = False) -> dict:
    """GPU batch sizes and CPU extraction workers (mirrors Marker's convert CLI)."""
    perf: dict = {"disable_tqdm": True, "pdftext_workers": _pdftext_workers()}

    try:
        import torch

        if not torch.cuda.is_available():
            return perf
    except Exception:
        return perf

    if low_vram:
        perf.update(_LOW_VRAM_CUDA_BATCH)
        return perf

    if sys.platform == "win32":
        perf.update(_WINDOWS_CUDA_BATCH)
        return perf

    try:
        from marker.utils.batch import get_batch_sizes_worker_counts
        from marker.utils.gpu import GPUManager

        with GPUManager(0) as gpu_manager:
            batch_sizes, _ = get_batch_sizes_worker_counts(gpu_manager, 7)
        perf.update(batch_sizes)
    except Exception:
        pass
    return perf


def processor_list_for_options(no_equations: bool) -> list[str] | None:
    """Return explicit Marker processor paths, or None for PdfConverter defaults."""
    if not no_equations:
        return None
    from marker.converters.pdf import PdfConverter
    from marker.util import classes_to_strings

    return [
        p
        for p in classes_to_strings(list(PdfConverter.default_processors))
        if p not in _EQUATION_PROCESSORS
    ]


def log_perf_config(perf: dict, *, profile: str = "") -> None:
    workers = perf.get("pdftext_workers")
    log(f"  pdftext workers: {workers}")
    if profile:
        log(f"  perf profile: {profile}")
    batches = {k: perf[k] for k in _BATCH_KEYS if k in perf}
    if batches:
        parts = ", ".join(f"{k}={v}" for k, v in batches.items())
        log(f"  GPU batch sizes: {parts}")
    elif perf.get("pdftext_workers") is not None:
        log("  GPU batch sizes: (defaults — no CUDA or auto-detect unavailable)")


def already_done(out_paper_dir: Path, pdf_hash: str) -> bool:
    meta = out_paper_dir / "meta.json"
    if not meta.exists():
        return False
    try:
        data = json.loads(meta.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    return data.get("sha256") == pdf_hash


def build_converter(
    use_llm: bool,
    artifact_dict: dict,
    perf_config: dict | None = None,
    *,
    no_equations: bool = False,
):
    """Create a Marker PdfConverter using a shared model artifact dict."""
    from marker.config.parser import ConfigParser
    from marker.converters.pdf import PdfConverter

    config: dict = {"output_format": "markdown"}
    config.update(perf_config if perf_config is not None else marker_performance_config())
    if use_llm:
        config["use_llm"] = True

    parser = ConfigParser(config)
    processor_list = processor_list_for_options(no_equations)
    if processor_list is None:
        processor_list = parser.get_processors()
    return PdfConverter(
        config=parser.generate_config_dict(),
        artifact_dict=artifact_dict,
        processor_list=processor_list,
        renderer=parser.get_renderer(),
        llm_service=parser.get_llm_service() if use_llm else None,
    )


def log_cuda_failure_hints(
    exc: BaseException,
    pdf_name: str,
    *,
    no_equations: bool,
    low_vram: bool,
) -> None:
    tb = traceback.format_exc()
    tb_lower = tb.lower()
    if "equation" in tb_lower or "equationprocessor" in tb_lower:
        log(
            "    stage: EquationProcessor (Surya LaTeX OCR on math blocks) — "
            "the heaviest step; often fails on 6–8 GiB GPUs."
        )
    if "illegal memory access" in str(exc).lower():
        log(
            "    note: cudaErrorIllegalAddress usually means VRAM pressure or a bad "
            "equation/OCR batch, not a corrupt PDF."
        )
    log(f'    try: uv run process.py --file "{pdf_name}" --no-equations')
    if not low_vram:
        log(f'    try: uv run process.py --file "{pdf_name}" --low-vram')
    log("    try: uv run process.py --isolate   # fresh CUDA context per PDF in a batch")
    if not VERBOSE:
        log("    tip: add -v for full tracebacks.")


def rewrite_image_paths(markdown: str, figures_subdir: str = "figures") -> str:
    """Point every image reference at the figures/ subfolder."""

    def repl(m: re.Match) -> str:
        target = m.group(2)
        # leave absolute URLs alone
        if target.startswith(("http://", "https://", "data:")):
            return m.group(0)
        return f"{m.group(1)}{figures_subdir}/{Path(target).name}{m.group(3)}"

    return _IMG_RE.sub(repl, markdown)


def save_result(rendered, out_paper_dir: Path, stem: str) -> int:
    """Write markdown + figures. Returns the number of figures written."""
    from marker.output import text_from_rendered

    figures_dir = out_paper_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    text, _ext, images = text_from_rendered(rendered)

    count = 0
    for img_name, img in (images or {}).items():
        dest = figures_dir / Path(img_name).name
        img.save(dest)
        count += 1

    if count == 0:
        # nothing extracted; drop the empty dir so the layout stays clean
        try:
            figures_dir.rmdir()
        except OSError:
            pass

    md = rewrite_image_paths(text)
    md_path = out_paper_dir / f"{stem}.md"
    md_path.write_text(md, encoding="utf-8")
    log(f"    wrote {md_path.name} ({len(md):,} chars)", verbose_only=True)
    return count


def convert_one(
    pdf: Path,
    converter,
    use_llm: bool,
    force: bool,
    device: str,
    *,
    index: int,
    total: int,
    no_equations: bool,
) -> str:
    stem = slugify(pdf.stem)
    out_paper_dir = OUT_DIR / stem
    pdf_hash = sha256(pdf)

    log(f"\n[{index}/{total}] {pdf.name} ({format_size(pdf)})")
    log(f"    output: processed/{stem}/", verbose_only=True)
    log(f"    sha256: {pdf_hash[:16]}…", verbose_only=True)

    if not force and already_done(out_paper_dir, pdf_hash):
        log("    skip — already processed (hash matches meta.json; use --force to redo)")
        return "skipped"

    snap = gpu_memory_snapshot()
    if snap:
        log(f"    {snap}", verbose_only=True)

    out_paper_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    log("    converting with Marker…")
    rendered = converter(str(pdf))
    try:
        n_figs = save_result(rendered, out_paper_dir, stem)
    finally:
        del rendered
        release_gpu_memory(label="post-convert")
    duration = round(time.time() - started, 1)

    meta = {
        "source_pdf": pdf.name,
        "sha256": pdf_hash,
        "stem": stem,
        "figures": n_figs,
        "device": device,
        "use_llm": use_llm,
        "no_equations": no_equations,
        "duration_sec": duration,
        "processed_at": datetime.now(timezone.utc).isoformat(),
    }
    (out_paper_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    log(f"    done — {n_figs} figure(s), {duration}s -> processed/{stem}/")
    return "done"


def run_isolated_batch(pdfs: list[Path], args: argparse.Namespace) -> None:
    """Run each PDF in a fresh subprocess so a CUDA abort cannot poison the batch."""
    script = ROOT / "process.py"
    total = len(pdfs)
    log(f"Isolate mode: {total} PDF(s), one fresh process each (models reload every time).")
    rc_failed = 0
    for i, pdf in enumerate(pdfs, start=1):
        cmd = [sys.executable, str(script), "--file", pdf.name]
        if args.force:
            cmd.append("--force")
        if args.llm:
            cmd.append("--llm")
        if args.no_equations:
            cmd.append("--no-equations")
        if args.low_vram:
            cmd.append("--low-vram")
        if args.verbose:
            cmd.append("-v")
        log(f"\n{'=' * 60}\n[{i}/{total}] isolate subprocess: {pdf.name}\n{'=' * 60}")
        rc = subprocess.run(cmd, cwd=ROOT).returncode
        if rc != 0:
            rc_failed += 1
            log(f"  subprocess exit code: {rc}")
    log(f"\nIsolate finished: {total - rc_failed}/{total} succeeded.")
    sys.exit(1 if rc_failed else 0)


def select_pdfs(arg_file: str | None) -> list[Path]:
    if arg_file:
        cand = Path(arg_file)
        if not cand.is_absolute():
            cand = PDF_DIR / arg_file
        if cand.suffix.lower() != ".pdf":
            cand = cand.with_suffix(".pdf")
        if not cand.exists():
            sys.exit(f"error: PDF not found: {cand}")
        return [cand]
    return sorted(PDF_DIR.glob("*.pdf"))


def main() -> None:
    global VERBOSE

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", help="convert a single PDF (name in pdf/ or a path)")
    ap.add_argument("--force", action="store_true", help="reconvert even if already done")
    ap.add_argument("--llm", action="store_true", help="enable Marker's LLM cleanup pass (needs ANTHROPIC_API_KEY)")
    ap.add_argument("-v", "--verbose", action="store_true", help="extra diagnostics (GPU memory, tracebacks, hashes)")
    ap.add_argument(
        "--no-equations",
        action="store_true",
        help="skip EquationProcessor (no LaTeX OCR; much safer on 6–8 GiB GPUs)",
    )
    ap.add_argument(
        "--low-vram",
        action="store_true",
        help="force GPU batch size 1 (auto-enabled when VRAM ≤ 8 GiB)",
    )
    ap.add_argument(
        "--isolate",
        action="store_true",
        help="run each PDF in its own subprocess (slower; survives fatal CUDA errors)",
    )
    args = ap.parse_args()
    VERBOSE = args.verbose

    PDF_DIR.mkdir(exist_ok=True)
    OUT_DIR.mkdir(exist_ok=True)

    pdfs = select_pdfs(args.file)
    if not pdfs:
        log(f"No PDFs found in {PDF_DIR}. Drop some in and re-run.")
        return

    if args.isolate and len(pdfs) > 1:
        run_isolated_batch(pdfs, args)
        return

    device = detect_device()
    low_vram = resolve_low_vram(args.low_vram, device)
    log(f"Device: {device}")
    for line in torch_runtime_info(device):
        log(f"  {line}")
    if low_vram and device == "cuda" and not args.low_vram:
        vram = gpu_total_vram_gib()
        log(f"  auto low-VRAM profile (≤{LOW_VRAM_GIB_THRESHOLD:g} GiB detected: {vram:.1f} GiB)")
    elif low_vram:
        log("  low-VRAM profile enabled (batch sizes -> 1)")
    if args.no_equations:
        log("  EquationProcessor disabled (--no-equations)")
    elif low_vram and device == "cuda":
        log(
            "  tip: if CUDA fails on math-heavy PDFs, retry with --no-equations "
            "(EquationProcessor is the usual culprit on laptop GPUs)."
        )
    if device == "cpu":
        log("  warning: CPU-only — expect minutes per paper. Consider a GPU host or an overnight batch.")
    if sys.platform == "win32" and device == "cuda":
        log(
            "  Windows CUDA: conservative batch sizes, 2 pdftext workers, and a short "
            "GPU cooldown between PDFs to reduce CLOCK_WATCHDOG_TIMEOUT / driver TDR risk."
        )
    if args.llm and not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("error: --llm set but ANTHROPIC_API_KEY is not in the environment (see .env.example).")

    _configure_torch_threads()

    from marker.models import create_model_dict

    log("Loading Marker models…")
    t0 = time.time()
    perf_config = marker_performance_config(low_vram=low_vram)
    profile = "low-vram" if low_vram else ("windows-cuda" if sys.platform == "win32" else "auto")
    log_perf_config(perf_config, profile=profile)
    artifact_dict = create_model_dict()
    log(f"  models loaded in {time.time() - t0:.1f}s")
    snap = gpu_memory_snapshot()
    if snap:
        log(f"  {snap}")

    converter = None
    conversions_since_recycle = 0
    cuda_aborted = False
    total = len(pdfs)

    log(f"\nQueue ({total} PDF(s)):")
    for i, pdf in enumerate(pdfs, start=1):
        log(f"  {i}. {pdf.name} ({format_size(pdf)})")

    stats = {"done": 0, "skipped": 0, "failed": 0, "aborted": 0}
    for i, pdf in enumerate(pdfs, start=1):
        if cuda_aborted:
            stats["aborted"] += 1
            log(f"\n[{i}/{total}] {pdf.name}")
            log("    skip — batch stopped after fatal CUDA error (re-run this file in a fresh process)")
            continue

        try:
            if converter is None:
                log("\n  building PdfConverter…", verbose_only=True)
                converter = build_converter(
                    args.llm, artifact_dict, perf_config, no_equations=args.no_equations
                )
                log("  PdfConverter ready", verbose_only=True)
            result = convert_one(
                pdf,
                converter,
                args.llm,
                args.force,
                device,
                index=i,
                total=total,
                no_equations=args.no_equations,
            )
            stats[result] += 1
            if result == "done":
                conversions_since_recycle += 1
                if conversions_since_recycle >= RECYCLE_CONVERTER_EVERY:
                    log("  recycling PdfConverter (memory hygiene)", verbose_only=True)
                    del converter
                    converter = None
                    conversions_since_recycle = 0
                    release_gpu_memory(cooldown=True, device=device, label="recycle")
        except Exception as exc:
            stats["failed"] += 1
            err_type = type(exc).__name__
            log(f"    FAILED [{err_type}]: {exc}")
            if VERBOSE:
                traceback.print_exc()

            if converter is not None:
                del converter
                converter = None
                conversions_since_recycle = 0
            release_gpu_memory(device=device, label="after-failure")

            fatal = is_fatal_cuda_error(exc) or not cuda_context_healthy()
            if fatal:
                cuda_aborted = True
                remaining = total - i
                log(
                    f"    fatal CUDA error — GPU context is unusable for the rest of this run "
                    f"({remaining} PDF(s) will be skipped)."
                )
                log_cuda_failure_hints(
                    exc,
                    pdf.name,
                    no_equations=args.no_equations,
                    low_vram=low_vram,
                )

    if converter is not None:
        del converter
    release_gpu_memory(label="shutdown")

    log(
        f"\nFinished: {stats['done']} converted, {stats['skipped']} skipped, "
        f"{stats['failed']} failed, {stats['aborted']} not attempted (CUDA abort)."
    )
    if stats["failed"] or stats["aborted"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
