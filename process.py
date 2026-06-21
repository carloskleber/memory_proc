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
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def _configure_cpu_threads() -> None:
    """Cap BLAS/OpenMP threads so pdftext workers and PyTorch don't oversubscribe."""
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    # Surya spawns nested pools unless this is set (Marker's convert CLI does the same).
    os.environ.setdefault("IN_STREAMLIT", "true")
    os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
    os.environ.setdefault("GLOG_minloglevel", "2")
    # Reduces VRAM fragmentation on long batch runs.
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
_WINDOWS_CUDA_BATCH: dict[str, int] = {
    "recognition_batch_size": 12,
    "layout_batch_size": 4,
    "detection_batch_size": 4,
    "ocr_error_batch_size": 4,
    "table_rec_batch_size": 4,
    "equation_batch_size": 4,
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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
    except Exception:
        pass


def release_gpu_memory(*, cooldown: bool = False, device: str = "") -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except Exception:
        pass
    if cooldown and sys.platform == "win32" and device == "cuda":
        sec = float(os.environ.get("MARKER_GPU_COOLDOWN_SEC", "3"))
        if sec > 0:
            time.sleep(sec)


def marker_performance_config() -> dict:
    """GPU batch sizes and CPU extraction workers (mirrors Marker's convert CLI)."""
    perf: dict = {"disable_tqdm": True, "pdftext_workers": _pdftext_workers()}

    if sys.platform == "win32":
        try:
            import torch

            if torch.cuda.is_available():
                perf.update(_WINDOWS_CUDA_BATCH)
        except Exception:
            pass
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


def already_done(out_paper_dir: Path, pdf_hash: str) -> bool:
    meta = out_paper_dir / "meta.json"
    if not meta.exists():
        return False
    try:
        data = json.loads(meta.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    return data.get("sha256") == pdf_hash


def build_converter(use_llm: bool, artifact_dict: dict, perf_config: dict | None = None):
    """Create a Marker PdfConverter using a shared model artifact dict."""
    from marker.config.parser import ConfigParser
    from marker.converters.pdf import PdfConverter

    config: dict = {"output_format": "markdown"}
    config.update(perf_config if perf_config is not None else marker_performance_config())
    if use_llm:
        config["use_llm"] = True

    parser = ConfigParser(config)
    return PdfConverter(
        config=parser.generate_config_dict(),
        artifact_dict=artifact_dict,
        processor_list=parser.get_processors(),
        renderer=parser.get_renderer(),
        llm_service=parser.get_llm_service() if use_llm else None,
    )


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
    (out_paper_dir / f"{stem}.md").write_text(md, encoding="utf-8")
    return count


def convert_one(pdf: Path, converter, use_llm: bool, force: bool, device: str) -> str:
    stem = slugify(pdf.stem)
    out_paper_dir = OUT_DIR / stem
    pdf_hash = sha256(pdf)

    if not force and already_done(out_paper_dir, pdf_hash):
        print(f"  skip (already done): {pdf.name}")
        return "skipped"

    out_paper_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    rendered = converter(str(pdf))
    try:
        n_figs = save_result(rendered, out_paper_dir, stem)
    finally:
        del rendered
        release_gpu_memory()
    duration = round(time.time() - started, 1)

    meta = {
        "source_pdf": pdf.name,
        "sha256": pdf_hash,
        "stem": stem,
        "figures": n_figs,
        "device": device,
        "use_llm": use_llm,
        "duration_sec": duration,
        "processed_at": datetime.now(timezone.utc).isoformat(),
    }
    (out_paper_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"  done: {pdf.name} -> processed/{stem}/ ({n_figs} figures, {duration}s)")
    return "done"


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
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", help="convert a single PDF (name in pdf/ or a path)")
    ap.add_argument("--force", action="store_true", help="reconvert even if already done")
    ap.add_argument("--llm", action="store_true", help="enable Marker's LLM cleanup pass (needs ANTHROPIC_API_KEY)")
    args = ap.parse_args()

    PDF_DIR.mkdir(exist_ok=True)
    OUT_DIR.mkdir(exist_ok=True)

    pdfs = select_pdfs(args.file)
    if not pdfs:
        print(f"No PDFs found in {PDF_DIR}. Drop some in and re-run.")
        return

    device = detect_device()
    print(f"Device: {device}")
    if device == "cpu":
        print("  warning: CPU-only — expect minutes per paper. Consider a GPU host or an overnight batch.")
    if sys.platform == "win32" and device == "cuda":
        print(
            "  Windows CUDA: conservative batch sizes, 2 pdftext workers, and a short "
            "GPU cooldown between PDFs to reduce CLOCK_WATCHDOG_TIMEOUT / driver TDR risk."
        )
    if args.llm and not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("error: --llm set but ANTHROPIC_API_KEY is not in the environment (see .env.example).")

    _configure_torch_threads()

    from marker.models import create_model_dict

    print("Loading Marker models...")
    perf_config = marker_performance_config()
    artifact_dict = create_model_dict()
    converter = None
    conversions_since_recycle = 0

    print(f"Processing {len(pdfs)} PDF(s):")
    stats = {"done": 0, "skipped": 0}
    for pdf in pdfs:
        try:
            if converter is None:
                converter = build_converter(args.llm, artifact_dict, perf_config)
            result = convert_one(pdf, converter, args.llm, args.force, device)
            stats[result] += 1
            if result == "done":
                conversions_since_recycle += 1
                if conversions_since_recycle >= RECYCLE_CONVERTER_EVERY:
                    del converter
                    converter = None
                    conversions_since_recycle = 0
                    release_gpu_memory(cooldown=True, device=device)
        except Exception as exc:  # keep the batch going if one paper fails
            print(f"  FAILED: {pdf.name}: {exc}")
            if converter is not None:
                del converter
                converter = None
                conversions_since_recycle = 0
                release_gpu_memory()

    if converter is not None:
        del converter
    release_gpu_memory()

    print(f"\nFinished: {stats['done']} converted, {stats['skipped']} skipped.")


if __name__ == "__main__":
    main()
