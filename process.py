#!/usr/bin/env python3
"""Convert scientific-paper PDFs to Markdown using Marker.

Scans ./pdf for PDFs, converts each to Markdown (equations as LaTeX, figures
extracted to a per-paper figures/ folder), and writes results under ./processed.

Re-runs are cheap: a paper is skipped when its SHA-256 already matches the hash
recorded in its meta.json. Use --force to reconvert anyway.

On CUDA each paper is converted in a child process. A CUDA abort (OOM, illegal
memory access, driver reset) kills that child only; the batch keeps going, and
the offending paper is retried with progressively smaller batch sizes before the
next paper returns to the normal settings.

Usage:
    uv run process.py                 # convert all new PDFs in pdf/
    uv run process.py --file NAME     # convert a single PDF (name or path)
    uv run process.py --force         # reconvert even if already done
    uv run process.py --llm           # enable Marker's LLM cleanup pass
    uv run process.py --tier 1        # start from conservative settings
    uv run process.py --no-isolate    # single process (old behaviour)
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import shutil
import subprocess
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

# Worker exit codes. The parent uses these to decide whether a retry with
# smaller batches is worth attempting.
EXIT_OK = 0
EXIT_ERROR = 1  # the document itself is broken; smaller batches won't help
EXIT_SKIPPED = 2
EXIT_GPU = 17  # OOM / CUDA error caught in-process: retry smaller
EXIT_TIMEOUT = 18  # parent killed a stuck child: retry smaller

# Surya's CUDA batch defaults (recognition 256, detection 36, layout 32) assume
# an 8 GB+ card, and Marker's auto-tuner leaves them untouched whenever the GPU
# fits only one worker -- i.e. on exactly the small cards that cannot take them.
# Pin explicit sizes by VRAM instead. Bigger cards keep Marker's own tuning.
_BATCH_BY_VRAM: list[tuple[float, dict[str, int]]] = [
    (
        4.0,  # 3 GB class
        {
            "recognition_batch_size": 16,
            "layout_batch_size": 4,
            "detection_batch_size": 4,
            "ocr_error_batch_size": 8,
            "table_rec_batch_size": 4,
            "equation_batch_size": 2,
        },
    ),
    (
        8.0,  # 4-8 GB class
        {
            "recognition_batch_size": 48,
            "layout_batch_size": 8,
            "detection_batch_size": 8,
            "ocr_error_batch_size": 16,
            "table_rec_batch_size": 8,
            "equation_batch_size": 4,
        },
    ),
]

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

# Retry ladder, applied on top of the baseline after a failure. Tier 0 is the
# baseline itself; each further tier trades throughput for headroom. Only the
# failing document climbs the ladder -- the next one starts at tier 0 again.
_RETRY_TIERS: list[dict] = [
    {},
    {
        "recognition_batch_size": 4,
        "layout_batch_size": 1,
        "detection_batch_size": 1,
        "ocr_error_batch_size": 2,
        "table_rec_batch_size": 1,
        "equation_batch_size": 1,
        "pdftext_workers": 1,
    },
    {
        "recognition_batch_size": 1,
        "layout_batch_size": 1,
        "detection_batch_size": 1,
        "ocr_error_batch_size": 1,
        "table_rec_batch_size": 1,
        "equation_batch_size": 1,
        "pdftext_workers": 1,
        # Smaller page renders shrink every activation downstream of them.
        "highres_image_dpi": 144,
        "lowres_image_dpi": 72,
    },
]
MAX_TIER = len(_RETRY_TIERS) - 1


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


def gpu_vram_gb() -> float:
    """Total VRAM of the active CUDA device, 0.0 when there is none."""
    try:
        import torch

        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.get_device_properties(0).total_memory / (1024**3)
    except Exception:
        return 0.0


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


def _baseline_batch_config() -> dict:
    """Batch sizes for the current GPU, before any retry-tier shrinking."""
    if sys.platform == "win32":
        return dict(_WINDOWS_CUDA_BATCH)

    vram = gpu_vram_gb()
    for ceiling, sizes in _BATCH_BY_VRAM:
        if vram < ceiling:
            return dict(sizes)

    try:
        from marker.utils.batch import get_batch_sizes_worker_counts
        from marker.utils.gpu import GPUManager

        with GPUManager(0) as gpu_manager:
            batch_sizes, _ = get_batch_sizes_worker_counts(gpu_manager, 7)
        return dict(batch_sizes)
    except Exception:
        return {}


def marker_performance_config(tier: int = 0) -> dict:
    """GPU batch sizes and CPU extraction workers for a given retry tier."""
    perf: dict = {"disable_tqdm": True, "pdftext_workers": _pdftext_workers()}

    try:
        import torch

        has_cuda = torch.cuda.is_available()
    except Exception:
        has_cuda = False

    if has_cuda:
        perf.update(_baseline_batch_config())
    perf.update(_RETRY_TIERS[max(0, min(tier, MAX_TIER))])
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

    fresh_dir = not out_paper_dir.exists()
    out_paper_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    try:
        rendered = converter(str(pdf))
        try:
            n_figs = save_result(rendered, out_paper_dir, stem)
        finally:
            del rendered
            release_gpu_memory()
    except BaseException:
        # Don't leave a half-written directory behind: a later run would see a
        # stale/absent meta.json and the folder would look done-ish but wrong.
        if fresh_dir:
            shutil.rmtree(out_paper_dir, ignore_errors=True)
        raise
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


def _looks_like_gpu_error(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(
        needle in text
        for needle in ("cuda", "out of memory", "cublas", "cudnn", "device-side assert", "nvml")
    )


# --------------------------------------------------------------------------
# worker: converts exactly one PDF, then exits. Run as a child process so that
# a CUDA abort (which poisons the CUDA context for good) dies with it.
# --------------------------------------------------------------------------


def worker_main(args) -> int:
    pdfs = select_pdfs(args.file)
    pdf = pdfs[0]
    device = detect_device()
    _configure_torch_threads()

    from marker.models import create_model_dict

    perf_config = marker_performance_config(args.tier)
    artifact_dict = create_model_dict()
    converter = build_converter(args.llm, artifact_dict, perf_config)
    try:
        result = convert_one(pdf, converter, args.llm, args.force, device)
    except Exception as exc:
        print(f"  FAILED: {pdf.name}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_GPU if _looks_like_gpu_error(exc) else EXIT_ERROR
    finally:
        del converter
        release_gpu_memory()
    return EXIT_SKIPPED if result == "skipped" else EXIT_OK


def _describe_rc(rc: int) -> str:
    if rc == EXIT_GPU:
        return "GPU/CUDA error"
    if rc == EXIT_TIMEOUT:
        return "timeout"
    if rc == EXIT_ERROR:
        return "conversion error"
    if rc < 0:
        return f"killed by signal {-rc}"
    return f"crashed (exit {rc})"


def run_isolated(pdf: Path, tier: int, args, timeout: float) -> tuple[str, int]:
    """Convert one PDF in a child process. Returns (outcome, returncode)."""
    cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", "--file", str(pdf), "--tier", str(tier)]
    if args.llm:
        cmd.append("--llm")
    if args.force:
        cmd.append("--force")

    try:
        proc = subprocess.run(cmd, timeout=timeout if timeout > 0 else None)
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT after {timeout:.0f}s: {pdf.name} (child killed)")
        return "failed", EXIT_TIMEOUT

    if rc == EXIT_OK:
        return "done", rc
    if rc == EXIT_SKIPPED:
        return "skipped", rc
    return "failed", rc


def process_isolated(pdfs: list[Path], args, device: str) -> dict:
    stats = {"done": 0, "skipped": 0, "failed": 0}
    failures: list[str] = []
    timeout = args.timeout if args.timeout is not None else (
        float(os.environ.get("MARKER_DOC_TIMEOUT_SEC", "1800")) if device == "cuda" else 0.0
    )
    max_tier = args.tier if args.no_retry else MAX_TIER

    for pdf in pdfs:
        # Cheap hash check up front so finished papers never pay for a child process.
        if not args.force and already_done(OUT_DIR / slugify(pdf.stem), sha256(pdf)):
            print(f"  skip (already done): {pdf.name}")
            stats["skipped"] += 1
            continue

        tier = args.tier
        while True:
            if tier > args.tier:
                print(f"  retry: {pdf.name} at tier {tier} (conservative batch sizes)")
            outcome, rc = run_isolated(pdf, tier, args, timeout)
            if outcome != "failed":
                stats[outcome] += 1
                break
            if rc == EXIT_ERROR or tier >= max_tier:
                reason = _describe_rc(rc)
                print(f"  GIVING UP: {pdf.name}: {reason} (last tier {tier}) -- continuing with the batch")
                stats["failed"] += 1
                failures.append(pdf.name)
                break
            if rc != EXIT_TIMEOUT:  # the timeout already announced itself
                print(f"  {_describe_rc(rc)}: {pdf.name} at tier {tier}")
            tier += 1
            # Let the driver settle before handing it another context.
            time.sleep(float(os.environ.get("MARKER_GPU_COOLDOWN_SEC", "3")))

    stats["failures"] = failures
    return stats


def process_in_line(pdfs: list[Path], args, device: str) -> dict:
    """Single-process fallback (--no-isolate). A hard CUDA abort ends the run."""
    _configure_torch_threads()

    from marker.models import create_model_dict

    print("Loading Marker models...")
    artifact_dict = create_model_dict()
    converter = None
    converter_tier = args.tier
    conversions_since_recycle = 0

    stats = {"done": 0, "skipped": 0, "failed": 0}
    failures: list[str] = []
    max_tier = args.tier if args.no_retry else MAX_TIER

    for pdf in pdfs:
        tier = args.tier
        while True:
            try:
                if converter is None or converter_tier != tier:
                    converter = build_converter(args.llm, artifact_dict, marker_performance_config(tier))
                    converter_tier = tier
                result = convert_one(pdf, converter, args.llm, args.force, device)
                stats[result] += 1
                if result == "done":
                    conversions_since_recycle += 1
                    if conversions_since_recycle >= RECYCLE_CONVERTER_EVERY:
                        del converter
                        converter = None
                        conversions_since_recycle = 0
                        release_gpu_memory(cooldown=True, device=device)
                break
            except Exception as exc:  # keep the batch going if one paper fails
                if converter is not None:
                    del converter
                    converter = None
                    conversions_since_recycle = 0
                release_gpu_memory()
                retryable = _looks_like_gpu_error(exc) and tier < max_tier
                if not retryable:
                    print(f"  FAILED: {pdf.name}: {exc}")
                    stats["failed"] += 1
                    failures.append(pdf.name)
                    break
                tier += 1
                print(f"  {type(exc).__name__} on {pdf.name}; retrying at tier {tier} (conservative batch sizes)")
                time.sleep(float(os.environ.get("MARKER_GPU_COOLDOWN_SEC", "3")))

    if converter is not None:
        del converter
    release_gpu_memory()

    stats["failures"] = failures
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", help="convert a single PDF (name in pdf/ or a path)")
    ap.add_argument("--force", action="store_true", help="reconvert even if already done")
    ap.add_argument("--llm", action="store_true", help="enable Marker's LLM cleanup pass (needs ANTHROPIC_API_KEY)")
    ap.add_argument(
        "--tier",
        type=int,
        default=0,
        choices=range(MAX_TIER + 1),
        help="starting settings tier: 0 = auto for this GPU, higher = smaller batches",
    )
    ap.add_argument("--no-retry", action="store_true", help="do not retry a failed PDF with smaller batches")
    ap.add_argument(
        "--no-isolate",
        action="store_true",
        help="convert in this process instead of one child per PDF (a CUDA abort then ends the run)",
    )
    ap.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="seconds before a stuck child is killed and retried (0 disables; CUDA default 1800)",
    )
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()

    PDF_DIR.mkdir(exist_ok=True)
    OUT_DIR.mkdir(exist_ok=True)

    if args.worker:
        if not args.file:
            sys.exit("error: --worker requires --file")
        sys.exit(worker_main(args))

    pdfs = select_pdfs(args.file)
    if not pdfs:
        print(f"No PDFs found in {PDF_DIR}. Drop some in and re-run.")
        return

    device = detect_device()
    print(f"Device: {device}")
    if device == "cuda":
        vram = gpu_vram_gb()
        print(f"  VRAM: {vram:.1f} GB")
        if vram and vram < 4.0:
            print("  small GPU: pinning conservative batch sizes (surya's CUDA defaults assume 8 GB+).")
    if device == "cpu":
        print("  warning: CPU-only — expect minutes per paper. Consider a GPU host or an overnight batch.")
    if sys.platform == "win32" and device == "cuda":
        print(
            "  Windows CUDA: conservative batch sizes, 2 pdftext workers, and a short "
            "GPU cooldown between PDFs to reduce CLOCK_WATCHDOG_TIMEOUT / driver TDR risk."
        )
    if args.llm and not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("error: --llm set but ANTHROPIC_API_KEY is not in the environment (see .env.example).")

    isolate = not args.no_isolate
    if isolate:
        print("  isolation: one child process per PDF; a CUDA abort loses only that paper.")

    print(f"Processing {len(pdfs)} PDF(s):")
    stats = (process_isolated if isolate else process_in_line)(pdfs, args, device)

    print(f"\nFinished: {stats['done']} converted, {stats['skipped']} skipped, {stats['failed']} failed.")
    if stats["failures"]:
        print("Failed: " + ", ".join(stats["failures"]))
        sys.exit(1)


if __name__ == "__main__":
    main()
