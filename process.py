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
import hashlib
import json
import os
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

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


def already_done(out_paper_dir: Path, pdf_hash: str) -> bool:
    meta = out_paper_dir / "meta.json"
    if not meta.exists():
        return False
    try:
        data = json.loads(meta.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    return data.get("sha256") == pdf_hash


def build_converter(use_llm: bool):
    """Create a Marker PdfConverter, loading the model artifacts once."""
    from marker.config.parser import ConfigParser
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict

    config: dict = {"output_format": "markdown"}
    if use_llm:
        config["use_llm"] = True

    parser = ConfigParser(config)
    return PdfConverter(
        config=parser.generate_config_dict(),
        artifact_dict=create_model_dict(),
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
    n_figs = save_result(rendered, out_paper_dir, stem)
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
    if args.llm and not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("error: --llm set but ANTHROPIC_API_KEY is not in the environment (see .env.example).")

    print(f"Loading Marker models...")
    converter = build_converter(args.llm)

    print(f"Processing {len(pdfs)} PDF(s):")
    stats = {"done": 0, "skipped": 0}
    for pdf in pdfs:
        try:
            stats[convert_one(pdf, converter, args.llm, args.force, device)] += 1
        except Exception as exc:  # keep the batch going if one paper fails
            print(f"  FAILED: {pdf.name}: {exc}")

    print(f"\nFinished: {stats['done']} converted, {stats['skipped']} skipped.")


if __name__ == "__main__":
    main()
