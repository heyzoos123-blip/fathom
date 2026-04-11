# -*- coding: utf-8 -*-
"""
build_paper_v16.py -- Build the Fathom Paper v16 Zenodo bundle locally.

Contents: the 36-page long-form arXiv paper (fathom_arxiv_v1.{tex,pdf}
with the new Section 9.5 spectral pilot + sensitivity + biological
parallel), the existing ICML v2 short paper (unchanged since v15),
references.bib, style files, referenced figures, both pre-registrations
for the new experiments (spectral sensitivity + WM prioritization),
the three new findings documents from 2026-04-11, the new analysis
scripts, the new spectral JSONs, and a v16 README describing the
delta vs v15.

This bundle is intended to be uploaded as a new version on the existing
Fathom concept DOI (10.5281/zenodo.19326174), branching from paper v15
(record 19504993).

The atlas bundle itself (fathom_atlas_v0.3.zip) is UNCHANGED vs v15 and
is re-uploaded to the same v16 record as a sibling file, so the unified
release pattern established in v15 is preserved: one Zenodo record,
two sibling files, all atlas data accessible alongside the paper.

Usage:
    python zenodo_v16/build_paper_v16.py

Produces:
    zenodo_v16/fathom_paper_v16.zip
    zenodo_v16/bundle_manifest_v16.json

No network access. No API calls. Local zip build only.
"""

import hashlib
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "zenodo_v16"
OUT_DIR.mkdir(exist_ok=True)


# ── Paper v16 bundle contents ────────────────────────────────────
# (src_rel_to_repo_root, arc_rel_inside_bundle_prefix)

PAPER_FILES = [
    # --- the long-form arxiv paper (the headline of v16) ---
    ("paper/fathom_arxiv_v1.tex", "paper/fathom_arxiv_v1.tex"),
    ("paper/fathom_arxiv_v1.pdf", "paper/fathom_arxiv_v1.pdf"),
    ("paper/arxiv.sty",           "paper/arxiv.sty"),

    # --- the ICML short paper (unchanged in content but source updated
    #     to cite weng2026sustained) ---
    ("paper/fathom_icml2026_v2.tex", "paper/fathom_icml2026_v2.tex"),
    ("paper/fathom_icml2026_v2.pdf", "paper/fathom_icml2026_v2.pdf"),
    ("paper/icml2026.sty",           "paper/icml2026.sty"),
    ("paper/icml2026.bst",           "paper/icml2026.bst"),
    ("paper/algorithm.sty",          "paper/algorithm.sty"),
    ("paper/algorithmic.sty",        "paper/algorithmic.sty"),
    ("paper/fancyhdr.sty",           "paper/fancyhdr.sty"),

    # --- shared bib ---
    ("paper/references.bib", "paper/references.bib"),

    # --- figures referenced by the long-form paper ---
    ("paper/figures/spectral_hallucination_signature.pdf",
     "paper/figures/spectral_hallucination_signature.pdf"),
    ("paper/figures/cognitive_state_map.png",
     "paper/figures/cognitive_state_map.png"),
    ("paper/figures/cognitive_autopsy_overlay.png",
     "paper/figures/cognitive_autopsy_overlay.png"),
    ("paper/figures/spectrum_n200_full.png",
     "paper/figures/spectrum_n200_full.png"),
    # These figures are referenced only in the ICML short paper but
    # are included for self-containment of the bundle:
    ("paper/figures/S_orthogonality.png", "paper/figures/S_orthogonality.png"),
    ("paper/figures/biphasic_horizon_curve.png",
     "paper/figures/biphasic_horizon_curve.png"),

    # --- new pre-registrations (v16) ---
    ("prereg/spectral_sensitivity_prereg.md",
     "prereg/spectral_sensitivity_prereg.md"),
    ("prereg/wm_prioritization_prereg.md",
     "prereg/wm_prioritization_prereg.md"),

    # --- new findings documents (v16) ---
    ("findings/findings_2026_04_11_alpha_bridge.md",
     "findings/findings_2026_04_11_alpha_bridge.md"),
    ("findings/findings_2026_04_11_spectral_pilot.md",
     "findings/findings_2026_04_11_spectral_pilot.md"),
    ("findings/findings_2026_04_11_spectral_sensitivity_v22.md",
     "findings/findings_2026_04_11_spectral_sensitivity_v22.md"),

    # --- new analysis scripts (spectral + v22) ---
    ("analysis/analyze_S_spectral.py",
     "analysis/analyze_S_spectral.py"),
    ("analysis/analyze_S_spectral_controls.py",
     "analysis/analyze_S_spectral_controls.py"),
    ("analysis/analyze_S_spectral_v22.py",
     "analysis/analyze_S_spectral_v22.py"),
    ("analysis/make_spectral_figure.py",
     "analysis/make_spectral_figure.py"),

    # --- WM prioritization scaffold (design + runner + analyzer, not yet run) ---
    ("analysis/build_wm_prompts.py",
     "analysis/build_wm_prompts.py"),
    ("analysis/analyze_wm_prioritization.py",
     "analysis/analyze_wm_prioritization.py"),
    ("runners/run_wm_prioritization.py",
     "runners/run_wm_prioritization.py"),
    ("prompts/wm_prioritization_prompts.json",
     "prompts/wm_prioritization_prompts.json"),

    # --- raw spectral data (v16 headline numbers come from these) ---
    ("truthfulqa_results/S_baselines_n200_spectral.json",
     "truthfulqa_results/S_baselines_n200_spectral.json"),
    ("truthfulqa_results/S_baselines_n200_spectral_v22.json",
     "truthfulqa_results/S_baselines_n200_spectral_v22.json"),
    ("truthfulqa_results/S_baselines_n200_spectral_controls.json",
     "truthfulqa_results/S_baselines_n200_spectral_controls.json"),
    ("truthfulqa_results/S_baselines_n200_v22_audit.json",
     "truthfulqa_results/S_baselines_n200_v22_audit.json"),
    ("truthfulqa_results/S_truthfulqa_llama_n200_spectral.json",
     "truthfulqa_results/S_truthfulqa_llama_n200_spectral.json"),
    ("truthfulqa_results/S_truthfulqa_llama_n200_spectral_v22.json",
     "truthfulqa_results/S_truthfulqa_llama_n200_spectral_v22.json"),
    ("truthfulqa_results/S_truthfulqa_llama_n200_v22_audit.json",
     "truthfulqa_results/S_truthfulqa_llama_n200_v22_audit.json"),

    # --- v16 README describing the delta ---
    ("zenodo_v16/README.md", "README.md"),
]


def build_zip(zip_path, bundle_name, files):
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    total_bytes = 0
    file_count = 0
    missing = []
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED,
                         compresslevel=9) as z:
        for src_rel, arc_rel in files:
            src_abs = ROOT / src_rel
            if not src_abs.exists():
                missing.append(src_rel)
                continue
            arcname = f"{bundle_name}/{arc_rel}"
            z.write(src_abs, arcname=arcname)
            total_bytes += src_abs.stat().st_size
            file_count += 1
    return file_count, total_bytes, missing


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head():
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT,
            capture_output=True, text=True, check=False,
        )
        return r.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def main():
    print("=" * 68)
    print("BUILDING FATHOM PAPER v16 BUNDLE")
    print("=" * 68)
    print()

    paper_zip = OUT_DIR / "fathom_paper_v16.zip"
    print(f"Bundle -> {paper_zip.name}")
    n_files, bytes_raw, missing = build_zip(
        paper_zip, "fathom_paper_v16", PAPER_FILES,
    )
    if missing:
        print(f"  [WARN] {len(missing)} missing files:")
        for m in missing:
            print(f"    - {m}")

    if not paper_zip.exists():
        print("[FATAL] no files bundled, aborting")
        sys.exit(1)

    zip_hash = sha256(paper_zip)
    print(f"  files included:   {n_files}")
    print(f"  size on disk:     {paper_zip.stat().st_size:>10,} bytes")
    print(f"  raw contents:     {bytes_raw:>10,} bytes")
    print(f"  sha256:           {zip_hash}")
    print()

    manifest = {
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_commit": git_head(),
        "paper_bundle": {
            "path": str(paper_zip.relative_to(ROOT)),
            "filename": paper_zip.name,
            "n_files": n_files,
            "bytes_on_disk": paper_zip.stat().st_size,
            "bytes_raw": bytes_raw,
            "sha256": zip_hash,
            "missing_files": missing,
        },
    }
    manifest_path = OUT_DIR / "bundle_manifest_v16.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[saved] {manifest_path.relative_to(ROOT)}")
    print()
    print("Ready for upload. Next steps:")
    print("  1. Inspect the bundle manifest and the zip contents.")
    print("  2. Run upload_paper_v16.py --dry-run to verify metadata.")
    print("  3. Only publish after explicit approval.")


if __name__ == "__main__":
    main()
