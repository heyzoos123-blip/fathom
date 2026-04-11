# -*- coding: utf-8 -*-
"""
upload_paper_v16.py -- Publish Fathom Paper v16 to Zenodo as a new
version on the existing concept DOI (10.5281/zenodo.19326174),
branching from v15 (record 19504993).

Delta vs v15: see zenodo_v16/README.md. Headline:
  - long-form arxiv paper (fathom_arxiv_v1.pdf) included for the first time
  - new Section 9.5: spectral pilot + pre-registered sensitivity + bio parallel
  - Weng et al. 2026 biological parallel citation
  - in-silico WM prioritization experiment scaffold (pre-reg + runner)

Both bundles are attached as sibling files on the v16 record:
  - fathom_paper_v16.zip   (new, built by build_paper_v16.py)
  - fathom_atlas_v0.3.zip  (unchanged from v15)

Safety:
  - Default is DRY RUN (no HTTP writes).
  - --draft creates the draft + uploads files but does NOT publish.
  - --publish does the full flow and writes zenodo_v16/paper_v16_published.json.
  - ZENODO_TOKEN must be in environment. Never hardcoded.

Usage:
    python zenodo_v16/upload_paper_v16.py                   # dry run
    ZENODO_TOKEN=xxx python zenodo_v16/upload_paper_v16.py --draft
    ZENODO_TOKEN=xxx python zenodo_v16/upload_paper_v16.py --publish
"""

import json
import os
import sys
from pathlib import Path

try:
    import requests
except ImportError:
    print("[FATAL] requests package not installed")
    sys.exit(1)


# Both bundles live in the non-worktree research repo directory.
PAPER_BUNDLE = Path(r"C:\Users\heyzo\clawd\sae-reasoning-depth\figures\.claude\worktrees\intelligent-brown\zenodo_v16\fathom_paper_v16.zip")
ATLAS_BUNDLE = Path(r"C:\Users\heyzo\clawd\sae-reasoning-depth\zenodo_v14\fathom_atlas_v0.3.zip")

PAPER_BUNDLE_NAME = "fathom_paper_v16.zip"
ATLAS_BUNDLE_NAME = "fathom_atlas_v0.3.zip"

ZENODO_BASE = "https://zenodo.org/api"

CONCEPT_ID = 19326174       # Fathom paper concept DOI (v1..v15 history)
LATEST_V15_ID = 19504993    # v15: unified release (paper + atlas sibling files)
ORPHAN_ATLAS_ID = 19502716  # Standalone atlas record, already marked obsoleted


METADATA = {
    "metadata": {
        "upload_type": "publication",
        "publication_type": "workingpaper",
        "title": (
            "Fathom: Cognitive Measurement Instruments for Transformer "
            "Internals via SAE Feature Coherence Geometry"
        ),
        "creators": [
            {"name": "Rodabaugh, Alexander",
             "affiliation": "Fathom Intelligence"},
        ],
        "description": (
            "<p><strong>v16 -- long-form paper + spectral pilot + "
            "biological parallel.</strong> This version introduces "
            "the 36-page long-form arXiv paper to the Zenodo record "
            "(prior versions only shipped the ICML workshop-format "
            "short paper), adds a new Section 9.5 reporting an "
            "exploratory spectral pilot of C_delta trajectories and "
            "its pre-registered sensitivity analysis, and frames the "
            "finding against a newly published biological parallel in "
            "human cognitive neuroscience.</p>"
            "<h3>What is new in v16</h3>"
            "<ul>"
            "<li><strong>Spectral pilot of C_delta trajectories.</strong> "
            "Fourier analysis of per-token coherence trajectories on "
            "n=200 TruthfulQA items in Gemma-2-2b-IT and "
            "Llama-3.2-1B-Instruct. Correct generations carry more "
            "spectral power than hallucinated generations at a "
            "characteristic low-mid-period peak bin (period 4 tokens "
            "for Gemma, period 8 tokens for Llama). The peak bin and "
            "direction replicate exactly under Fathom's v21 and v22 "
            "labelings on both models. Per-bin p &lt; 0.01 on all four "
            "runs.</li>"
            "<li><strong>Pre-registered sensitivity analysis.</strong> "
            "A v22 semantic-relabeling sensitivity test was "
            "pre-registered and sealed in git commit 90a6f7d before "
            "any v22 spectral analysis was run. The sealed decision "
            "rule required family-wise p &lt; 0.05 on both models. "
            "Under the v22 labeling the family-wise permutation "
            "returned p = 0.058 (Gemma) and p = 0.081 (Llama) -- both "
            "borderline, both failing the sealed threshold. Per the "
            "pre-registered rule, H1 is reported as <strong>not "
            "supported</strong>. The direction and peak bin replicate "
            "robustly; the multi-comparison correction across 20 bins "
            "is too conservative at current sample sizes. A bin-locked "
            "confirmatory test is the proposed next step.</li>"
            "<li><strong>Biological parallel.</strong> The frequency-"
            "domain analysis is motivated by Weng, Borst &amp; Akyurek "
            "(2026, <em>Imaging Neuroscience</em>, "
            "doi:10.1162/imag.a.1199) -- \"Sustained alpha oscillations "
            "serve attentional prioritization in working memory, not "
            "maintenance.\" Their sustained alpha dissociation between "
            "prioritization (alpha-indexed) and passive storage "
            "(synaptic, not alpha-indexed) is structurally analogous "
            "to Fathom's S_early / K dissociation. We make no causal-"
            "identity claim, but the substrate-independent construct "
            "(\"localization of active selection\") is the first "
            "explicit bridge from Fathom to the frequency-domain "
            "tradition in cognitive neuroscience.</li>"
            "<li><strong>In-silico WM prioritization experiment "
            "scaffold.</strong> A pre-registered in-silico replication "
            "of the Weng two-item WM paradigm is shipped as "
            "scaffolding: sealed pre-reg, deterministic 60-trial "
            "prompt battery (30 fact pairs x 2 probe targets, balanced "
            "short/long delay), runner script using CoherenceSteererExt, "
            "and a locked analyzer for the primary test "
            "(S_early_A &gt; S_early_B paired permutation, "
            "n_perm = 10000, seed = 42). The experiment has not yet "
            "been run; it is shipped as a sealed design for when GPU "
            "time is available.</li>"
            "<li><strong>Long-form arXiv paper included.</strong> "
            "v14 and v15 shipped only the ICML workshop-format short "
            "paper as the PDF artefact. v16 ships the 36-page "
            "long-form <code>fathom_arxiv_v1</code> as the headline "
            "paper alongside the ICML short paper, which is retained "
            "for workshop citation.</li>"
            "</ul>"
            "<h3>What is unchanged vs v15</h3>"
            "<ul>"
            "<li>The atlas v0.3 data bundle is byte-identical to v15 "
            "and is re-attached as a sibling file on this record.</li>"
            "<li>The Fathom v14 measurement instruments (K, C, "
            "C_delta, S, D) are unchanged.</li>"
            "<li>The atlas v0.3 RLHF attractor replication results "
            "(n=6, mean LOO cosine +0.769, permutation p = 0.0315) "
            "are unchanged.</li>"
            "</ul>"
            "<h3>Supersedes</h3>"
            "<p>Paper v15: <code>10.5281/zenodo.19504993</code>. "
            "Full paper-series concept history: "
            "<code>10.5281/zenodo.19326174</code>.</p>"
            "<h3>License and patents</h3>"
            "<p>CC-BY-4.0 on the paper and atlas data. MIT on the "
            "analysis code. Related provisional patents: "
            "US 64/020,489, 64/021,113, 64/026,964.</p>"
        ),
        "keywords": [
            "Fathom",
            "cognitive atlas",
            "sparse autoencoder",
            "mechanistic interpretability",
            "cognitive measurement",
            "hallucination detection",
            "commitment intensity",
            "inverse participation ratio",
            "spectral analysis",
            "alpha oscillations",
            "working memory prioritization",
            "transformer internals",
            "RLHF",
            "cross-architecture",
            "pre-registration",
            "ICML 2026",
            "open-weight language models",
        ],
        "license": "cc-by-4.0",
        "publication_date": "2026-04-11",
        "related_identifiers": [
            {
                "relation": "isNewVersionOf",
                "identifier": f"10.5281/zenodo.{LATEST_V15_ID}",
                "resource_type": "publication-workingpaper",
                "scheme": "doi",
            },
            # Atlas standalone record (already marked obsoleted by v15)
            {
                "relation": "hasPart",
                "identifier": f"10.5281/zenodo.{ORPHAN_ATLAS_ID}",
                "resource_type": "dataset",
                "scheme": "doi",
            },
            # Cite Weng et al. 2026 as external resource (biological parallel)
            {
                "relation": "isDocumentedBy",
                "identifier": "10.1162/imag.a.1199",
                "resource_type": "publication-article",
                "scheme": "doi",
            },
        ],
    }
}


def fail(msg):
    print(f"[FATAL] {msg}")
    sys.exit(1)


def main():
    args = sys.argv[1:]
    dry_run = "--publish" not in args and "--draft" not in args
    do_publish = "--publish" in args

    print("=" * 68)
    print("UPLOAD: Fathom Paper v16 to Zenodo -- UNIFIED RELEASE")
    print("=" * 68)
    print(f"Paper bundle: {PAPER_BUNDLE.name}")
    if PAPER_BUNDLE.exists():
        print(f"  size:       {PAPER_BUNDLE.stat().st_size:,} bytes")
    else:
        print(f"  MISSING")
    print(f"Atlas bundle: {ATLAS_BUNDLE.name}")
    if ATLAS_BUNDLE.exists():
        print(f"  size:       {ATLAS_BUNDLE.stat().st_size:,} bytes")
    else:
        print(f"  MISSING")
    print(f"Concept DOI:  10.5281/zenodo.{CONCEPT_ID}")
    print(f"Branch from:  10.5281/zenodo.{LATEST_V15_ID} (v15)")
    print(f"Mode:         {'DRY RUN' if dry_run else ('PUBLISH' if do_publish else 'DRAFT ONLY')}")
    print()

    if dry_run:
        print("[DRY RUN] No HTTP calls. Metadata preview:")
        print("-" * 68)
        print(f"  title:    {METADATA['metadata']['title']}")
        print(f"  type:     {METADATA['metadata']['upload_type']} / "
              f"{METADATA['metadata']['publication_type']}")
        print(f"  license:  {METADATA['metadata']['license']}")
        print(f"  keywords: {len(METADATA['metadata']['keywords'])} tags")
        print(f"  related_identifiers:")
        for rel in METADATA["metadata"]["related_identifiers"]:
            print(f"    - {rel['relation']} {rel['identifier']}")
        print(f"  files to upload:")
        if PAPER_BUNDLE.exists():
            print(f"    + {PAPER_BUNDLE_NAME} "
                  f"({PAPER_BUNDLE.stat().st_size:,} bytes)")
        if ATLAS_BUNDLE.exists():
            print(f"    + {ATLAS_BUNDLE_NAME} "
                  f"({ATLAS_BUNDLE.stat().st_size:,} bytes)")
        print()
        print("[DRY RUN] To draft (uploads files, does NOT publish):")
        print("  ZENODO_TOKEN=<token> python zenodo_v16/upload_paper_v16.py --draft")
        print("[DRY RUN] To publish:")
        print("  ZENODO_TOKEN=<token> python zenodo_v16/upload_paper_v16.py --publish")
        return

    if not PAPER_BUNDLE.exists():
        fail(f"Paper bundle missing: {PAPER_BUNDLE}")
    if not ATLAS_BUNDLE.exists():
        fail(f"Atlas bundle missing: {ATLAS_BUNDLE}")

    token = os.environ.get("ZENODO_TOKEN", "").strip()
    if not token:
        fail("ZENODO_TOKEN env var not set. Refusing to run.")

    headers = {"Authorization": f"Bearer {token}"}

    print(f"[1/6] Creating new version from v15 (ID: {LATEST_V15_ID})...")
    r = requests.post(
        f"{ZENODO_BASE}/deposit/depositions/{LATEST_V15_ID}/actions/newversion",
        headers=headers,
    )
    if r.status_code != 201:
        fail(f"newversion failed: {r.status_code} {r.text[:300]}")
    new_dep = r.json()
    draft_url = new_dep["links"].get("latest_draft")
    if not draft_url:
        fail("no latest_draft link in newversion response")
    new_id = int(draft_url.rstrip("/").split("/")[-1])
    print(f"      new draft ID: {new_id}")

    print(f"[2/6] Clearing files inherited from v15 ...")
    r = requests.get(f"{ZENODO_BASE}/deposit/depositions/{new_id}",
                     headers=headers)
    if r.status_code != 200:
        fail(f"GET draft failed: {r.status_code}")
    draft = r.json()
    for f_info in draft.get("files", []):
        fid = f_info["id"]
        fname = f_info.get("filename", fid)
        rd = requests.delete(
            f"{ZENODO_BASE}/deposit/depositions/{new_id}/files/{fid}",
            headers=headers,
        )
        print(f"      deleted: {fname} ({rd.status_code})")

    bucket_url = draft["links"]["bucket"]

    print(f"[3/6] Uploading {PAPER_BUNDLE_NAME} "
          f"({PAPER_BUNDLE.stat().st_size:,} bytes) ...")
    with open(PAPER_BUNDLE, "rb") as f:
        r = requests.put(
            f"{bucket_url}/{PAPER_BUNDLE_NAME}",
            data=f,
            headers={"Authorization": f"Bearer {token}"},
        )
    if r.status_code not in (200, 201):
        fail(f"paper upload failed: {r.status_code} {r.text[:300]}")
    print(f"      uploaded: {r.json().get('size', 0):,} bytes")

    print(f"[4/6] Uploading {ATLAS_BUNDLE_NAME} "
          f"({ATLAS_BUNDLE.stat().st_size:,} bytes) ...")
    with open(ATLAS_BUNDLE, "rb") as f:
        r = requests.put(
            f"{bucket_url}/{ATLAS_BUNDLE_NAME}",
            data=f,
            headers={"Authorization": f"Bearer {token}"},
        )
    if r.status_code not in (200, 201):
        fail(f"atlas upload failed: {r.status_code} {r.text[:300]}")
    print(f"      uploaded: {r.json().get('size', 0):,} bytes")

    print(f"[5/6] Setting metadata ...")
    r = requests.put(
        f"{ZENODO_BASE}/deposit/depositions/{new_id}",
        json=METADATA,
        headers={**headers, "Content-Type": "application/json"},
    )
    if r.status_code != 200:
        fail(f"set metadata failed: {r.status_code} {r.text[:300]}")
    print(f"      metadata set for {new_id}")

    if do_publish:
        print(f"[6/6] Publishing ...")
        r = requests.post(
            f"{ZENODO_BASE}/deposit/depositions/{new_id}/actions/publish",
            headers=headers,
        )
        if r.status_code != 202:
            fail(f"publish failed: {r.status_code} {r.text[:300]}")
        result = r.json()
        doi = result.get("doi", "unknown")
        url = result.get("links", {}).get("record_html", "unknown")
        print(f"      PUBLISHED")
        print(f"      DOI: {doi}")
        print(f"      URL: {url}")
        out_record = {
            "new_record_id": new_id,
            "doi": doi,
            "url": url,
            "concept_doi": f"10.5281/zenodo.{CONCEPT_ID}",
            "supersedes": f"10.5281/zenodo.{LATEST_V15_ID}",
            "bundles": [PAPER_BUNDLE_NAME, ATLAS_BUNDLE_NAME],
        }
        ROOT = Path(__file__).resolve().parent.parent
        with open(ROOT / "zenodo_v16" / "paper_v16_published.json",
                  "w", encoding="utf-8") as f:
            json.dump(out_record, f, indent=2)
        print(f"      Saved: zenodo_v16/paper_v16_published.json")
    else:
        print(f"[6/6] Draft created (use --publish to go live)")
        print(f"      Preview: https://zenodo.org/deposit/{new_id}")


if __name__ == "__main__":
    main()
