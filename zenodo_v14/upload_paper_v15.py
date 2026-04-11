# -*- coding: utf-8 -*-
"""
upload_paper_v15.py -- Publish Fathom Paper v15 to Zenodo as a UNIFIED
release that contains both the paper and the Fathom Cognitive Atlas v0.3
data bundle as sibling files on a single record.

Why v15 exists:
    v14 published the paper as one record (10.5281/zenodo.19502710) and
    the Fathom Cognitive Atlas v0.3 as a separate standalone concept
    (10.5281/zenodo.19502715 / 19502716). That split left the new atlas
    record at zero downloads on day one and made the Fathom release look
    fragmented. v15 corrects the split by branching a new version on the
    existing paper concept (10.5281/zenodo.19326174) and attaching BOTH
    fathom_paper_v14.zip AND fathom_atlas_v0.3.zip as files on the same
    record. Downloads and citations accrue to the single concept DOI that
    has held the entire v1 -> v14 history.

What it does:
    1. POST /newversion on v14 (19502710) to create a new draft record.
    2. DELETE every file inherited from v14 (we want a clean slate and we
       upload both bundles explicitly).
    3. PUT the paper bundle (fathom_paper_v14.zip) to the draft bucket.
    4. PUT the atlas bundle (fathom_atlas_v0.3.zip) to the same bucket.
    5. PUT updated metadata that describes v15 as the unified release,
       keeping the isNewVersionOf relation to v14.
    6. POST /publish (only with --publish) to go live.

Safety:
    - Default mode is DRY RUN. No HTTP writes.
    - --draft creates the draft and uploads the files but does NOT publish.
    - --publish does the full flow.
    - ZENODO_TOKEN must be in the environment. Never hardcoded.
    - Both bundle files must exist before any write action.

Usage:
    python zenodo_v14/upload_paper_v15.py                # dry run
    ZENODO_TOKEN=xxx python zenodo_v14/upload_paper_v15.py --draft
    ZENODO_TOKEN=xxx python zenodo_v14/upload_paper_v15.py --publish
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
# Hard-coded to absolute paths so the script runs correctly from the
# worktree without depending on the worktree having the .zip copies.
PAPER_BUNDLE = Path(r"C:\Users\heyzo\clawd\sae-reasoning-depth\zenodo_v14\fathom_paper_v14.zip")
ATLAS_BUNDLE = Path(r"C:\Users\heyzo\clawd\sae-reasoning-depth\zenodo_v14\fathom_atlas_v0.3.zip")

PAPER_BUNDLE_NAME = "fathom_paper_v14.zip"
ATLAS_BUNDLE_NAME = "fathom_atlas_v0.3.zip"

ZENODO_BASE = "https://zenodo.org/api"

CONCEPT_ID = 19326174       # Fathom paper concept DOI (v1..v14 history)
LATEST_V14_ID = 19502710    # v14: "Cognitive Measurement Instruments..."
ORPHAN_ATLAS_ID = 19502716  # The standalone atlas v0.3 record being absorbed
ORPHAN_ATLAS_CONCEPT = 19502715


METADATA = {
    "metadata": {
        "upload_type": "publication",
        "publication_type": "workingpaper",
        "title": (
            "Fathom: Cognitive Measurement Instruments for Transformer "
            "Internals via SAE Feature Coherence Geometry"
        ),
        "creators": [
            {"name": "Rodabaugh, Alexander", "affiliation": "Fathom Intelligence"},
        ],
        "description": (
            "<p><strong>v15 -- Unified release.</strong> This version "
            "consolidates the Fathom paper and the Fathom Cognitive Atlas "
            "v0.3 data bundle into a single record under the existing "
            "Fathom concept DOI "
            "(<code>10.5281/zenodo.19326174</code>). Both "
            "<code>fathom_paper_v14.zip</code> and "
            "<code>fathom_atlas_v0.3.zip</code> are attached as sibling "
            "files on this record so the paper, the writeup of the "
            "Cognitive Atlas, and the underlying probe set / captures / "
            "pre-registration / verify scripts are all citable together "
            "under a single DOI.</p>"
            "<h3>Paper content (unchanged from v14)</h3>"
            "<p>Fathom is a suite of cognitive measurement instruments for "
            "transformer internals built on sparse autoencoder (SAE) feature "
            "coherence geometry. v14 introduced the honesty axis "
            "<em>D = cos(h<sup>(L)</sup>, W<sub>U</sub>[y])</em>, the cosine "
            "similarity between the final-layer residual stream and the "
            "unembedding row of the chosen token. D is computable on any "
            "transformer with a logprob or residual stream interface, with "
            "no SAE required.</p>"
            "<h3>Measurement</h3>"
            "<ul>"
            "<li>Commitment intensity S via inverse participation ratio. "
            "Pooled d = 0.494 across three independent designs "
            "(Fisher p = 0.0008).</li>"
            "<li>Head-to-head beat over logit entropy "
            "(AUC 0.663 vs 0.607, p = 0.013).</li>"
            "<li>Three SAE-free probes across four architectures fail "
            "(Fisher p &gt; 0.6).</li>"
            "</ul>"
            "<h3>Diagnosis</h3>"
            "<ul>"
            "<li>Gini coefficient of per-layer coherence achieves "
            "AUC = 0.685 on n = 200 TruthfulQA pairs, while scalar "
            "depth (K) is at chance.</li>"
            "<li>Cognitive autopsy localizes the fault: layer 11 is the "
            "primary locus in 63% of hallucinations (p = 2.7e-12).</li>"
            "</ul>"
            "<h3>Intervention</h3>"
            "<ul>"
            "<li>Best-of-N sampling scored by Cognitive Health Score "
            "improves on 100% of test prompts vs greedy.</li>"
            "<li>Coordinated suppression of 220 hallucination-associated "
            "features across 22 layers produces 33% first-token changes "
            "on held-out data.</li>"
            "</ul>"
            "<h3>Honesty axis (new in v14)</h3>"
            "<ul>"
            "<li>Cross-architecture D-axis pilot on 30 matched "
            "direct/adversarial prompt pairs across six models from three "
            "families (Gemma-2-2B, Llama-3.2-3B, Qwen2.5-3B, each plus "
            "and minus RLHF). Fisher combined p &lt; 10<sup>-14</sup>.</li>"
            "<li>In the pre-registered early window (tokens 0-6), the "
            "adversarial-framing direction replicates on all six models, "
            "with the strongest single signal at d<sub>z</sub> = -1.86 "
            "(p = 4.6e-11) on Qwen2.5-3B base.</li>"
            "<li>Preliminary orthogonality data indicates D correlates "
            "strongly with C_delta (r ~ 0.99 on paired deltas); D is "
            "therefore reframed as an SAE-free proxy for late-layer "
            "commitment geometry rather than as an independent axis.</li>"
            "</ul>"
            "<h3>Fathom Cognitive Atlas v0.3 (included in this release)</h3>"
            "<p>The Fathom Cognitive Atlas v0.3 is the first public "
            "cross-architecture cognitive state atlas of open-weight "
            "language models. For each model we run a fixed probe set of "
            "90 prompts across 6 categories (retrieval, reasoning, refusal, "
            "creative, adversarial, hallucination) and capture the full "
            "per-token cognitive trajectory (D, entropy, logprob, "
            "top2_margin) for every (model, prompt). "
            "The <code>fathom_atlas_v0.3.zip</code> file attached to this "
            "record contains:</p>"
            "<ul>"
            "<li>12 open-model captures (6 base + 6 instruct) across 3 "
            "families (Gemma, Llama, Qwen) and 2 generations "
            "(Gemma-2/Gemma-3, Llama-3.2 3B/1B, Qwen2.5 3B/1.5B).</li>"
            "<li>1 exploratory closed-model capture (OpenAI GPT-4o-mini "
            "via logprob API).</li>"
            "<li>Probe set v0.1 (90 prompts, SHA-pinned, hand-curated, "
            "versioned).</li>"
            "<li>Analysis scripts (atlas_capture.py, atlas_analyze.py, "
            "atlas_attractor.py, attractor_bootstrap.py, "
            "validate_entropy_estimator.py). All reproducible from "
            "committed JSON in under a minute on CPU.</li>"
            "<li>Sealed pre-registration "
            "(<code>PREREG_v0.3_attractor_replication.md</code>) "
            "committed as git <code>e32cc75</code> 93 minutes before any "
            "v0.3 data was captured.</li>"
            "</ul>"
            "<h3>Atlas v0.3 headline result (pre-registered)</h3>"
            "<p>RLHF Attractor direction prediction, n = 6 model family "
            "pairs. Sealed decision rule: H1 SUPPORTED iff "
            "(mean entropy early-window LOO cosine &ge; 0.40) AND "
            "(permutation p &lt; 0.05, one-sided, 2000 perms) AND "
            "(bootstrap 95% CI lower bound &gt; 0). "
            "At n = 6: mean LOO cosine = <strong>+0.769</strong>, "
            "permutation p = <strong>0.0315</strong>, bootstrap 95% CI "
            "= <strong>[+0.571, +0.869]</strong>. All three conditions "
            "passed without modification. All 6 of 6 families show "
            "positive LOO cosine in the pre-registered early window.</p>"
            "<h3>Supersedes</h3>"
            "<p>This v15 unified release supersedes the separate atlas "
            "standalone record "
            "(<code>10.5281/zenodo.19502716</code>, concept "
            "<code>10.5281/zenodo.19502715</code>), which is retained for "
            "reference but contains the same atlas bundle file.</p>"
            "<h3>License</h3>"
            "<p>CC-BY-4.0 on the paper and atlas data. MIT on the analysis "
            "code.</p>"
            "<p>Related provisional patents: "
            "US 64/020,489 . 64/021,113 . 64/026,964.</p>"
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
                "identifier": f"10.5281/zenodo.{LATEST_V14_ID}",
                "resource_type": "publication-workingpaper",
                "scheme": "doi",
            },
            {
                "relation": "hasPart",
                "identifier": f"10.5281/zenodo.{ORPHAN_ATLAS_ID}",
                "resource_type": "dataset",
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
    print("UPLOAD: Fathom Paper v15 to Zenodo -- UNIFIED RELEASE")
    print("=" * 68)
    print(f"Paper bundle: {PAPER_BUNDLE.name}")
    print(f"  size:       {PAPER_BUNDLE.stat().st_size:,} bytes"
          if PAPER_BUNDLE.exists() else "  MISSING")
    print(f"Atlas bundle: {ATLAS_BUNDLE.name}")
    print(f"  size:       {ATLAS_BUNDLE.stat().st_size:,} bytes"
          if ATLAS_BUNDLE.exists() else "  MISSING")
    print(f"Concept DOI:  10.5281/zenodo.{CONCEPT_ID}")
    print(f"Branch from:  10.5281/zenodo.{LATEST_V14_ID} (v14)")
    print(f"Absorbs:      10.5281/zenodo.{ORPHAN_ATLAS_ID} "
          f"(orphan atlas record, absorbed via hasPart)")
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
        print("  related_identifiers:")
        for rel in METADATA['metadata']['related_identifiers']:
            print(f"    - {rel['relation']} {rel['identifier']}")
        print(f"  files:")
        print(f"    + {PAPER_BUNDLE_NAME} ({PAPER_BUNDLE.stat().st_size:,} bytes)")
        print(f"    + {ATLAS_BUNDLE_NAME} ({ATLAS_BUNDLE.stat().st_size:,} bytes)")
        print()
        print("[DRY RUN] To draft (uploads files, does NOT publish):")
        print("  ZENODO_TOKEN=<token> python zenodo_v14/upload_paper_v15.py --draft")
        print("[DRY RUN] To publish:")
        print("  ZENODO_TOKEN=<token> python zenodo_v14/upload_paper_v15.py --publish")
        return

    if not PAPER_BUNDLE.exists():
        fail(f"Paper bundle missing: {PAPER_BUNDLE}")
    if not ATLAS_BUNDLE.exists():
        fail(f"Atlas bundle missing: {ATLAS_BUNDLE}")

    token = os.environ.get("ZENODO_TOKEN", "").strip()
    if not token:
        fail("ZENODO_TOKEN env var not set. Refusing to run.")

    headers = {"Authorization": f"Bearer {token}"}

    # 1. Create new version draft from v14
    print(f"[1/6] Creating new version from v14 (ID: {LATEST_V14_ID})...")
    r = requests.post(
        f"{ZENODO_BASE}/deposit/depositions/{LATEST_V14_ID}/actions/newversion",
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

    # 2. Get current draft state (for bucket URL + existing files to delete)
    print(f"[2/6] Clearing files inherited from v14 ...")
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

    # 3. Upload paper bundle
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

    # 4. Upload atlas bundle
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

    # 5. Set metadata
    print(f"[5/6] Setting metadata ...")
    r = requests.put(
        f"{ZENODO_BASE}/deposit/depositions/{new_id}",
        json=METADATA,
        headers={**headers, "Content-Type": "application/json"},
    )
    if r.status_code != 200:
        fail(f"set metadata failed: {r.status_code} {r.text[:300]}")
    print(f"      metadata set for {new_id}")

    # 6. Publish or leave as draft
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
            "supersedes": f"10.5281/zenodo.{LATEST_V14_ID}",
            "absorbs": f"10.5281/zenodo.{ORPHAN_ATLAS_ID}",
            "bundles": [PAPER_BUNDLE_NAME, ATLAS_BUNDLE_NAME],
        }
        ROOT = Path(__file__).resolve().parent.parent
        with open(ROOT / "zenodo_v14" / "paper_v15_published.json",
                  "w", encoding="utf-8") as f:
            json.dump(out_record, f, indent=2)
        print(f"      Saved: zenodo_v14/paper_v15_published.json")
    else:
        print(f"[6/6] Draft created (use --publish to go live)")
        print(f"      Preview: https://zenodo.org/deposit/{new_id}")


if __name__ == "__main__":
    main()
