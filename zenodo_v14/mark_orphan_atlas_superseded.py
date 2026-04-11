# -*- coding: utf-8 -*-
"""
mark_orphan_atlas_superseded.py -- Edit the published metadata of the
standalone Fathom Cognitive Atlas v0.3 record (10.5281/zenodo.19502716)
to clearly mark it as superseded by paper v15 (10.5281/zenodo.19504993).

Why:
    Paper v15 is a unified release that contains both the paper and the
    Fathom Cognitive Atlas v0.3 data bundle as sibling files on a single
    record under the existing Fathom concept DOI (10.5281/zenodo.19326174,
    where all v1..v14 download history accumulates). The standalone atlas
    record (19502716) was a day-one orphan with zero downloads. Zenodo
    does not allow self-delete of published records, but we can edit the
    metadata in place (DOI preserved) to:
      1. Prepend a "SUPERSEDED" banner to the description pointing users
         at paper v15.
      2. Add an isSupersededBy related_identifier pointing at paper v15
         (the standard DataCite relation for deprecated records).

Safety:
    - Default mode is DRY RUN. No HTTP writes.
    - --apply enables writes.
    - ZENODO_TOKEN must be in the environment. Never hardcoded.
    - The existing isSupplementTo relations (to 19326174 concept and
      19502710 v14) are preserved, not removed.
    - The file contents of the atlas record are NOT touched.

Usage:
    python zenodo_v14/mark_orphan_atlas_superseded.py
    ZENODO_TOKEN=xxx python zenodo_v14/mark_orphan_atlas_superseded.py --apply
"""

import json
import os
import sys
from copy import deepcopy

try:
    import requests
except ImportError:
    print("[FATAL] requests package not installed")
    sys.exit(1)


ZENODO_BASE = "https://zenodo.org/api"

ORPHAN_ATLAS_ID = 19502716       # standalone atlas v0.3 record
PAPER_V15_ID = 19504993          # unified paper v15 (supersedes the orphan)
PAPER_CONCEPT_ID = 19326174      # paper concept DOI (where downloads live)


SUPERSEDED_BANNER = (
    "<div style='border:2px solid #c62828;background:#fff3e0;padding:12px "
    "16px;margin-bottom:16px;font-weight:bold;'>"
    "<span style='color:#c62828;'>SUPERSEDED -- "
    "2026-04-11</span><br>"
    "This standalone dataset record has been absorbed into "
    "<strong>Fathom Paper v15</strong>, a unified release that contains "
    "both the Fathom paper and this atlas data bundle as sibling files "
    "on a single record under the existing Fathom concept DOI.<br><br>"
    "<strong>Please cite and download the unified record instead:</strong>"
    "<br>"
    "Paper v15: <a href='https://doi.org/10.5281/zenodo.19504993'>"
    "10.5281/zenodo.19504993</a><br>"
    "Concept DOI (always resolves to latest): "
    "<a href='https://doi.org/10.5281/zenodo.19326174'>"
    "10.5281/zenodo.19326174</a><br><br>"
    "The file attached to this standalone record "
    "(<code>fathom_atlas_v0.3.zip</code>) is byte-identical to the one "
    "included in paper v15. This standalone record remains published for "
    "archival continuity only."
    "</div>"
)


TARGET_RELATIONS = [
    # Point at paper v15 as the obsoleting/superseding record.
    # Zenodo accepts DataCite "isObsoletedBy" (not "isSupersededBy").
    {
        "relation": "isObsoletedBy",
        "identifier": f"10.5281/zenodo.{PAPER_V15_ID}",
        "resource_type": "publication-workingpaper",
        "scheme": "doi",
    },
    # Also mark the orphan as a part of the paper v15 whole.
    {
        "relation": "isPartOf",
        "identifier": f"10.5281/zenodo.{PAPER_V15_ID}",
        "resource_type": "publication-workingpaper",
        "scheme": "doi",
    },
]


def fail(msg):
    print(f"[FATAL] {msg}")
    sys.exit(1)


def get_token():
    token = os.environ.get("ZENODO_TOKEN", "").strip()
    if not token:
        fail("ZENODO_TOKEN env var not set")
    return token


def get_record(record_id, token):
    r = requests.get(
        f"{ZENODO_BASE}/deposit/depositions/{record_id}",
        params={"access_token": token},
    )
    if r.status_code != 200:
        fail(f"GET {record_id} failed: HTTP {r.status_code} {r.text[:200]}")
    return r.json()


def unlock_for_edit(record_id, token):
    r = requests.post(
        f"{ZENODO_BASE}/deposit/depositions/{record_id}/actions/edit",
        params={"access_token": token},
    )
    if r.status_code not in (201, 400):
        fail(f"EDIT {record_id} failed: HTTP {r.status_code} {r.text[:200]}")
    if r.status_code == 400:
        return get_record(record_id, token)
    return r.json()


def put_metadata(record_id, metadata, token):
    r = requests.put(
        f"{ZENODO_BASE}/deposit/depositions/{record_id}",
        params={"access_token": token},
        data=json.dumps({"metadata": metadata}),
        headers={"Content-Type": "application/json"},
    )
    if r.status_code != 200:
        fail(f"PUT {record_id} failed: HTTP {r.status_code} {r.text[:500]}")
    return r.json()


def publish_record(record_id, token):
    r = requests.post(
        f"{ZENODO_BASE}/deposit/depositions/{record_id}/actions/publish",
        params={"access_token": token},
    )
    if r.status_code != 202:
        fail(f"PUBLISH {record_id} failed: HTTP {r.status_code} {r.text[:500]}")
    return r.json()


def merge_relations(existing, to_add):
    seen = set()
    for rel in existing or []:
        seen.add((rel.get("relation"), rel.get("identifier")))
    merged = list(existing or [])
    added = []
    for rel in to_add:
        key = (rel["relation"], rel["identifier"])
        if key in seen:
            continue
        merged.append(rel)
        added.append(rel)
        seen.add(key)
    return merged, added


def main():
    args = sys.argv[1:]
    apply_changes = "--apply" in args

    print("=" * 68)
    print("Mark orphan atlas v0.3 record as SUPERSEDED by paper v15")
    print("=" * 68)
    print(f"Target record: 10.5281/zenodo.{ORPHAN_ATLAS_ID}")
    print(f"Superseded by: 10.5281/zenodo.{PAPER_V15_ID}")
    print(f"Mode:          {'APPLY' if apply_changes else 'DRY RUN'}")
    print()

    token = get_token()

    dep = get_record(ORPHAN_ATLAS_ID, token)
    title = dep.get("title") or dep.get("metadata", {}).get("title", "<untitled>")
    state = dep.get("state")
    print(f"Current state: {state}")
    print(f"Current title: {title[:70]}")

    current_metadata = deepcopy(dep.get("metadata", {}))
    current_desc = current_metadata.get("description", "") or ""
    current_rels = current_metadata.get("related_identifiers", []) or []

    # Prepend the superseded banner to the description. Avoid double-
    # prepending if the banner is already there (idempotent).
    already_superseded = "SUPERSEDED" in current_desc and str(PAPER_V15_ID) in current_desc
    if already_superseded:
        new_desc = current_desc
        desc_changed = False
    else:
        new_desc = SUPERSEDED_BANNER + current_desc
        desc_changed = True

    merged_rels, added_rels = merge_relations(current_rels, TARGET_RELATIONS)

    print()
    print("PLANNED CHANGES:")
    print(f"  description banner added: {desc_changed}")
    print(f"  new relations to add:    {len(added_rels)}")
    for rel in added_rels:
        print(f"    + {rel['relation']} {rel['identifier']}")
    if not desc_changed and not added_rels:
        print("  [OK] nothing to do -- already superseded.")
        return

    if not apply_changes:
        print()
        print("[DRY RUN] No HTTP writes performed.")
        print("  Re-run with --apply to push.")
        return

    # APPLY path
    print()
    print("[APPLY] unlocking record for edit...")
    unlock_for_edit(ORPHAN_ATLAS_ID, token)

    new_metadata = deepcopy(current_metadata)
    new_metadata["description"] = new_desc
    new_metadata["related_identifiers"] = merged_rels

    print("[APPLY] PUT updated metadata...")
    put_metadata(ORPHAN_ATLAS_ID, new_metadata, token)

    print("[APPLY] publishing updated record (DOI preserved)...")
    publish_record(ORPHAN_ATLAS_ID, token)

    print()
    print("[DONE] Orphan atlas record marked as superseded.")


if __name__ == "__main__":
    main()
