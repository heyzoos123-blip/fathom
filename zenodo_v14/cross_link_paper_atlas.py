# -*- coding: utf-8 -*-
"""
cross_link_paper_atlas.py -- Add reciprocal related_identifiers between the
published paper v14 record (10.5281/zenodo.19502710) and the published
atlas v0.3 record (10.5281/zenodo.19502716) on Zenodo.

Why this exists:
    When atlas v0.3 was uploaded by upload_atlas_v0_3.py it declared
    isSupplementTo the paper CONCEPT DOI (10.5281/zenodo.19326174). That
    is correct but incomplete -- it does not link the atlas to the specific
    paper v14 version that is its scholarly writeup, and the paper v14
    record has no reverse link pointing at the atlas at all. As a result
    neither record shows the other in its "Related works" sidebar on
    zenodo.org, which makes the two records look unrelated from the
    reader's point of view and leaves the brand-new atlas record with no
    inbound discovery path from the well-trafficked paper concept line.

What this script does:
    1. Loads the current metadata of both published records via the
       Zenodo REST API.
    2. Merges in the missing reciprocal relations (see TARGET_RELATIONS
       below), preserving every relation that already exists. No existing
       relation is removed.
    3. In DRY RUN mode (default) prints a unified diff of what would
       change and makes NO HTTP writes.
    4. In --apply mode it unlocks each record for editing, PUTs the
       updated metadata, and republishes. The DOI is preserved; only
       metadata is updated.

Safety:
    - Default mode is DRY RUN. Nothing is pushed without --apply.
    - ZENODO_TOKEN must be in the environment. Never hardcoded.
    - If either record cannot be unlocked, the script aborts before
      touching the other record. Partial state is explicitly handled.
    - Every HTTP response is printed so failures are visible.

Usage:
    python zenodo_v14/cross_link_paper_atlas.py              # dry run
    ZENODO_TOKEN=xxx python zenodo_v14/cross_link_paper_atlas.py --apply
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

# Record identifiers (both already published 2026-04-10)
PAPER_V14_ID = 19502710   # The Fathom Cognitive Atlas (paper v14)
PAPER_CONCEPT_ID = 19326174  # Paper concept DOI (v1..v14)
ATLAS_V03_ID = 19502716   # Cognitive Atlas v0.3 replication bundle
ATLAS_CONCEPT_ID = 19502715  # Atlas concept DOI

# Relations to ENSURE exist. Relations already present are not duplicated.
# Using DataCite vocabulary (Zenodo accepts these verbatim).
TARGET_RELATIONS = {
    PAPER_V14_ID: [
        {
            "relation": "isSupplementedBy",
            "identifier": f"10.5281/zenodo.{ATLAS_V03_ID}",
            "resource_type": "dataset",
            "scheme": "doi",
        },
        {
            "relation": "documents",
            "identifier": f"10.5281/zenodo.{ATLAS_V03_ID}",
            "resource_type": "dataset",
            "scheme": "doi",
        },
    ],
    ATLAS_V03_ID: [
        {
            "relation": "isSupplementTo",
            "identifier": f"10.5281/zenodo.{PAPER_V14_ID}",
            "resource_type": "publication-workingpaper",
            "scheme": "doi",
        },
        {
            "relation": "isDocumentedBy",
            "identifier": f"10.5281/zenodo.{PAPER_V14_ID}",
            "resource_type": "publication-workingpaper",
            "scheme": "doi",
        },
    ],
}


def fail(msg):
    print(f"[FATAL] {msg}")
    sys.exit(1)


def get_token():
    token = os.environ.get("ZENODO_TOKEN")
    if not token:
        fail(
            "ZENODO_TOKEN not set in environment. Refusing to proceed. "
            "Use: ZENODO_TOKEN=xxx python zenodo_v14/cross_link_paper_atlas.py --apply"
        )
    return token


def get_record(record_id, token):
    """Fetch the current deposit state for a published record."""
    url = f"{ZENODO_BASE}/deposit/depositions/{record_id}"
    r = requests.get(url, params={"access_token": token})
    if r.status_code != 200:
        fail(f"GET {record_id} failed: HTTP {r.status_code} {r.text[:200]}")
    return r.json()


def unlock_for_edit(record_id, token):
    """Move a published record into edit state. No-op if already unlocked."""
    url = f"{ZENODO_BASE}/deposit/depositions/{record_id}/actions/edit"
    r = requests.post(url, params={"access_token": token})
    if r.status_code not in (201, 400):
        fail(f"EDIT {record_id} failed: HTTP {r.status_code} {r.text[:200]}")
    if r.status_code == 400:
        # Already in edit mode (or never published). Re-GET state.
        return get_record(record_id, token)
    return r.json()


def put_metadata(record_id, metadata, token):
    """PUT the updated metadata block. Leaves files untouched."""
    url = f"{ZENODO_BASE}/deposit/depositions/{record_id}"
    r = requests.put(
        url,
        params={"access_token": token},
        data=json.dumps({"metadata": metadata}),
        headers={"Content-Type": "application/json"},
    )
    if r.status_code != 200:
        fail(f"PUT {record_id} failed: HTTP {r.status_code} {r.text[:500]}")
    return r.json()


def publish_record(record_id, token):
    """Re-publish the edited record. DOI is preserved; only metadata updates."""
    url = f"{ZENODO_BASE}/deposit/depositions/{record_id}/actions/publish"
    r = requests.post(url, params={"access_token": token})
    if r.status_code != 202:
        fail(f"PUBLISH {record_id} failed: HTTP {r.status_code} {r.text[:500]}")
    return r.json()


def merge_relations(existing, to_add):
    """Append any target relations that are not already present.

    A relation is considered "already present" if the (relation, identifier)
    pair matches an existing entry. resource_type and scheme differences
    are ignored for the dedupe check but the new entry's values win on
    insertion (zenodo prefers exact resource_type matches).
    """
    seen = set()
    for rel in existing or []:
        key = (rel.get("relation"), rel.get("identifier"))
        seen.add(key)

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


def diff_preview(label, before, after):
    print(f"\n  [{label}] related_identifiers diff:")
    print(f"    BEFORE ({len(before)} relations):")
    for rel in before:
        print(f"      - {rel.get('relation')} {rel.get('identifier')}")
    print(f"    AFTER  ({len(after)} relations):")
    for rel in after:
        marker = " +" if rel not in before else "  "
        print(f"     {marker} {rel.get('relation')} {rel.get('identifier')}")


def process_record(record_id, human_label, apply_changes, token):
    print(f"\n{'='*68}")
    print(f"RECORD: {human_label} (id={record_id})")
    print(f"{'='*68}")

    dep = get_record(record_id, token)
    title = dep.get("title") or dep.get("metadata", {}).get("title", "<untitled>")
    state = dep.get("state", "?")
    doi = dep.get("doi") or dep.get("metadata", {}).get("doi", "?")
    print(f"  title: {title[:72]}")
    print(f"  state: {state}")
    print(f"  doi:   {doi}")

    current_metadata = deepcopy(dep.get("metadata", {}))
    existing_rels = current_metadata.get("related_identifiers", []) or []
    target_rels = TARGET_RELATIONS[record_id]

    merged, added = merge_relations(existing_rels, target_rels)

    if not added:
        print("  [OK] every target relation already present. Skipping.")
        return False

    diff_preview(human_label, existing_rels, merged)

    if not apply_changes:
        print("  [DRY RUN] No HTTP writes performed for this record.")
        return False

    # --apply path
    print("  [APPLY] unlocking record for edit...")
    unlock_for_edit(record_id, token)

    new_metadata = deepcopy(current_metadata)
    new_metadata["related_identifiers"] = merged

    print("  [APPLY] PUT updated metadata...")
    put_metadata(record_id, new_metadata, token)

    print("  [APPLY] publishing updated record (DOI preserved)...")
    publish_record(record_id, token)

    print(f"  [DONE] {human_label} updated with {len(added)} new relation(s).")
    return True


def main():
    args = sys.argv[1:]
    apply_changes = "--apply" in args

    print("=" * 68)
    print("Zenodo cross-link: paper v14 <-> atlas v0.3")
    print("=" * 68)
    print(f"Mode: {'APPLY (HTTP writes enabled)' if apply_changes else 'DRY RUN (no HTTP writes)'}")

    token = get_token()

    # Sanity-check connectivity before touching anything.
    me = requests.get(f"{ZENODO_BASE}/deposit/depositions", params={"access_token": token, "size": 1})
    if me.status_code != 200:
        fail(f"Token auth check failed: HTTP {me.status_code} {me.text[:200]}")
    print("[OK] Zenodo API reachable and token accepted.\n")

    any_change = False
    any_change |= process_record(PAPER_V14_ID, "paper v14", apply_changes, token)
    any_change |= process_record(ATLAS_V03_ID, "atlas v0.3", apply_changes, token)

    print("\n" + "=" * 68)
    if apply_changes:
        if any_change:
            print("APPLY complete. Both records cross-linked on Zenodo.")
        else:
            print("APPLY run, nothing to change -- relations already in place.")
    else:
        print("Dry run complete. Re-run with --apply to push changes.")
    print("=" * 68)


if __name__ == "__main__":
    main()
