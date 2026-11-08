#!/usr/bin/env python
"""Recover missing original fields by exact matching; export reproducible audits.

Only snapshot files that have not changed while being read. A raw file used for
enrichment is never mistaken for the actual experimental population.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd
from patentflow.data import ROOT, applicants, atomic_json, dates, file_hash, load_taxonomy, signature, subclasses


def run(lang):
    source = ROOT / "datasets" / ("patent_" + lang) / ("all-" + lang + ".csv")
    before = source.stat()
    df = pd.read_csv(source, dtype=str, keep_default_na=False)
    audit = {"source": str(source.relative_to(ROOT)), "rows": len(df),
             "patent_ids": int(df.patentId.nunique()), "sha256": file_hash(source),
             "source_size": before.st_size, "source_mtime_ns": before.st_mtime_ns,
             "legacy_selection_unknown": True}
    # Restore only unambiguous fields. CN code recovery is essential for hierarchy.
    raw = ROOT / "datasets" / (lang.upper() + "-all.csv")
    recovered = {}
    conflicts = set()
    if lang in ("cn", "us") and raw.exists():
        raw_before = raw.stat()
        sigs = set(signature(r, lang) for r in df.to_dict("records"))
        for chunk in pd.read_csv(raw, dtype=str, keep_default_na=False, chunksize=50000):
            for row in chunk.to_dict("records"):
                sig = signature(row, lang)
                if sig not in sigs:
                    continue
                value = (row.get("classCode", ""), row.get("title", ""))
                if sig in recovered and recovered[sig] != value:
                    conflicts.add(sig)
                recovered[sig] = value
        raw_after = raw.stat()
        stable = (raw_before.st_size, raw_before.st_mtime_ns) == (raw_after.st_size, raw_after.st_mtime_ns)
        if not stable:
            if lang == "cn":
                raise RuntimeError("CN original is changing; cannot recover trustworthy IPC mappings")
            recovered = {}
            audit["raw_enrichment"] = {"status": "skipped_changing_file", "start_bytes": raw_before.st_size,
                                       "end_bytes": raw_after.st_size}
        else:
            for sig in conflicts:
                recovered.pop(sig, None)
            audit["raw_enrichment"] = {"status": "stable_exact_match", "path": str(raw.relative_to(ROOT)),
                                       "sha256": file_hash(raw), "ambiguous_signatures": len(conflicts)}
    parsed_dates = dates(df.publicationDate)
    audit["invalid_dates"] = int(parsed_dates.isna().sum())
    rows, excluded = [], Counter()
    mismatches, restored = 0, 0
    for i, row in enumerate(df.to_dict("records")):
        if pd.isna(parsed_dates.iloc[i]):
            excluded["invalid_date"] += 1
            continue
        value = recovered.get(signature(row, lang))
        codes = subclasses(row.get("classCode", "") if lang != "cn" else value[0] if value else "")
        title = row.get("title", "") or (value[1] if value else "")
        restored += int(value is not None)
        orgs = applicants(row["applicantName" if lang == "cn" else "orgname"])
        mismatches += int(len(orgs) != len(set(row["orgID"].split(";"))))
        if not codes or not orgs:
            excluded["missing_ipc" if not codes else "missing_organization"] += 1
            continue
        rows.append({"patent_id": lang + ":" + row["patentId"],
                     "date": parsed_dates.iloc[i].strftime("%Y-%m-%d"),
                     "year": int(parsed_dates.iloc[i].year), "orgs": orgs,
                     "subclasses": codes, "title": title, "abstract": row.get("summary", ""),
                     "source_row": i, "title_missing": not bool(title)})
    audit["excluded"] = dict(excluded)
    audit["exact_match_recovered_rows"] = restored
    audit["organization_count_disagrees_with_legacy_ids"] = mismatches
    audit["retained_rows"] = len(rows)
    audit["year_counts"] = dict(sorted(Counter(r["year"] for r in rows).items()))
    audit["title_missing"] = sum(r["title_missing"] for r in rows)
    audit["joint_patents_original_names"] = sum(len(r["orgs"]) > 1 for r in rows)
    audit["joint_patents_legacy_ids"] = int(df.orgID.str.contains(";").sum())
    audit["organizations"] = len({o for r in rows for o in r["orgs"]})
    audit["subclasses"] = len({c for r in rows for c in r["subclasses"]})
    audit["events_subclass"] = sum(len(r["orgs"]) * len(r["subclasses"]) for r in rows)
    taxonomy = load_taxonomy(lang)
    audit["missing_taxonomy_codes"] = sorted({c for r in rows for c in r["subclasses"] if c not in taxonomy})
    audit["text_contains_cjk_fraction"] = sum(any('\u4e00' <= c <= '\u9fff' for c in r["abstract"])
                                                for r in rows) / max(len(rows), 1)
    after = source.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError("Experimental source changed during audit")
    out = ROOT / "data" / "canonical"
    out.mkdir(parents=True, exist_ok=True)
    rows.sort(key=lambda r: (r["date"], r["patent_id"]))
    path = out / (lang + ".jsonl")
    with path.with_suffix(".tmp").open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    path.with_suffix(".tmp").replace(path)
    audit["canonical_sha256"] = file_hash(path)
    atomic_json(ROOT / "reports" / "input_audit" / (lang + ".json"), audit)
    print(json.dumps(audit, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--languages", nargs="+", default=["cn", "in", "us"])
    args = parser.parse_args()
    for language in args.languages:
        run(language)
