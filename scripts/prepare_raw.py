#!/usr/bin/env python
"""Normalize complete original CSVs into a disk-backed, chronological store.

Run after scripts/wait_for_inputs.py reports ready_for_full_scan. No legacy
subset, legacy orgID, or legacy IPC numeric ID participates in this pipeline.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import pandas as pd
from patentflow.data import ROOT, applicants, atomic_json, dates, file_hash, load_taxonomy, subclasses, taxonomy_paths
from patentflow.input_gate import snapshot, writable_handles


def prepare(lang):
    source = ROOT / 'datasets' / (lang.upper() + '-all.csv')
    files = [source] + taxonomy_paths(lang)
    before = snapshot(files)
    receipt = json.loads((ROOT / 'state/input_transfer.json').read_text())
    if not receipt['ready_for_full_scan'] or any(receipt['files'].get(str(p)) != before[str(p)] for p in files):
        raise RuntimeError('Inputs not settled or have changed since transfer receipt; rerun wait_for_inputs.py')
    if writable_handles(files):
        raise RuntimeError('Input still open for writing')
    out = ROOT / 'data/raw_processed'
    out.mkdir(parents=True, exist_ok=True)
    final_path = out / (lang + '.sqlite')
    if final_path.exists():
        raise FileExistsError('Refusing to overwrite completed data store: ' + str(final_path))
    stage = out / (lang + '.building.sqlite')
    if stage.exists():
        raise FileExistsError('Partial build exists; inspect and archive before rebuilding: ' + str(stage))
    conn = sqlite3.connect(str(stage))
    conn.execute('PRAGMA journal_mode=WAL')
    conn.execute('PRAGMA synchronous=NORMAL')
    conn.execute('CREATE TABLE patents (source_row INTEGER PRIMARY KEY, fingerprint TEXT UNIQUE NOT NULL, '
                 'date TEXT NOT NULL, year INTEGER NOT NULL, orgs TEXT NOT NULL, subclasses TEXT NOT NULL, '
                 'title TEXT NOT NULL, abstract TEXT NOT NULL, title_missing INTEGER NOT NULL, '
                 'abstract_missing INTEGER NOT NULL)')
    taxonomy = load_taxonomy(lang)
    audit = {'country': lang, 'source': str(source.relative_to(ROOT)), 'formal_source': True,
             'source_kind': 'complete_raw_export', 'rows_read': 0, 'duplicates_exact_content': 0,
             'synthetic_patent_identity': True,
             'identity_note': 'Source has no validated publication number; row identifiers are internal record IDs.',
             'excluded': {}, 'years': {}, 'months': {}, 'taxonomy_missing_subclasses': {},
             'taxonomy': [{'path': str(p.relative_to(ROOT)), 'sha256': file_hash(p)} for p in taxonomy_paths(lang)]}
    exclusions, years, months, unknown = Counter(), Counter(), Counter(), Counter()
    title_missing = abstract_missing = joint = cjk = 0
    organizations, codes_seen = set(), set()
    source_row = 0
    start = time.monotonic()
    quarantine_path = out / (lang + '.quarantine.jsonl')
    with quarantine_path.open('w') as quarantine:
        for chunk in pd.read_csv(source, dtype=str, keep_default_na=False, chunksize=20000):
            if 'classCode' not in chunk or 'publicationDate' not in chunk:
                raise ValueError('Required original columns absent')
            parsed = dates(chunk.publicationDate)
            secondary = dates(chunk.date_parsed) if 'date_parsed' in chunk else None
            batch, metadata = [], []
            for local_i, row in enumerate(chunk.to_dict('records')):
                row_id = source_row
                source_row += 1
                dt = parsed.iloc[local_i]
                orgs = applicants(row.get('applicantName', row.get('orgname', '')))
                codes = subclasses(row.get('classCode', ''))
                title, abstract = row.get('title', '').strip(), row.get('summary', '').strip()
                reason = None
                if pd.isna(dt):
                    reason = 'invalid_publication_date'
                elif secondary is not None and pd.notna(secondary.iloc[local_i]) and dt != secondary.iloc[local_i]:
                    reason = 'conflicting_date_fields'
                elif dt > pd.Timestamp.now().normalize():
                    reason = 'future_publication_date'
                elif not orgs:
                    reason = 'missing_applicant'
                elif not codes:
                    reason = 'missing_valid_ipc_subclass'
                if reason:
                    exclusions[reason] += 1
                    quarantine.write(json.dumps({'source_row': row_id, 'reason': reason}, ensure_ascii=False) + '\n')
                    continue
                date = dt.strftime('%Y-%m-%d')
                value = [date, orgs, codes, title, abstract]
                fingerprint = hashlib.sha256(json.dumps(value, ensure_ascii=False).encode()).hexdigest()
                batch.append((row_id, fingerprint, date, int(dt.year), json.dumps(orgs, ensure_ascii=False),
                              json.dumps(codes), title, abstract, int(not title), int(not abstract)))
                metadata.append((fingerprint, date, orgs, codes, not title, not abstract,
                                 any('\u4e00' <= c <= '\u9fff' for c in abstract)))
            # INSERT OR IGNORE is used only for exact canonical-content duplicates.
            for values, meta in zip(batch, metadata):
                cursor = conn.execute('INSERT OR IGNORE INTO patents VALUES (?,?,?,?,?,?,?,?,?,?)', values)
                if cursor.rowcount == 0:
                    audit['duplicates_exact_content'] += 1
                    continue
                _, date, orgs, codes, no_title, no_abstract, has_cjk = meta
                years[date[:4]] += 1
                months[date[:7]] += 1
                title_missing += no_title
                abstract_missing += no_abstract
                joint += len(orgs) > 1
                cjk += has_cjk
                organizations.update(orgs)
                codes_seen.update(codes)
                unknown.update(c for c in codes if c not in taxonomy)
            conn.commit()
            print(lang, 'rows_read', source_row, 'retained', sum(years.values()),
                  'elapsed_s', round(time.monotonic() - start, 1), flush=True)
    if before != snapshot(files) or writable_handles(files):
        conn.close()
        raise RuntimeError('Source changed during CSV scan; partial store not promoted')
    conn.execute('CREATE INDEX chronological ON patents(date, source_row)')
    conn.execute('CREATE INDEX by_year ON patents(year, date, source_row)')
    conn.commit()
    total = int(conn.execute('SELECT COUNT(*) FROM patents').fetchone()[0])
    if total == 0:
        raise RuntimeError('No usable records')
    audit.update({'rows_read': source_row, 'retained_unique_records': total, 'excluded': dict(exclusions),
                  'years': dict(sorted(years.items())), 'months': dict(sorted(months.items())),
                  'organizations': len(organizations), 'subclasses': len(codes_seen),
                  'observed_subclass_codes': sorted(codes_seen),
                  'title_missing': title_missing, 'abstract_missing': abstract_missing,
                  'joint_applicant_records': joint, 'text_contains_cjk_fraction': cjk / total,
                  'taxonomy_missing_subclasses': dict(sorted(unknown.items())),
                  'sha256': file_hash(source), 'source_snapshot': before[str(source)],
                  'elapsed_seconds': time.monotonic() - start})
    if before != snapshot(files):
        raise RuntimeError('Source changed while hashing')
    conn.execute('CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
    conn.execute('INSERT INTO metadata VALUES (?,?)', ('audit', json.dumps(audit, ensure_ascii=False)))
    conn.commit()
    conn.execute('PRAGMA wal_checkpoint(TRUNCATE)')
    conn.close()
    stage.replace(final_path)
    atomic_json(ROOT / 'reports/raw_audit' / (lang + '.json'), audit)
    print('COMPLETE', lang, json.dumps(audit, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--languages', nargs='+', default=['cn', 'us', 'in'], choices=['cn', 'us', 'in'])
    args = parser.parse_args()
    for lang in args.languages:
        prepare(lang)
