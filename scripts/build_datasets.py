#!/usr/bin/env python
"""Build country event stores, annual CSR matrices, and train-selected views."""
import argparse
from array import array
from collections import Counter, defaultdict
from datetime import date
import json
from pathlib import Path
import sqlite3
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
from scipy import sparse
from patentflow.data import ROOT, atomic_json, file_hash, load_taxonomy
from patentflow.history import RecentCollaborators, timestamp_groups
from patentflow.entities import eligible, entity_type


def build(lang):
    source = ROOT / 'data/raw_processed' / (lang + '.sqlite')
    conn = sqlite3.connect('file:' + str(source) + '?mode=ro', uri=True)
    audit = json.loads(conn.execute("SELECT value FROM metadata WHERE key='audit'").fetchone()[0])
    protocol = json.loads((ROOT/'configs/protocol.json').read_text())
    entity_scope = protocol['entity_scope']
    raw_years = sorted(map(int, audit['years']))
    start_year = protocol['observation_start_year'][lang]
    complete_years = [y for y in raw_years if y>=start_year and all('%d-%02d' % (y,m) in audit['months'] for m in range(1,13))]
    test_year = max(y for y in complete_years if y-1 in complete_years and y-2 in complete_years)
    val_year, train_end = test_year - 1, test_year - 2
    years = [y for y in raw_years if start_year<=y<=test_year]
    terminal_complete = raw_years[-1] == test_year
    supervised_years = [y for y in complete_years if y-1 in complete_years]
    if val_year not in supervised_years or test_year not in supervised_years:
        raise ValueError('Validation/test require consecutive years with all-month coverage')
    if train_end <= years[0]:
        raise ValueError('Insufficient complete years for annual supervised forecasting')
    out = ROOT / 'data/processed' / lang
    if (out / 'manifest.json').exists():
        raise FileExistsError('Completed country store exists: ' + str(out))
    out.mkdir(parents=True, exist_ok=True)
    taxonomy = load_taxonomy(lang)
    codes = [c for c in audit['observed_subclass_codes'] if c in taxonomy]
    code_ids = {c: i for i, c in enumerate(codes)}
    org_ids, organizations, first_year, first_day = {}, [], [], []
    srcs, dsts, stamps, pats, weights, event_years = array('i'), array('i'), array('i'), array('q'), array('f'), array('i')
    patent_counts, edge_counts, joint_counts = Counter(), Counter(), Counter()
    train_codes, train_sections = set(), Counter()
    peers = RecentCollaborators(3)
    screening = Counter()
    screened_names = {category:set() for category in ['company','institution','unresolved']}
    current_year = None
    start = time.monotonic()

    def save_peers(year):
        partners = np.full((len(organizations), 3), -1, dtype=np.int32)
        last_dates = np.full_like(partners, -1)
        for org, items in peers.peers.items():
            for j, (peer, stamp) in enumerate(items):
                partners[org, j], last_dates[org, j] = peer, stamp
        np.savez_compressed(out / ('collaborators_%d.npz' % year), partners=partners, last_day=last_dates)

    query = conn.execute('SELECT source_row,date,year,orgs,subclasses FROM patents WHERE year>=? AND year<=? ORDER BY date,source_row', (start_year,test_year))
    for date_str, group in timestamp_groups(query, key=lambda row: row[1]):
        day = (date.fromisoformat(date_str) - date(1970, 1, 1)).days
        year = int(date_str[:4])
        if current_year is not None and year != current_year:
            save_peers(current_year)
            print(lang, 'completed_year', current_year, 'orgs', len(organizations), 'edges', len(srcs), flush=True)
        current_year = year
        daily_orgs = []
        for row_id, _, _, org_json, code_json in group:
            raw_names, raw_codes = json.loads(org_json), json.loads(code_json)
            for name in raw_names:
                screened_names[entity_type(name)].add(name)
            names = [name for name in raw_names if eligible(name, entity_scope)]
            row_codes = [code for code in raw_codes if code in code_ids]
            screening['unknown_dictionary_code_occurrences'] += len(raw_codes)-len(row_codes)
            screening['excluded_applicant_occurrences'] += len(raw_names)-len(names)
            if not row_codes or not names:
                screening['records_no_dictionary_ipc' if not row_codes else 'records_no_eligible_organization'] += 1
                continue
            nodes = []
            for name in names:
                if name not in org_ids:
                    org_ids[name] = len(organizations)
                    organizations.append(name)
                    first_year.append(year)
                    first_day.append(day)
                nodes.append(org_ids[name])
            daily_orgs.append(nodes)
            n_edges = len(nodes) * len(row_codes)
            patent_counts[year] += 1
            joint_counts[year] += len(nodes) > 1
            edge_counts[year] += n_edges
            if year <= train_end:
                train_codes.update(row_codes)
                train_sections.update({section: len(nodes) * sum(c.startswith(section) for c in row_codes)
                                       for section in {c[0] for c in row_codes}})
            for node in nodes:
                for code in row_codes:
                    srcs.append(node); dsts.append(code_ids[code]); stamps.append(day)
                    pats.append(row_id); weights.append(1. / n_edges); event_years.append(year)
        peers.observe_timestamp(day, daily_orgs)
    save_peers(current_year)
    conn.close()
    srcs, dsts, stamps = np.array(srcs), np.array(dsts), np.array(stamps)
    pats, weights, event_years = np.array(pats), np.array(weights), np.array(event_years)
    n_org, n_codes = len(organizations), len(codes)
    np.savez_compressed(out / 'events.npz', src=srcs, dst=dsts, day=stamps, patent_row=pats,
                        weight=weights, year=event_years)
    np.savez_compressed(out / 'organization_first_seen.npz', year=np.array(first_year, dtype=np.int32),
                        day=np.array(first_day, dtype=np.int32))
    atomic_json(out / 'organizations.json', organizations)
    atomic_json(out / 'subclasses.json', codes)
    actual_tree = {}
    for code in codes:
        for level, prefix in [(1, code[:1]), (2, code[:3]), (3, code)]:
            actual_tree[prefix] = taxonomy.get(prefix, {'code': prefix, 'level': level, 'description': '',
                'parent': prefix[:3] if level == 3 else prefix[:1] if level == 2 else None,
                'description_missing': True})
    atomic_json(out / 'observed_taxonomy.json', actual_tree)
    for year in years:
        mask = event_years == year
        matrix = sparse.coo_matrix((np.ones(mask.sum(), dtype=np.float32), (srcs[mask], dsts[mask])),
                                   shape=(n_org, n_codes)).tocsr()
        sparse.save_npz(out / ('counts_%d.npz' % year), matrix)
    section_diversity = Counter(c[0] for c in train_codes)
    top3 = sorted(section_diversity, key=lambda s: (-section_diversity[s], -train_sections[s], s))[:3]
    manifest = {'country': lang, 'source_sha256': audit['sha256'], 'years': years,
                'train_end_year': train_end, 'validation_year': val_year, 'test_year': test_year,
                'observation_start_year':start_year,
                'records_outside_main_window':sum(v for k,v in audit['years'].items() if int(k)<start_year or int(k)>test_year),
                'supervised_target_years': supervised_years, 'complete_coverage_years': complete_years,
                'entity_scope': entity_scope, 'screening': dict(screening),
                'applicant_name_categories':{k:len(v) for k,v in screened_names.items()},
                'protocol_sha256': file_hash(ROOT/'configs/protocol.json'),
                'terminal_year_right_censored': not terminal_complete,
                'calendar_completeness_note': 'December presence is a coverage check, not proof of exhaustive source collection.',
                'n_organizations': n_org, 'n_subclasses': n_codes, 'n_events': len(srcs),
                'record_counts_by_year': dict(patent_counts), 'event_counts_by_year': dict(edge_counts),
                'joint_records_by_year': dict(joint_counts), 'top3_sections': top3,
                'train_section_diversity': dict(section_diversity), 'train_section_events': dict(train_sections),
                'elapsed_seconds': time.monotonic() - start, 'views': []}
    for section in ['All'] + top3:
        used_codes = set(np.unique(dsts).tolist())
        selected = [i for i, c in enumerate(codes) if i in used_codes and (section == 'All' or c[0] == section)]
        candidates = [i for i in selected if codes[i] in train_codes]
        view_name = lang.upper() + '_' + section
        view = {'name': view_name, 'country': lang, 'section': section, 'country_store': lang,
                'subclass_ids': selected, 'train_candidate_ids': candidates,
                'n_subclasses': len(selected), 'n_candidates': len(candidates),
                'train_end_year': train_end, 'validation_year': val_year, 'test_year': test_year,
                'protocol': protocol['version'], 'prediction_level': 3}
        atomic_json(ROOT / 'data/processed/views' / (view_name + '.json'), view)
        manifest['views'].append(view_name)
    atomic_json(out / 'manifest.json', manifest)
    atomic_json(out / 'applicant_screening.json', {k:sorted(v) for k,v in screened_names.items()})
    print('COMPLETE', json.dumps(manifest), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--languages', nargs='+', default=['cn', 'us', 'in'])
    args = p.parse_args()
    for lang in args.languages:
        build(lang)
