#!/usr/bin/env python
"""Statistics and training-only feasibility probes; no test-guided design."""
import argparse
import csv
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
from scipy import sparse
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize
from patentflow.data import ROOT, atomic_json
from patentflow.dataset import DatasetView
from patentflow.metrics import ranking_metrics


def stats(name):
    data = DatasetView(name)
    result = {'dataset': name, 'country': data.spec['country'], 'section': data.spec['section'],
              'subclasses': len(data.codes), 'train_candidates': len(data.candidate_ids), 'splits': {}}
    for split in ['train','validation','test']:
        years = data.split_years(split)
        event_years = [y for y in data.manifest['years'] if y<=data.spec['train_end_year']] if split=='train' else years
        count = sum((data.counts(y) for y in event_years), sparse.csr_matrix((len(data.first_year),len(data.codes))))
        history = data.history(years[0]-1)
        query_count = active_count = cold_count = positive_count = repeat_count = oov_count = 0
        domain_cold = 0
        for year in years:
            orgs, truth = data.queries(year)
            target = data.counts(year)[orgs].sign()
            hist = data.history(year-1)[orgs]
            active = np.asarray(target.sum(axis=1)).ravel() > 0
            query_count += len(orgs)
            active_count += int(active.sum())
            cold_count += int((data.counts(year)[data.first_year == year].sum(axis=1)>0).sum())
            domain_cold += int((active & (np.asarray(hist.sum(axis=1)).ravel()==0)).sum())
            positive_count += target.nnz
            repeat_count += hist.sign().multiply(target).nnz
            oov_count += target.nnz - target[:, data.candidate_ids].nnz
        degrees = np.asarray(count.sum(axis=1)).ravel()
        degrees = degrees[degrees>0]
        result['splits'][split] = {'target_years':years, 'observed_event_years':event_years, 'events':int(count.sum()),
            'active_nodes_in_target_years':int(len(degrees)), 'unique_pairs':int(count.nnz),
            'historically_known_queries':query_count, 'active_queries':active_count,
            'empty_queries':query_count-active_count, 'new_organization_active_queries_excluded':cold_count,
            'active_queries_without_section_history':domain_cold, 'target_positive_count':positive_count,
            'historically_repeated_target_count':repeat_count,
            'repeat_fraction':repeat_count/max(positive_count,1), 'oov_target_count':oov_count,
            'degree_quantiles':dict(zip(['min','median','p90','p99','max'],
                np.quantile(degrees,[0,.5,.9,.99,1]).tolist())) if len(degrees) else {}}
    # One internal training transition; it never accesses validation/test labels.
    year = data.spec['train_end_year']
    orgs, truth = data.queries(year)
    history = data.history(year-1)
    target = data.counts(year)
    active = np.array([bool(len(t)) for t in truth])
    active_orgs = orgs[active]
    peers = data.collaborators(year-1)
    degree = np.asarray(history.sum(axis=1)).ravel()
    buckets = np.floor(np.log2(degree+1)).astype(int)
    pool = {b:np.flatnonzero((buckets==b)&(data.first_year<year)) for b in np.unique(buckets)}
    rng = np.random.RandomState(42)
    peer_available = valid = observed_hits = control_hits = novel_total = 0
    per_query_lift = []
    for org in active_orgs:
        collaborators = peers[org]
        collaborators = collaborators[collaborators>=0]
        if not len(collaborators):
            continue
        peer_available += 1
        own = set(history[org].indices)
        novel = set(target[org].indices)-own
        if not novel:
            continue
        control = []
        for peer in collaborators:
            options = pool[buckets[peer]]
            options = options[options!=org]
            if len(options):
                control.append(int(rng.choice(options)))
        if not control:
            continue
        union = set(history[collaborators].indices)
        random_union = set(history[control].indices)
        observed, random_hit = len(union & novel), len(random_union & novel)
        observed_hits += observed
        control_hits += random_hit
        novel_total += len(novel)
        valid += 1
        per_query_lift.append((observed-random_hit)/len(novel))
    lifts = np.asarray(per_query_lift)
    ci = None
    if len(lifts):
        bootstrap = np.array([rng.choice(lifts,size=len(lifts),replace=True).mean() for _ in range(500)])
        ci = np.quantile(bootstrap,[.025,.975]).tolist()
    feasibility = {'probe_target_year':year, 'uses_only_training_period':True,
        'collaboration':{'active_known_orgs':len(active_orgs), 'with_prior_collaborators':peer_available,
            'coverage':peer_available/max(len(active_orgs),1), 'eligible_novel_label_queries':valid,
            'novel_positive_count':novel_total, 'peer_novel_coverage':observed_hits/max(novel_total,1),
            'degree_bucket_control_novel_coverage':control_hits/max(novel_total,1),
            'macro_coverage_lift':float(lifts.mean()) if len(lifts) else None,
            'organization_bootstrap_95ci':ci,
            'limitation':'observational control matched on log2 history degree; not a causal cooperation effect'}}
    # Behavior community probe uses only earlier history to fit centroids.
    historical = np.flatnonzero(degree>0)
    if len(historical)>=32:
        normalized = normalize(history, norm='l1')
        km = MiniBatchKMeans(n_clusters=32, random_state=42, n_init=3, batch_size=2048)
        km.fit(normalized[historical])
        eval_nodes = active_orgs[degree[active_orgs]>0]
        # Bound probe computation; deterministic sample is reported, not a final experiment.
        if len(eval_nodes)>20000:
            eval_nodes = np.sort(rng.choice(eval_nodes,20000,replace=False))
        cluster = km.predict(normalized[eval_nodes])
        score = km.cluster_centers_[cluster][:,data.candidate_ids]
        truth = [target[o].indices for o in eval_nodes]
        cluster_metric = ranking_metrics(score,truth,candidate_ids=data.candidate_ids)
        global_score = np.broadcast_to(np.asarray(history.sum(axis=0)).ravel()[data.candidate_ids],score.shape)
        global_metric = ranking_metrics(global_score,truth,candidate_ids=data.candidate_ids)
        feasibility['behavior_communities'] = {'probe':'32-cluster past IPC frequency centroids',
            'n_historical_orgs':len(historical), 'n_eval_orgs':len(eval_nodes),
            'cluster_size_quantiles':np.quantile(np.bincount(km.labels_,minlength=32),[0,.5,1]).tolist(),
            'centroid_metrics':cluster_metric['active'], 'global_frequency_metrics':global_metric['active'],
            'limitation':'evidence for local business-profile structure, not validation of the proposed Fluxion module'}
    atomic_json(ROOT/'reports/dataset_statistics'/(name+'.json'), result)
    atomic_json(ROOT/'reports/feasibility'/(name+'.json'), feasibility)
    print('COMPLETE', name, 'train_probe_collaboration',json.dumps(feasibility['collaboration']),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--datasets',nargs='+')
    args=p.parse_args()
    names=args.datasets or [p.stem for p in sorted((ROOT/'data/processed/views').glob('*.json'))]
    if not names:
        raise RuntimeError('No prepared views')
    for name in names:
        stats(name)
