"""Batch evaluator saves full top-40 rankings and per-query metrics."""
import numpy as np
from .metrics import ranking_metrics


def evaluate_queries(data,year,score_function,path,batch_size=128):
    orgs,truth=data.queries(year)
    totals=None
    per_query={}
    rankings=[]
    for start in range(0,len(orgs),batch_size):
        nodes=orgs[start:start+batch_size]
        scores=score_function(nodes)
        result=ranking_metrics(scores,truth[start:start+batch_size],candidate_ids=data.candidate_ids)
        for key,value in result.pop('per_query').items():
            per_query.setdefault(key,[]).append(value)
        rankings.append(data.candidate_ids[np.argsort(-scores,axis=1,kind='stable')[:,:min(40,len(data.candidate_ids))]])
        if totals is None:
            totals=result
        else:
            for key in ['n_queries','n_active','n_empty','oov_positive_count','positive_count']:
                totals[key]+=result[key]
    if totals is None:
        raise ValueError('No queries')
    active=np.array([bool(len(t)) for t in truth])
    arrays={k:np.concatenate(v) for k,v in per_query.items()}
    for key,value in arrays.items():
        totals['active'][key]=float(value[active].mean()) if active.any() else None
        totals['all_zero_filled'][key]=float(value.mean())
    if path is not None:
        np.savez_compressed(path,org=orgs,active=active,ranked_ids=np.concatenate(rankings),**arrays)
    return totals
