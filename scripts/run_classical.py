#!/usr/bin/env python
"""Train genuine linear SVM, random forest, or XGBoost on causal pair features."""
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier
from patentflow.classical import FEATURE_NAMES,PairFeatures,sampled_training_pairs
from patentflow.data import ROOT,atomic_json,file_hash
from patentflow.dataset import DatasetView
from patentflow.evaluate import evaluate_queries
from patentflow.resources import ResourceMonitor


def run(name,method,seed,level,max_pairs,threads):
    out=ROOT/'results/baselines'/name/('%s_l%d_s%d'%(method,level,seed))
    if (out/'metrics.json').exists():
        return
    out.mkdir(parents=True,exist_ok=True)
    start=time.monotonic()
    data=DatasetView(name,level)
    with ResourceMonitor(out/'resources.jsonl') as monitor:
        x,y=sampled_training_pairs(data,seed,max_pairs)
        if len(np.unique(y))<2:
            raise ValueError('Training sample has only one class')
        settings={'SVM':[.1,1.], 'RF':[2,8], 'XGBoost':[3,6]}[method]
        trials=[]
        best_score=-float('inf');best_model=None;best_setting=None
        validation_year=data.split_years('validation')[0]
        val_features=PairFeatures(data,validation_year-1)
        for setting in settings:
            if method=='SVM':
                model=make_pipeline(StandardScaler(),LinearSVC(C=setting,dual=False,max_iter=5000,random_state=seed,class_weight='balanced'))
            elif method=='RF':
                model=RandomForestClassifier(n_estimators=150,max_depth=16,min_samples_leaf=setting,
                    max_features=.8,n_jobs=threads,random_state=seed,class_weight='balanced_subsample')
            else:
                model=XGBClassifier(n_estimators=300,max_depth=setting,learning_rate=.05,subsample=.8,
                    colsample_bytree=.9,n_jobs=threads,random_state=seed,tree_method='hist',eval_metric='logloss',
                    scale_pos_weight=float((y==0).sum()/max((y==1).sum(),1)))
            fit_start=time.monotonic()
            model.fit(x,y)
            fit_seconds=time.monotonic()-fit_start
            score=evaluate_queries(data,validation_year,lambda nodes:val_features.score_model(model,nodes),None)
            value=score['active']['ndcg_20']
            trials.append({'setting':setting,'fit_seconds':fit_seconds,'validation':score})
            print(name,method,seed,setting,'val_ndcg20',value,'fit_seconds',round(fit_seconds,2),flush=True)
            if value is not None and value>best_score:
                best_model,best_score,best_setting=model,value,setting
        if best_model is None:
            raise ValueError('No valid validation queries')
        joblib.dump(best_model,out/'model.joblib')
        split_metrics={}
        for split in ['validation','test']:
            year=data.split_years(split)[0]
            features=val_features if split=='validation' else PairFeatures(data,year-1)
            split_metrics[split]=evaluate_queries(data,year,lambda nodes:features.score_model(best_model,nodes),out/(split+'_queries.npz'))
        result={'dataset':name,'method':method,'seed':seed,'level':level,'feature_names':FEATURE_NAMES,
                'label_codes':data.codes,'candidate_ids':data.candidate_ids.tolist(),
                'training_pair_count':len(y),'training_positive_count':int(y.sum()),'max_pairs':max_pairs,
                'negative_samples_per_query':10,'best_setting':best_setting,'selection_metric':'validation_ndcg_20',
                'trials':trials,'splits':split_metrics,'wall_seconds':time.monotonic()-start,
                'source_sha256':data.manifest['source_sha256'],'protocol_sha256':file_hash(ROOT/'configs/protocol.json'),
                'peak_process_rss_bytes':max(monitor.peak_rss,monitor.process.memory_info().rss)}
        atomic_json(out/'metrics.json',result)
    print('COMPLETE',name,method,seed,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--datasets',nargs='+',required=True)
    p.add_argument('--methods',nargs='+',choices=['SVM','RF','XGBoost'],default=['SVM','XGBoost','RF'])
    p.add_argument('--seeds',nargs='+',type=int,default=[42,2026,3407])
    p.add_argument('--level',type=int,default=3)
    p.add_argument('--max-pairs',type=int,default=300000)
    p.add_argument('--threads',type=int,default=4)
    args=p.parse_args()
    for name in args.datasets:
        for method in args.methods:
            for seed in args.seeds:
                run(name,method,seed,args.level,args.max_pairs,args.threads)
