#!/usr/bin/env python
"""MENCP-Forecast: original text/hierarchy operators on last historical patents."""
import argparse
import json
from pathlib import Path
import random
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import torch
from torch.nn import functional as F
from patentflow.data import ROOT, atomic_json
from patentflow.dataset import DatasetView
from patentflow.evaluate import evaluate_queries
from patentflow.resources import ResourceMonitor
from patentflow.mencp_baseline import HistoricalTokens, MENCPForecast


def main(args):
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    if args.device.startswith('cuda'):
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.set_per_process_memory_fraction(.3,torch.device(args.device))
    data=DatasetView(args.dataset,args.level)
    history=HistoricalTokens(data,args.tokens,args.history)
    model=MENCPForecast(data,np.load(history.path/'initial_embeddings.npy'),args.device,args.filters,args.dropout)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.0001)
    out=ROOT/'results/deep'/args.dataset/'MENCP'/args.tag/('s%d'%args.seed)
    out.mkdir(parents=True,exist_ok=True)
    cfg=vars(args).copy()
    if (out/'config.json').exists():
        old=json.loads((out/'config.json').read_text())
        for key in cfg:
            if key not in ['device','threads','resume','evaluate_test'] and cfg[key]!=old.get(key):
                raise ValueError('Different configuration: '+key)
    atomic_json(out/'config.json',cfg)
    atomic_json(out/'original_source_hashes.json',model.provenance())
    atomic_json(out/'codes.json',{'codes':data.codes,'candidate_ids':data.candidate_ids.tolist()})
    years=data.split_years('train')
    queries={y:data.queries(y)[0] for y in years}
    labels={y:data.counts(y)[:,data.candidate_ids].sign() for y in years}
    positive=sum(np.asarray(labels[y][queries[y]].sum(axis=0)).ravel() for y in years)
    nqueries=sum(len(q) for q in queries.values())
    pos_weight=torch.as_tensor(np.clip((nqueries-positive)/(positive+1),1,30),dtype=torch.float32,device=args.device)
    checkpoint=out/'best.pt'
    best,best_epoch,first,stale=-float('inf'),-1,0,0
    if checkpoint.exists():
        if not args.resume:raise FileExistsError('Use --resume or a new tag')
        saved=torch.load(checkpoint,map_location=args.device)
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
        best,best_epoch=saved['validation_ndcg20'],saved['epoch']
        first=args.epochs if (out/'validation.json').exists() else best_epoch+1
    def evaluate(year,path=None):
        model.eval()
        def score(nodes):
            with torch.no_grad():
                ids=torch.as_tensor(history.batch(nodes,year-1),device=args.device)
                return model(ids).cpu().numpy()
        return evaluate_queries(data,year,score,path,batch_size=args.batch_size)
    start=time.monotonic()
    with ResourceMonitor(out/'resources.jsonl') as monitor:
        for epoch in range(first,args.epochs):
            model.train();rng=np.random.RandomState(args.seed+epoch)
            total,count=0.,0;begin=time.monotonic()
            for year in years:
                order=rng.permutation(queries[year])[:args.max_queries_per_year]
                for left in range(0,len(order),args.batch_size):
                    orgs=order[left:left+args.batch_size]
                    ids=torch.as_tensor(history.batch(orgs,year-1),device=args.device)
                    target=torch.as_tensor(labels[year][orgs].toarray(),device=args.device)
                    optimizer.zero_grad(set_to_none=True)
                    logits=model(ids)
                    loss=F.binary_cross_entropy_with_logits(logits,target,pos_weight=pos_weight)
                    if not torch.isfinite(loss):raise FloatingPointError('Non-finite MENCP loss')
                    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
                    optimizer.step()
                    total+=float(loss.detach())*len(orgs);count+=len(orgs)
            metrics=evaluate(data.spec['validation_year'])
            value=metrics['active']['ndcg_20']
            record={'epoch':epoch,'loss':total/max(count,1),'queries':count,
                    'seconds':time.monotonic()-begin,'validation':metrics}
            with (out/'epochs.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            print(args.dataset,'MENCP',args.seed,'epoch',epoch,'val_ndcg20',value,'seconds',record['seconds'],flush=True)
            if value is not None and value>best:
                best,best_epoch,stale=value,epoch,0
                temp=out/'best.tmp.pt'
                torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch,
                    'validation_ndcg20':value,'config':cfg,'source_sha256':data.manifest['source_sha256'],
                    'protocol_sha256':data.manifest['protocol_sha256']},temp)
                temp.replace(checkpoint)
            else:stale+=1
            if stale>=args.patience:break
        model.load_state_dict(torch.load(checkpoint,map_location=args.device)['model'])
        splits={}
        for split in (['validation','test'] if args.evaluate_test else ['validation']):
            splits[split]=evaluate(data.split_years(split)[0],out/(split+'_queries.npz'))
        atomic_json(out/('metrics.json' if args.evaluate_test else 'validation.json'),{
            'dataset':args.dataset,'method':'MENCP','adaptation':'historical-patent text forecasting',
            'seed':args.seed,'config':cfg,'best_epoch':best_epoch,'best_validation_ndcg20':best,
            'parameter_count':sum(p.numel() for p in model.parameters()),'splits':splits,
            'wall_seconds':time.monotonic()-start,'peak_process_rss_bytes':monitor.peak_rss,
            'gpu_peak_allocated_bytes':torch.cuda.max_memory_allocated(args.device) if args.device.startswith('cuda') else 0,
            'source_sha256':data.manifest['source_sha256'],'protocol_sha256':data.manifest['protocol_sha256'],
            'test_evaluated':args.evaluate_test})


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',required=True);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--tag',default='v2_default');p.add_argument('--device',default='cuda:0')
    p.add_argument('--level',type=int,default=3);p.add_argument('--threads',type=int,default=4)
    p.add_argument('--tokens',type=int,default=64);p.add_argument('--history',type=int,default=4)
    p.add_argument('--filters',type=int,default=32);p.add_argument('--dropout',type=float,default=.1)
    p.add_argument('--lr',type=float,default=.001);p.add_argument('--epochs',type=int,default=20)
    p.add_argument('--patience',type=int,default=4);p.add_argument('--batch-size',type=int,default=64)
    p.add_argument('--max-queries-per-year',type=int,default=20000)
    p.add_argument('--resume',action='store_true');p.add_argument('--evaluate-test',action='store_true')
    main(p.parse_args())
