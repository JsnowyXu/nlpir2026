#!/usr/bin/env python
"""Train original neighbor models; evaluate every candidate, without negatives sampling."""
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
from patentflow.sampling_baselines import SamplingForecast


def training_pairs(data,seed,budget):
    rng=np.random.RandomState(seed);result={}
    years=data.split_years('train');per_year=budget//len(years)
    for year in years:
        orgs,truth=data.queries(year);candidates=data.candidate_ids
        selected=np.sort(rng.choice(len(orgs),min(max(1,per_year//12),len(orgs)),replace=False))
        rows=[]
        for i in selected:
            positive=np.intersect1d(candidates,truth[i])
            pool=np.setdiff1d(candidates,positive,assume_unique=True)
            negative=rng.choice(pool,min(10,len(pool)),replace=False)
            rows.extend((orgs[i],c,1) for c in positive)
            rows.extend((orgs[i],c,0) for c in negative)
        array=np.asarray(rows,dtype=np.int64)
        if len(array)>per_year:array=array[rng.choice(len(array),per_year,replace=False)]
        result[year]=array
    return result


def main(args):
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    if args.device.startswith('cuda'):
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.set_per_process_memory_fraction(.3,torch.device(args.device))
    data=DatasetView(args.dataset,args.level)
    out=ROOT/'results/deep'/args.dataset/args.method/args.tag/('s%d'%args.seed)
    out.mkdir(parents=True,exist_ok=True)
    cfg=vars(args).copy()
    if (out/'config.json').exists():
        old=json.loads((out/'config.json').read_text())
        for key in cfg:
            if key not in ['device','threads','resume','evaluate_test'] and cfg[key]!=old.get(key):
                raise ValueError('Different configuration: '+key)
    atomic_json(out/'config.json',cfg)
    model=SamplingForecast(data,args.method,args.device,args.neighbors,args.layers,args.time_dim,
        args.sequence_length,args.channels,args.patch,not args.no_high_order,args.seed,args.dropout)
    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)
    atomic_json(out/'original_source_hashes.json',model.provenance())
    atomic_json(out/'codes.json',{'codes':data.codes,'candidate_ids':data.candidate_ids.tolist()})
    pairs=training_pairs(data,args.seed,args.max_pairs)
    np.savez_compressed(out/'training_pairs.npz',**{str(k):v for k,v in pairs.items()})
    labels=np.concatenate([v[:,2] for v in pairs.values()])
    weight=torch.tensor(float((labels==0).sum()/max((labels==1).sum(),1)),device=args.device)
    checkpoint=out/'best.pt';best,best_epoch,first,stale=-float('inf'),-1,0,0
    if checkpoint.exists():
        if not args.resume:raise FileExistsError('Use --resume or new tag')
        saved=torch.load(checkpoint,map_location=args.device)
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
        best,best_epoch=saved['validation_ndcg20'],saved['epoch']
        first=args.epochs if (out/'validation.json').exists() else best_epoch+1
    def evaluate(year,path=None):
        model.eval();begin=time.monotonic()
        if args.method=='TGAT':
            score=model.prepare_tgat_evaluation(year,batch_size=args.batch_size)
        else:
            def score(orgs):
                sources=np.repeat(orgs,len(data.candidate_ids))
                fields=np.tile(data.candidate_ids,len(orgs))
                values=[]
                with torch.no_grad():
                    for offset in range(0,len(sources),args.eval_pair_batch):
                        values.append(model.pairs(sources[offset:offset+args.eval_pair_batch],
                            fields[offset:offset+args.eval_pair_batch],year).cpu().numpy())
                return np.concatenate(values).reshape(len(orgs),-1)
        metrics=evaluate_queries(data,year,score,path,batch_size=64)
        metrics['full_catalog_evaluation_seconds']=time.monotonic()-begin
        return metrics
    start=time.monotonic()
    with ResourceMonitor(out/'resources.jsonl') as monitor:
        for epoch in range(first,args.epochs):
            model.train();rng=np.random.RandomState(args.seed+epoch)
            begin=time.monotonic();total,count=0.,0
            for year,array in pairs.items():
                order=rng.permutation(len(array))
                for offset in range(0,len(order),args.batch_size):
                    rows=array[order[offset:offset+args.batch_size]]
                    optimizer.zero_grad(set_to_none=True)
                    logits=model.pairs(rows[:,0],rows[:,1],year)
                    target=torch.as_tensor(rows[:,2],dtype=torch.float32,device=args.device)
                    loss=F.binary_cross_entropy_with_logits(logits,target,pos_weight=weight)
                    if not torch.isfinite(loss):raise FloatingPointError('Non-finite loss')
                    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
                    total+=float(loss.detach())*len(rows);count+=len(rows)
            train_seconds=time.monotonic()-begin
            metrics=evaluate(data.spec['validation_year'])
            value=metrics['active']['ndcg_20']
            record={'epoch':epoch,'loss':total/max(count,1),'training_pairs':count,'train_seconds':train_seconds,
                    'seconds':time.monotonic()-begin,'validation':metrics}
            with (out/'epochs.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            print(args.dataset,args.method,args.seed,'epoch',epoch,'val_ndcg20',value,
                  'train_seconds',train_seconds,'eval_seconds',metrics['full_catalog_evaluation_seconds'],flush=True)
            if value is not None and value>best:
                best,best_epoch,stale=value,epoch,0
                temp=out/'best.tmp.pt'
                torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch,
                    'validation_ndcg20':value,'config':cfg,'source_sha256':data.manifest['source_sha256'],
                    'protocol_sha256':data.manifest['protocol_sha256']},temp);temp.replace(checkpoint)
            else:stale+=1
            if stale>=args.patience:break
        model.load_state_dict(torch.load(checkpoint,map_location=args.device)['model'])
        splits={}
        for split in (['validation','test'] if args.evaluate_test else ['validation']):
            splits[split]=evaluate(data.split_years(split)[0],out/(split+'_queries.npz'))
        atomic_json(out/('metrics.json' if args.evaluate_test else 'validation.json'),{
            'dataset':args.dataset,'method':args.method,'seed':args.seed,'config':cfg,
            'best_epoch':best_epoch,'best_validation_ndcg20':best,'parameter_count':sum(p.numel() for p in model.parameters() if p.requires_grad),
            'splits':splits,'training_pair_count':len(labels),'training_positive_count':int(labels.sum()),
            'wall_seconds':time.monotonic()-start,'peak_process_rss_bytes':monitor.peak_rss,
            'gpu_peak_allocated_bytes':torch.cuda.max_memory_allocated(args.device) if args.device.startswith('cuda') else 0,
            'source_sha256':data.manifest['source_sha256'],'protocol_sha256':data.manifest['protocol_sha256'],
            'test_evaluated':args.evaluate_test})


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',required=True);p.add_argument('--method',required=True,choices=['TGAT','DyGFormer','RepeatMixer'])
    p.add_argument('--seed',type=int,default=42);p.add_argument('--tag',default='v2_default')
    p.add_argument('--device',default='cuda:0');p.add_argument('--threads',type=int,default=4)
    p.add_argument('--level',type=int,default=3);p.add_argument('--time-dim',type=int,default=32)
    p.add_argument('--neighbors',type=int,default=10);p.add_argument('--layers',type=int,default=2)
    p.add_argument('--sequence-length',type=int,default=32);p.add_argument('--channels',type=int,default=32)
    p.add_argument('--patch',type=int,default=2);p.add_argument('--no-high-order',action='store_true')
    p.add_argument('--dropout',type=float,default=.1);p.add_argument('--lr',type=float,default=.001)
    p.add_argument('--epochs',type=int,default=20);p.add_argument('--patience',type=int,default=4)
    p.add_argument('--batch-size',type=int,default=128);p.add_argument('--eval-pair-batch',type=int,default=1024)
    p.add_argument('--max-pairs',type=int,default=300000)
    p.add_argument('--resume',action='store_true');p.add_argument('--evaluate-test',action='store_true')
    main(p.parse_args())
