#!/usr/bin/env python
"""Explicit validation-only ablation and sensitivity plans, ordered by work estimate."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from patentflow.data import ROOT,atomic_json
from patentflow.dataset import DatasetView

seeds=[42,2026,3407]
names=[country+'_'+section for country in ['IN','CN','US'] for section in ['B','F','C','All']]
def work(name):
    data=DatasetView(name)
    queries=sum(min(20000,int((data.first_year<y).sum())) for y in data.split_years('train'))
    queries+=int((data.first_year<data.spec['validation_year']).sum())
    return queries*len(data.candidate_ids)
names.sort(key=work)
variants=[('ablation_full',[])]+[('ablation_no_'+key,['--disable',key]) for key in
    ['text','structure','hierarchy','cooperation','community','repeat','anchor','alignment']]
variants += [('ablation_history_only',['--disable','text','structure','hierarchy','cooperation','community','anchor','alignment']),
             ('ablation_no_text_hierarchy',['--disable','text','hierarchy','anchor','alignment'])]
jobs=[]
for name in names:
    country,section=name.split('_')
    slug=country.lower()+('' if section=='All' else '_'+section)
    for tag,extra in variants:
        for seed in seeds:
            jobs.append(dict(id=name+'_'+tag+'_s'+str(seed),device='cpu',
                argv=['main_experiment/train.py','--dataset',name,'--seed',str(seed),'--tag',tag,
                      '--device','{device}','--resume']+extra,
                output=f'results/main/{name}/{tag}/s{seed}/validation.json',
                requires=[f'data/memory/{slug}/f180_s1095_n1/manifest.json'],estimated_pair_work=work(name)))
atomic_json(ROOT/'configs/ablation_campaign.json',dict(name='ablation_campaign_v2',jobs=jobs,
    selection='validation only; no test evaluated',platform='CPU, 4 threads per job, including full comparator'))

grid={'lr':[.0003,.003],'dropout':[0.,.3],'dim':[64,128],'communities':[8,64],
      'temperature':[.2,1.],'rank-loss':[0.,.2,1.],'alignment':[0.,.03,.3],
      'weight-decay':[0.,.001],'max-queries-per-year':[5000,50000]}
jobs=[]
for name in ['IN_All','CN_All','US_All']:
    for key,values in grid.items():
        for value in values:
            tag='sensitivity_'+key.replace('-','_')+'_'+str(value).replace('.','p')
            for seed in seeds:
                jobs.append(dict(id=name+'_'+tag+'_s'+str(seed),device='cpu',
                    argv=['main_experiment/train.py','--dataset',name,'--seed',str(seed),'--tag',tag,
                          '--device','{device}','--resume','--'+key,str(value)],
                    output=f'results/main/{name}/{tag}/s{seed}/validation.json',
                    requires=[f'data/memory/{name.split("_")[0].lower()}/f180_s1095_n1/manifest.json'],
                    sensitivity_parameter=key,sensitivity_value=value))
atomic_json(ROOT/'configs/sensitivity_campaign.json',dict(name='sensitivity_campaign_v2',jobs=jobs,
    selection='validation only; one-factor-at-a-time; baseline from ablation_full',
    not_yet_in_this_plan=['memory half lives and novelty require additional cache builds']))
print('Ablation jobs:',len(names)*len(variants)*len(seeds),'Sensitivity jobs:',len(jobs))
