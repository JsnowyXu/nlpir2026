#!/usr/bin/env python
"""Causal semantic/structural moments, in one chronological replay per country."""
import argparse
from datetime import date
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from patentflow.data import ROOT,atomic_json
from patentflow.moments import DecayedMoments


def build(lang,fast,slow,novelty,section='All'):
    source=ROOT/'data/processed'/lang
    manifest=json.loads((source/'manifest.json').read_text())
    textpath=ROOT/'data/text'/lang
    textmeta=json.loads((textpath/'manifest.json').read_text())
    store_name=lang if section=='All' else lang+'_'+section
    output=ROOT/'data/memory'/store_name/('f%d_s%d_n%g'%(fast,slow,novelty))
    if (output/'manifest.json').exists(): return
    output.mkdir(parents=True,exist_ok=True)
    events=np.load(source/'events.npz')
    src,dst,days,years,pid,baseweight=[events[k] for k in ['src','dst','day','year','patent_row','weight']]
    if section!='All':
        view=json.loads((ROOT/'data/processed/views'/(lang.upper()+'_'+section+'.json')).read_text())
        mask=np.isin(dst,view['subclass_ids'])
        src,dst,days,years,pid,baseweight=[v[mask] for v in [src,dst,days,years,pid,baseweight]]
    patent_vectors=np.load(textpath/'patent_vectors.npy',mmap_mode='r')
    taxonomy=np.load(textpath/'taxonomy_anchors.npz')
    anchor_by_code=dict(zip(taxonomy['codes'],taxonomy['anchors']))
    codes=json.loads((source/'subclasses.json').read_text())
    anchors=np.stack([anchor_by_code[c] for c in codes])
    # Independent, fixed projection of IPC identity keeps the structural branch
    # separable from global taxonomy semantics in ablations.
    structural_codes=np.random.RandomState(42).normal(size=anchors.shape).astype(np.float32)
    structural_codes/=np.linalg.norm(structural_codes,axis=1,keepdims=True)+1e-8
    states={}
    for name,n_nodes in [('org',manifest['n_organizations']),('ipc',manifest['n_subclasses'])]:
        for clock,tau in [('fast',fast),('slow',slow)]:
            states[name+'_'+clock]=DecayedMoments(n_nodes,200,tau)
    start=time.monotonic()
    cuts=np.r_[0,np.flatnonzero(np.diff(days))+1,len(days)]
    current_year=int(years[0])
    def save(year):
        cutoff=(date(year,12,31)-date(1970,1,1)).days
        result={}
        for key,state in states.items():
            result[key],result[key+'_mass'],result[key+'_age']=state.snapshot(cutoff)
        np.savez_compressed(output/('%d.npz'%year),**result)
        print(lang,'memory_snapshot',year,'elapsed_s',round(time.monotonic()-start,1),flush=True)
    for left,right in zip(cuts[:-1],cuts[1:]):
        day,year=int(days[left]),int(years[left])
        if year!=current_year:
            save(current_year);current_year=year
        org,ipc=src[left:right],dst[left:right]
        texts=patent_vectors[pid[left:right]]
        # All organization contexts are read before any state at this date is written.
        prior=states['org_slow'].mean(org)
        behavior=prior[:,100:]
        similarity=(behavior*structural_codes[ipc]).sum(axis=1)/(np.linalg.norm(behavior,axis=1)+1e-8)
        surprise=1.-np.clip(similarity,-1,1)
        weight=baseweight[left:right]*(1.+novelty*surprise)
        org_message=np.concatenate([texts,structural_codes[ipc]],axis=1)
        ipc_message=np.concatenate([texts,prior[:,100:]],axis=1)
        for clock in ['fast','slow']:
            states['org_'+clock].update(org,org_message,weight,day)
            states['ipc_'+clock].update(ipc,ipc_message,weight,day)
    save(current_year)
    atomic_json(output/'manifest.json',{'country':lang,'fast_half_life_days':fast,'slow_half_life_days':slow,
        'novelty_weight':novelty,'source_sha256':manifest['source_sha256'],'word2vec_sha256':textmeta['model_sha256'],
        'protocol_sha256':manifest['protocol_sha256'],'elapsed_seconds':time.monotonic()-start,
        'history_scope':'selected_section_only' if section!='All' else 'country_All', 'section':section,
        'structural_projection':{'seed':42,'dimension':100,'independent_of_taxonomy_semantics':True},
        'message_channels':{'org':['patent_text','ipc_identity_projection'],'ipc':['patent_text','pre_day_org_structural_memory']}})


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--languages',nargs='+',required=True)
    p.add_argument('--fast',type=int,default=180)
    p.add_argument('--slow',type=int,default=1095)
    p.add_argument('--novelty',type=float,default=1.)
    p.add_argument('--section',default='All')
    a=p.parse_args()
    for lang in a.languages:build(lang,a.fast,a.slow,a.novelty,a.section)
