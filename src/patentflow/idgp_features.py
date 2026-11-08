"""Causal half-year interaction-type counts and 20-event gaps for IDGP."""
from datetime import date
import json
import numpy as np
from .data import ROOT, atomic_json


def prepare(stream, name):
    level=stream.data.level
    out=ROOT/'data/idgp'/(name if level==3 else name+'_l%d'%level)
    if (out/'manifest.json').exists():
        return out
    out.mkdir(parents=True,exist_ok=True)
    fields=stream.dst-stream.org_count
    nfields=stream.node_count-stream.org_count
    first_year=int(stream.year.min())
    half_counts=np.zeros(((int(stream.year.max())-first_year+1)*2,nfields,4),dtype=np.float32)
    seen=np.zeros((stream.org_count,nfields),dtype=bool)
    _, unique_pair=np.unique(np.c_[stream.src,stream.rows],axis=0,return_index=True)
    applicant_count=np.bincount(stream.rows[unique_pair])
    cooperative=applicant_count[stream.rows]>1
    last_days=np.full((nfields,20),-1,dtype=np.int64)
    ipc_list,counts_list,gaps_list,starts,offsets=[],[],[],[],[0]
    for left,right,year in stream.groups(int(stream.year.max())):
        day=int(stream.day[left])
        current=date.fromordinal(date(1970,1,1).toordinal()+stream.origin_day+day-1)
        half=(year-first_year)*2+int(current.month>=7)
        src,fid=stream.src[left:right],fields[left:right]
        # Read all repetition flags before this date writes any seen pair.
        types=seen[src,fid].astype(int)*2+cooperative[left:right].astype(int)
        np.add.at(half_counts[half],(fid,types),1.)
        seen[src,fid]=True
        ipc,event_count=np.unique(fid,return_counts=True)
        for node,count in zip(ipc,event_count):
            count=min(int(count),20)
            last_days[node]=np.roll(last_days[node],count)
            last_days[node,:count]=day
        gap=(day-last_days[ipc]).astype(np.float32)*86400.
        gap[last_days[ipc]<0]=1e15
        recent=half_counts[max(0,half-4):half+1,ipc,:].transpose(1,0,2)
        padded=np.zeros((len(ipc),5,4),dtype=np.float32)
        padded[:,-recent.shape[1]:]=recent
        ipc_list.append(ipc);counts_list.append(padded);gaps_list.append(gap)
        starts.append(left);offsets.append(offsets[-1]+len(ipc))
    np.savez_compressed(out/'features.npz',ipc=np.concatenate(ipc_list),counts=np.concatenate(counts_list),
                        gaps=np.concatenate(gaps_list),event_starts=starts,offsets=offsets)
    atomic_json(out/'manifest.json',{'dataset':name,'level':level,'source_sha256':stream.data.manifest['source_sha256'],
        'protocol_sha256':stream.data.manifest['protocol_sha256'],
        'four_types':['new_org_ipc_solo','new_org_ipc_collaborative','repeated_org_ipc_solo','repeated_org_ipc_collaborative'],
        'windows':'last five calendar half-years including observed current partial half-year',
        'density':'last 20 observed edges, gaps in seconds; missing positions use 1e15 and no clipping',
        'same_date':'flags read before group write; counter includes complete observed date'})
    return out
