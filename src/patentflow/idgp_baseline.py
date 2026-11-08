"""IDGP-Forecast: original fast/slow, Fluxon GRU and trajectory Transformer."""
import sys
import numpy as np
import torch
from torch import nn
from .data import ROOT,file_hash
from .idgp_features import prepare

SOURCE=ROOT/'baseline/vendor/IDGP-main/IDGP-main'
sys.path.insert(0,str(SOURCE))
from models.PopGroup import PopGroup


class IDGPForecast(nn.Module):
    def __init__(self,stream,dim=100,time_dim=32,device='cpu',fluxion_size=16):
        super().__init__()
        self.stream,self.device_name=stream,device
        nfields=stream.node_count-stream.org_count
        self.original=PopGroup(np.zeros((stream.node_count+1,dim),dtype=np.float32),
            np.zeros((1,100),dtype=np.float32),time_dim,model_name='Fluxion',device=device,
            item_nums=nfields,user_nums=stream.org_count,fluxion_size=min(fluxion_size,nfields),
            router_type='cosine',fluxion_init_type='ball')
        # The source supports disabling clipping. Otherwise missing padding gaps
        # are clipped to 400 days and falsely contribute density to old/no history.
        self.original.density_encoder.clip_deltas_days=None
        self.org_head=nn.Linear(dim,dim)
        self.ipc_head=nn.Linear(dim*4,dim)
        self.bias=nn.Parameter(torch.zeros(nfields))
        self.register_buffer('ipc_cache',torch.zeros(nfields,dim*4))
        self.register_buffer('initial_flux',self.original.fluxon_bank.states.detach().clone())
        self.to(device)
        name=stream.data.spec['country'].upper()+'_'+stream.data.spec['section']
        features=np.load(prepare(stream,name)/'features.npz')
        self.slow_features={k:features[k] for k in features.files}
        self.group_of_start={int(v):i for i,v in enumerate(features['event_starts'])}

    @torch.no_grad()
    def reset(self):
        self.original.memory_bank.__init_memory_bank__()
        self.original.fluxon_bank.states.copy_(self.initial_flux)
        self.original.reset_item_history()
        self.ipc_cache.zero_()

    def propose(self,left,right):
        original,stream,device=self.original,self.stream,self.device_name
        src=torch.as_tensor(stream.src[left:right]+1,device=device)
        dst=torch.as_tensor(stream.dst[left:right]+1,device=device)
        source,dest=torch.cat([src,dst]),torch.cat([dst,src])
        day=float(stream.day[left]*86400)
        memory=original.memory_bank.node_memories.detach()
        last=original.memory_bank.node_last_updated_times
        time=original.time_encoder((day-last[source])[:,None]).squeeze(1)
        text=stream.messages(slice(left,right),device)
        messages=torch.cat([memory[source],memory[dest],time,torch.cat([text,text])],dim=1)
        unique,group=source.unique(return_inverse=True)
        aggregate=messages.new_zeros((len(unique),messages.shape[1])).index_add_(0,group,messages)
        mass=torch.bincount(group,minlength=len(unique)).clamp(min=1)
        fast=original.memory_updater.memory_updater(aggregate/mass[:,None],memory[unique])
        index=self.group_of_start[left]
        lo,hi=self.slow_features['offsets'][index:index+2]
        ipc_np=self.slow_features['ipc'][lo:hi]
        ipc=torch.as_tensor(ipc_np,device=device)
        counts=torch.as_tensor(self.slow_features['counts'][lo:hi],device=device)
        slow_count=original.lstm(original.typecounter(original.type_emb,counts))
        slow_time=original.density_encoder(self.slow_features['gaps'][lo:hi],(np.arange(len(ipc_np)),))
        slow=original.reduce_slow_memory_dim(torch.cat([slow_count,slow_time],dim=1))
        ipc_fast=fast[torch.searchsorted(unique,ipc+stream.org_count+1)]
        states=original.fluxon_bank.get_all_fluxon().detach().clone()
        route=original.router(torch.cat([ipc_fast,slow],dim=1),states)
        updated=original.fluxon_updater(ipc_fast,slow,route,states)
        flux=original.reduce_fluxion_dim(updated[route.squeeze(1)])
        padded=torch.cat([updated,updated.new_zeros(1,updated.shape[1])])
        # Original trajectory method writes buffers. Restore them until commit so
        # scoring any number of candidate pairs cannot append fictitious events.
        before_flux=original.item_flux_hist[ipc].clone()
        before_time=original.item_time_hist[ipc].clone()
        trajectory=original.aggregate_fluxion_trajectory(ipc,route.squeeze(1),padded,
                                                        torch.full((len(ipc),),day,device=device))
        after_flux=original.item_flux_hist[ipc].clone()
        after_time=original.item_time_hist[ipc].clone()
        original.item_flux_hist[ipc]=before_flux
        original.item_time_hist[ipc]=before_time
        representation=torch.cat([ipc_fast,slow,flux,trajectory],dim=1)
        return unique,fast,day,ipc,representation,updated,after_flux,after_time

    @torch.no_grad()
    def commit(self,pending,left,right):
        nodes,fast,day,ipc,representation,states,history,times=pending
        original=self.original
        original.memory_bank.node_memories[nodes]=fast.detach()
        original.memory_bank.node_last_updated_times[nodes]=day
        original.fluxon_bank.set_all_fluxon(None,states.detach())
        original.item_flux_hist[ipc]=history
        original.item_time_hist[ipc]=times
        self.ipc_cache[ipc]=representation.detach()

    def score(self,orgs,candidates,pending=None):
        device=self.device_name
        org=torch.as_tensor(orgs+1,device=device)
        fields=torch.as_tensor(candidates-self.stream.org_count,device=device)
        src=self.original.memory_bank.node_memories[org].detach()
        dst=self.ipc_cache[fields].detach()
        if pending is not None:
            nodes,fast,day,ipc,representation,_,_,_=pending
            for requested,available,updated,which in [(org,nodes,fast,'src'),(fields,ipc,representation,'dst')]:
                location=torch.searchsorted(available,requested)
                valid=(location<len(available))&(available[location.clamp(max=len(available)-1)]==requested)
                if which=='src':
                    src=src.clone();src[valid]=updated[location[valid]]
                else:
                    dst=dst.clone();dst[valid]=updated[location[valid]]
        return self.org_head(src) @ self.ipc_head(dst).t()/self.org_head.out_features**.5+self.bias[fields]

    @torch.no_grad()
    def replay(self,max_year):
        self.reset()
        for left,right,_ in self.stream.groups(max_year):
            self.commit(self.propose(left,right),left,right)

    def provenance(self):
        paths=[SOURCE/'models/PopGroup.py',SOURCE/'models/MultiScaleDensityEncoderv2.py']
        paths += [SOURCE/'models/Fluxons'/name for name in ['Fluxon.py','FluxonRouterCos.py','FluxionUpdaterCos.py']]
        return {str(p.relative_to(ROOT)):file_hash(p) for p in paths}
