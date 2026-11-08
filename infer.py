#!/usr/bin/env python3
"""Read-only, causal IPC Top-K inference for a known organization."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
os.environ['PATENTFLOW_WORKSPACE'] = str(ROOT)
sys.path.insert(0, str(ROOT / 'main_experiment' / 'src'))

import numpy as np
import torch
from patentflow_main.dataset import DatasetView
from patentflow_main.model import PatentFlow
from patentflow_main.neural_data import MemoryFeatures, anchors_for
from patentflow_main.io import file_hash


def predict(dataset, organization, top_k=20, device='cpu', checkpoint=None):
    """Return ranked IPC codes and raw model scores; input must be an observed organization."""
    path = Path(checkpoint) if checkpoint else ROOT / 'checkpoints' / dataset / 'best.pt'
    state = torch.load(path, map_location=device)
    cfg = state['config']
    if cfg['dataset'] != dataset:
        raise ValueError('Checkpoint dataset mismatch')
    data = DatasetView(dataset, cfg['level'])
    for key in ('source_sha256', 'protocol_sha256'):
        if state[key] != data.manifest[key]:
            raise ValueError('Checkpoint/prepared data mismatch: ' + key)
    names = json.loads((data.path / 'organizations.json').read_text())
    lookup = {name: i for i, name in enumerate(names)}
    normalized = ' '.join(organization.strip().casefold().split())
    if normalized not in lookup:
        raise ValueError('Unknown organization; use an exact name from data/processed/<country>/organizations.json')
    org_id = lookup[normalized]
    target_year = data.split_years('test')[0]
    cutoff = target_year - 1
    if data.first_year[org_id] >= target_year:
        raise ValueError('Organization was not observed before the prediction year')
    model = PatentFlow(anchors_for(data), dim=cfg['dim'], communities=cfg['communities'], dropout=cfg['dropout'], disabled=cfg['disable'], temperature=cfg['temperature']).to(device)
    model.load_state_dict(state['model'])
    model.eval()
    features = MemoryFeatures(data, cutoff, device, cfg['fast'], cfg['slow'], cfg['novelty'])
    features.refresh(model)
    with torch.no_grad():
        scores = model(*features.batch(np.array([org_id], dtype=np.int64)))[0].cpu().numpy()
    order = np.argsort(-scores, kind='stable')[:min(top_k, len(scores))]
    return {'dataset':dataset, 'organization':names[org_id], 'target_year':target_year, 'history_cutoff_year':cutoff, 'checkpoint_sha256':file_hash(path), 'predictions':[{'rank':rank+1,'ipc_code':data.codes[int(data.candidate_ids[i])], 'score':float(scores[i])} for rank,i in enumerate(order)]}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',required=True,choices=sorted(json.loads((ROOT/'checkpoints/selection.json').read_text())))
    parser.add_argument('--organization',required=True)
    parser.add_argument('--top-k',type=int,default=20)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--checkpoint',type=Path)
    args=parser.parse_args()
    if args.top_k < 1: parser.error('--top-k must be positive')
    print(json.dumps(predict(args.dataset,args.organization,args.top_k,args.device,args.checkpoint),ensure_ascii=False,indent=2))

if __name__=='__main__': main()
