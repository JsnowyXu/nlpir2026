#!/usr/bin/env python
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from patentflow.dataset import DatasetView
from patentflow.event_stream import EventStream
from patentflow.idgp_features import prepare

for country in ['IN','CN','US']:
    for section in ['B','F','C','All']:
        name=country+'_'+section
        prepare(EventStream(DatasetView(name)),name)
        print(name,'IDGP causal slow features complete',flush=True)
