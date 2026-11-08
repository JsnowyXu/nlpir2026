#!/usr/bin/env python
"""Training-vocabulary token IDs and causal last-patent histories for MENCP."""
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from gensim.models import Word2Vec
from patentflow.data import ROOT, atomic_json
from patentflow.dataset import DatasetView


def prepare(country, tokens, history):
    text = ROOT/'data/text'/country
    text_manifest = json.loads((text/'manifest.json').read_text())
    out = ROOT/'data/sequences'/country/('tokens%d_history%d' % (tokens, history))
    if (out/'manifest.json').exists():
        return
    out.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    model = Word2Vec.load(str(text/'word2vec.model'))
    words = model.wv.index2word
    lookup = {w: i+3 for i,w in enumerate(words)}
    embeddings = np.concatenate([np.zeros((3, 100), dtype=np.float32), model.wv.vectors])
    np.save(out/'initial_embeddings.npy', embeddings)
    rows = np.load(text/'patent_vectors.npy', mmap_mode='r').shape[0]
    ids = np.lib.format.open_memmap(out/'tokens.npy', mode='w+', dtype=np.int32, shape=(rows, tokens))
    ids[:] = 0
    with (text/'tokens.tsv').open() as f:
        for line in f:
            row, year, sentence = line.rstrip('\n').split('\t',2)
            encoded = [lookup.get(w, 1) for w in sentence.split()[:tokens]]
            ids[int(row), :len(encoded)] = encoded
    ids.flush()
    events = np.load(ROOT/'data/processed'/country/'events.npz')
    for section in ['All','B','F','C']:
        name = country.upper()+'_'+section
        data = DatasetView(name)
        mask = np.isin(events['dst'], data.spec['subclass_ids'])
        # Every patent occurs once per organization, irrespective of its IPC count.
        unique = np.unique(np.stack([events['src'][mask], events['patent_row'][mask],
                                    events['day'][mask], events['year'][mask]], axis=1), axis=0)
        order = np.lexsort((-unique[:,1], -unique[:,2], unique[:,0]))
        unique = unique[order]
        target = out/name; target.mkdir(exist_ok=True)
        for year in data.manifest['years']:
            if year >= data.spec['test_year']:
                continue
            past = unique[unique[:,3] <= year]
            result = np.full((len(data.first_year), history), -1, dtype=np.int64)
            if len(past):
                starts = np.maximum.accumulate(np.where(np.r_[True, np.diff(past[:,0]) != 0],
                                                       np.arange(len(past)), 0))
                rank = np.arange(len(past))-starts
                keep = rank < history
                result[past[keep,0], rank[keep]] = past[keep,1]
            np.save(target/('%d.npy' % year), result)
        print(name, 'historical_document_indices_complete', flush=True)
    atomic_json(out/'manifest.json', {'country':country, 'tokens_per_document':tokens,
        'last_patents':history, 'word2vec_sha256':text_manifest['model_sha256'],
        'protocol_sha256':data.manifest['protocol_sha256'], 'source_sha256':data.manifest['source_sha256'],
        'vocabulary_size':len(lookup)+3, 'reserved_ids':{'padding':0,'unknown_word':1,'empty_history':2},
        'seconds':time.monotonic()-start})


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--countries',nargs='+',required=True)
    p.add_argument('--tokens',type=int,default=64);p.add_argument('--history',type=int,default=4)
    a=p.parse_args()
    for country in a.countries:prepare(country,a.tokens,a.history)
