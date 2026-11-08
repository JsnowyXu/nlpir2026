#!/usr/bin/env python
"""Train 100-D Word2Vec on training records, then freeze and encode all dates."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sqlite3
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from gensim.models import Word2Vec
from patentflow.data import ROOT,atomic_json,file_hash,load_taxonomy
from patentflow.entities import eligible
from patentflow.resources import ResourceMonitor
from patentflow.text import tokenize,tokenize_row


class TrainingSentences:
    def __init__(self,path,cutoff):
        self.path,self.cutoff=path,cutoff
    def __iter__(self):
        with self.path.open() as f:
            for line in f:
                row_id,year,text=line.rstrip('\n').split('\t',2)
                if int(year)<=self.cutoff and text:
                    yield text.split(' ')


def run(lang,workers,epochs):
    country=ROOT/'data/processed'/lang
    manifest=json.loads((country/'manifest.json').read_text())
    out=ROOT/'data/text'/lang
    out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists():
        return
    start=time.monotonic()
    with ResourceMonitor(out/'resources.jsonl') as monitor:
        tokenfile=out/'tokens.tsv'
        if not tokenfile.exists():
            conn=sqlite3.connect('file:'+str(ROOT/'data/raw_processed'/(lang+'.sqlite'))+'?mode=ro',uri=True)
            taxonomy=load_taxonomy(lang)
            def rows():
                query=conn.execute('SELECT source_row,year,title,abstract,orgs,subclasses FROM patents WHERE year>=? AND year<=? ORDER BY source_row',
                                   (manifest['observation_start_year'],manifest['test_year']))
                for row_id,year,title,abstract,orgs,codes in query:
                    if not any(eligible(n,manifest['entity_scope']) for n in json.loads(orgs)):
                        continue
                    if not any(c in taxonomy for c in json.loads(codes)):
                        continue
                    yield row_id,year,title,abstract,lang
            with ProcessPoolExecutor(max_workers=workers) as pool, tokenfile.with_suffix('.tmp').open('w') as f:
                # Executor.map eagerly queues iterables in Python 3.8; bound each batch.
                from itertools import islice
                iterator=iter(rows());count=0
                while True:
                    batch=list(islice(iterator,4000))
                    if not batch: break
                    for row_id,year,tokens in pool.map(tokenize_row,batch,chunksize=100):
                        f.write('%d\t%d\t%s\n'%(row_id,year,' '.join(tokens)))
                        count+=1
                    if count%20000==0: print(lang,'tokenized',count,flush=True)
            conn.close()
            tokenfile.with_suffix('.tmp').replace(tokenfile)
        sentences=TrainingSentences(tokenfile,manifest['train_end_year'])
        model_path=out/'word2vec.model'
        if model_path.exists():
            model=Word2Vec.load(str(model_path))
        else:
            print(lang,'training_word2vec',flush=True)
            # gensim 3.8 uses size/iter; this environment's version is recorded below.
            model=Word2Vec(sentences=sentences,size=100,window=5,min_count=5,workers=workers,
                           sg=1,negative=5,iter=epochs,seed=42,max_final_vocab=300000)
            model.save(str(model_path))
        counts={w:v.count for w,v in model.wv.vocab.items()}
        corpus_total=max(sum(counts.values()),1)
        def encode(tokens):
            valid=[w for w in tokens if w in model.wv.vocab]
            if not valid: return np.zeros(100,dtype=np.float32),0.
            weights=np.array([.001/(.001+counts[w]/corpus_total) for w in valid],dtype=np.float32)
            vector=(model.wv[valid]*weights[:,None]).sum(axis=0)/weights.sum()
            return vector.astype(np.float32),len(valid)/max(len(tokens),1)
        conn=sqlite3.connect('file:'+str(ROOT/'data/raw_processed'/(lang+'.sqlite'))+'?mode=ro',uri=True)
        max_id=conn.execute('SELECT MAX(source_row) FROM patents').fetchone()[0]
        conn.close()
        vectors=np.lib.format.open_memmap(out/'patent_vectors.npy',mode='w+',dtype=np.float32,shape=(max_id+1,100))
        vectors[:]=0.
        rows=[];coverage=Counter();tokens_total=Counter();documents=Counter()
        with tokenfile.open() as f:
            for line in f:
                rid,year,text=line.rstrip('\n').split('\t',2)
                rid,year=int(rid),int(year)
                tokens=text.split() if text else []
                vectors[rid],fraction=encode(tokens)
                rows.append(rid)
                split='train' if year<=manifest['train_end_year'] else 'validation' if year==manifest['validation_year'] else 'test' if year==manifest['test_year'] else 'context_only'
                documents[split]+=1;tokens_total[split]+=len(tokens);coverage[split]+=round(fraction*len(tokens))
        vectors.flush()
        np.save(out/'included_source_rows.npy',np.asarray(rows,dtype=np.int64))
        tree=json.loads((country/'observed_taxonomy.json').read_text())
        codes=sorted(tree,key=lambda c:(len(c),c))
        anchors=[];anchor_coverage={}
        raw={}
        for code in codes:
            raw[code],fraction=encode(tokenize(tree[code]['description'],lang))
            anchor_coverage[code]=fraction
        for code in codes:
            # Frozen source semantics are retained separately from dynamic memories.
            family=[code]
            if len(code)==4: family+=[code[:3],code[:1]]
            elif len(code)==3: family+=[code[:1]]
            vector=np.mean([raw[c] for c in family if c in raw],axis=0)
            anchors.append(vector/(np.linalg.norm(vector)+1e-8))
        np.savez_compressed(out/'taxonomy_anchors.npz',codes=np.asarray(codes),anchors=np.stack(anchors))
        import gensim
        result={'country':lang,'dimension':100,'algorithm':'skipgram_negative_sampling','seed':42,
            'workers':workers,'parallel_training_bitwise_deterministic':workers==1,'epochs':epochs,
            'window':5,'min_count':5,'max_tokens_per_record':512,'max_final_vocab':300000,
            'gensim_version':gensim.__version__,'train_end_year':manifest['train_end_year'],
            'training_includes_country_All_history_for_all_section_views':True,
            'taxonomy_text_added_to_word2vec_training':False,'vocabulary_size':len(model.wv.vocab),
            'document_count_by_split':dict(documents),'token_coverage':{k:coverage[k]/max(tokens_total[k],1) for k in tokens_total},
            'anchor_token_coverage':anchor_coverage,'model_sha256':file_hash(model_path),
            'source_sha256':manifest['source_sha256'],'protocol_sha256':manifest['protocol_sha256'],
            'elapsed_seconds':time.monotonic()-start,'peak_process_rss_bytes':monitor.peak_rss}
        atomic_json(out/'manifest.json',result)
        print('COMPLETE',lang,'vocabulary',len(model.wv.vocab),'elapsed_s',result['elapsed_seconds'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--languages',nargs='+',required=True)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--epochs',type=int,default=5)
    args=p.parse_args()
    for lang in args.languages:run(lang,args.workers,args.epochs)
