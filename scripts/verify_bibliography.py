#!/usr/bin/env python
"""Cache primary registry/publisher metadata. Unmatched records remain pending."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
from difflib import SequenceMatcher
import html
import json
from pathlib import Path
import re
import sys
import urllib.parse
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import requests
from patentflow.data import ROOT,atomic_json

RECORDS=[
 ('edgpat','Event-Based Dynamic Graph Representation Learning for Patent Application Trend Prediction',None),
 ('mencp','Memory-enhanced hierarchical and temporal semantic learning for multi-label patent classification','10.1016/j.eswa.2025.127556'),
 ('repeatmixer','Repeat-Aware Neighbor Sampling for Dynamic Graph Learning','10.1145/3637528.3672001'),
 ('jodie','Predicting Dynamic Embedding Trajectory in Temporal Interaction Networks',None),
 ('dysat','DySAT: Deep Neural Representation Learning on Dynamic Graphs via Self-Attention Networks',None),
 ('patentbert','PatentBERT: Patent classification with fine-tuning a pre-trained BERT model',None),
 ('deeppatent','DeepPatent: patent classification with convolutional neural networks and word embedding',None),
 ('cascade','Continuous-Time Graph Learning for Cascade Popularity Prediction',None),
 ('hiagm','HiAGM: Hierarchy-Aware Global Model for Hierarchical Text Classification',None),
 ('himatch','HiMatch: Hierarchy-aware Label Semantics Matching Network for Hierarchical Text Classification',None),
 ('xgboost','XGBoost: A Scalable Tree Boosting System','10.1145/2939672.2939785'),
 ('rf','Random Forests','10.1023/A:1010933404324'),
 ('svm','Support-vector networks','10.1007/BF00994018'),
]
ARXIV=[
 ('tgn','2006.10637','Temporal Graph Networks for Deep Learning on Dynamic Graphs'),
 ('tgat','2002.07962','Inductive Representation Learning on Temporal Graphs'),
 ('dygformer','2303.13047','Towards Better Dynamic Graph Learning: New Architecture and Unified Library'),
 ('tami','2510.23577','TAMI: Taming Heterogeneity in Temporal Interactions for Temporal Graph Link Prediction'),
 ('tgb2','2406.09639','TGB 2.0: A Benchmark for Learning on Temporal Knowledge Graphs and Heterogeneous Graphs'),
 ('word2vec','1301.3781','Efficient Estimation of Word Representations in Vector Space'),
 ('skipgram','1310.4546','Distributed Representations of Words and Phrases and their Compositionality'),
 ('attention','1706.03762','Attention Is All You Need'),
 ('gat','1710.10903','Graph Attention Networks'),
 ('adam','1412.6980','Adam: A Method for Stochastic Optimization'),
 ('adamw','1711.05101','Decoupled Weight Decay Regularization'),
 ('graphmixer','2302.11636','Do We Really Need Complicated Model Architectures For Temporal Networks?'),
]
out=ROOT/'reports/bibliography';out.mkdir(exist_ok=True)
def normalize(value):return re.sub('[^a-z0-9]','',html.unescape(value).lower())
def fetch(record):
    key,title,doi=record
    path=out/(key+'.json')
    if path.exists():return json.loads(path.read_text())
    result={'key':key,'requested_title':title,'retrieved_utc':datetime.now(timezone.utc).isoformat(),
            'google_scholar_status':'not_yet_verified',
            'google_scholar_search':'https://scholar.google.com/scholar?q='+urllib.parse.quote('"'+title+'"')}
    try:
        if doi:
            url='https://api.crossref.org/works/'+urllib.parse.quote(doi,safe='/')
            response=requests.get(url,timeout=35);response.raise_for_status();items=[response.json()['message']]
        else:
            response=requests.get('https://api.crossref.org/works',params={'query.title':title,'rows':5},timeout=35)
            response.raise_for_status();items=response.json()['message']['items']
        ranked=sorted(items,key=lambda item:SequenceMatcher(None,normalize(title),normalize(' '.join(item.get('title',[])))).ratio(),reverse=True)
        item=ranked[0]
        similarity=SequenceMatcher(None,normalize(title),normalize(' '.join(item.get('title',[])))).ratio()
        result.update(status='primary_metadata_matched' if similarity>=.96 else 'manual_review_required',
            title_similarity=similarity,metadata_url='https://api.crossref.org/works/'+item['DOI'],
            metadata={k:item[k] for k in ['DOI','title','author','container-title','type','published','issued','event','volume','issue','page','URL'] if k in item})
    except Exception as e:
        result.update(status='fetch_failed',error=str(e))
    atomic_json(path,result);return result
def fetch_arxiv(record):
    key,identifier,title=record;path=out/(key+'.json')
    if path.exists():return json.loads(path.read_text())
    result={'key':key,'requested_title':title,'retrieved_utc':datetime.now(timezone.utc).isoformat(),
        'metadata_url':'https://arxiv.org/abs/'+identifier,'google_scholar_status':'not_yet_verified',
        'google_scholar_search':'https://scholar.google.com/scholar?q='+urllib.parse.quote('"'+title+'"')}
    try:
        response=requests.get(result['metadata_url'],timeout=35);response.raise_for_status()
        text=response.text
        def meta(name):
            return [html.unescape(x) for x in re.findall(r'<meta\s+name="'+re.escape(name)+r'"\s+content="([^"]*)"',text)]
        actual=meta('citation_title')[0];authors=meta('citation_author');when=meta('citation_date')[0]
        similarity=SequenceMatcher(None,normalize(title),normalize(actual)).ratio()
        result.update(status='primary_metadata_matched' if similarity>=.96 else 'manual_review_required',
            title_similarity=similarity,metadata={'title':[actual],'author_names':authors,'type':'arxiv-preprint',
                'arxiv':identifier,'issued_date':when,'DOI':'10.48550/arXiv.'+identifier,'doi_kind':'arxiv_repository',
                'doi_present_on_primary_page':('10.48550/arXiv.'+identifier).lower() in text.lower(),
                'URL':result['metadata_url']})
    except Exception as e:result.update(status='fetch_failed',error=str(e))
    atomic_json(path,result);return result

if __name__=='__main__':
    with ThreadPoolExecutor(max_workers=3) as pool:
        values=list(pool.map(fetch,RECORDS))+list(pool.map(fetch_arxiv,ARXIV))
    for result in values:
        m=result.get('metadata',{})
        print(result['key'],result['status'],m.get('DOI'),m.get('title'),m.get('container-title'),flush=True)
    atomic_json(out/'verification_index.json',{'records':values,
        'warning':'Primary metadata matching does not by itself verify Google Scholar indexing; preprint DOI is not a proceedings DOI.'})
