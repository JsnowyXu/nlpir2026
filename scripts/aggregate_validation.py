#!/usr/bin/env python
"""Live validation table, with pilots excluded and test values never consulted."""
from pathlib import Path
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import pandas as pd
from patentflow.data import ROOT,file_hash

rows=[]
protocol=file_hash(ROOT/'configs/protocol.json')
paths=list((ROOT/'results/baselines').rglob('metrics.json'))
paths+=list((ROOT/'results/main').rglob('validation.json'))
paths+=list((ROOT/'results/deep').rglob('validation.json'))
for path in paths:
    value=json.loads(path.read_text())
    if value.get('protocol_sha256')!=protocol:continue
    config=value.get('config',{})
    if config and config.get('tag')!='v2_default':continue
    metrics=value.get('splits',{}).get('validation',{}).get('active',{})
    if not metrics:continue
    rows.append(dict(dataset=value['dataset'],method=value['method'],seed=value['seed'],
        best_epoch=value.get('best_epoch'),parameter_count=value.get('parameter_count'),
        wall_seconds=value.get('wall_seconds'),source=str(path.relative_to(ROOT)),**metrics))
out=ROOT/'reports/validation';out.mkdir(exist_ok=True)
frame=pd.DataFrame(rows)
frame.to_csv(out/'per_seed.csv',index=False)
summary=frame.groupby(['dataset','method']).agg(n=('seed','nunique'),ndcg20_mean=('ndcg_20','mean'),
    ndcg20_std=('ndcg_20','std'),recall20_mean=('recall_20','mean'),mrr20_mean=('mrr_20','mean'),
    wall_seconds_mean=('wall_seconds','mean')).reset_index()
summary.to_csv(out/'summary.csv',index=False)
lines=['# 验证集阶段结果（实时、不含pilot）','',
       '仅用于选参和判断下一步实验；不是最终测试结果。n不足3的方法尚未完成。',
       'TOP/PersonalTOP是确定性结果，三个seed记录不是三次独立随机抽样。','',
       '| 数据集 | 方法 | seed数 | NDCG@20均值 | 标准差 |','|---|---|---:|---:|---:|']
for row in summary.itertuples():
    std='—' if pd.isna(row.ndcg20_std) else '%.4f'%row.ndcg20_std
    lines.append('| %s | %s | %d | %.4f | %s |'%(row.dataset,row.method,row.n,row.ndcg20_mean,std))
(out/'README.md').write_text('\n'.join(lines)+'\n')
print(summary[summary.dataset.str.endswith('_All')].to_string(index=False))
