#!/usr/bin/env python
"""Rebuild an auditable summary directly from result JSON files."""
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import psutil

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports' / ('experiment_summary_' + datetime.now().strftime('%Y%m%d'))
OUT.mkdir(parents=True, exist_ok=True)
protocol = hashlib.sha256((ROOT / 'configs/protocol.json').read_bytes()).hexdigest()
methods = ['PatentFlow','TOP','PersonalTOP','SVM','XGBoost','RF','EDGPAT','MENCP','RepeatMixer','TGAT','DyGFormer','TAMI','IDGP','TGN','TGN_edgetype']
datasets = [c+'_'+s for c in ['CN','US','IN'] for s in ['All','B','F','C']]
records, excluded, conflicts = {}, [], []
# validation.json is the canonical validation artifact; metrics.json supplies test
# and validation where a separate validation artifact is unavailable.
paths = sorted((ROOT/'results').rglob('metrics.json')) + sorted((ROOT/'results').rglob('validation.json'))
sources = {}
for path in paths:
    v = json.loads(path.read_text())
    rel = str(path.relative_to(ROOT))
    tag = v.get('config',{}).get('tag','v2_default' if 'baselines' in path.parts else 'unknown')
    if v.get('protocol_sha256') != protocol or not (tag == 'v2_default' or tag.startswith(('ablation_','sensitivity_','memory_','level1_','level2_'))):
        excluded.append({'source':rel,'tag':tag,'reason':'nonformal tag or protocol mismatch'})
        continue
    assert v['dataset'] in datasets and v['method'] in methods
    sources[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    for split, data in v.get('splits',{}).items():
        for population in ['active','all_zero_filled']:
            for name, value in data.get(population,{}).items():
                match = re.fullmatch(r'(.+)_(10|20|30|40)',name)
                if not match:
                    continue
                key = (v['dataset'],v['method'],tag,int(v['seed']),split,population,match[1],int(match[2]))
                if key in records and abs(records[key]['value']-value)>1e-10:
                    conflicts.append({'key':list(key),'old':records[key]['value'],'new':value,'old_source':records[key]['source'],'new_source':rel})
                records[key] = dict(zip(['dataset','method','tag','seed','split','population','metric','K'],key),value=value,source=rel)
frame = pd.DataFrame(records.values())
assert not frame.empty and frame.value.notna().all()
keys = ['dataset','method','tag','split','population','metric','K']
agg = frame.groupby(keys).value.agg(n_seeds='count',mean='mean',std='std',variance='var').reset_index()
main = frame[frame.tag.eq('v2_default')]
components = frame[~frame.tag.eq('v2_default')]
coverage = []
for method in methods:
    for ds in datasets:
        for split in ['validation','test']:
            seeds = sorted(main[(main.method==method)&(main.dataset==ds)&(main.split==split)].seed.unique().tolist())
            coverage.append(dict(method=method,dataset=ds,split=split,n_seeds=len(seeds),seeds=','.join(map(str,seeds)),missing_seeds=','.join(str(s) for s in [42,2026,3407] if s not in seeds)))
coverage = pd.DataFrame(coverage)
progress = coverage.groupby(['method','split']).n_seeds.sum().unstack().reindex(methods).reset_index()
queues=[]
for path in sorted((ROOT/'state').glob('*.json')):
    v=json.loads(path.read_text())
    if not isinstance(v,dict) or not isinstance(v.get('jobs'),dict): continue
    for job,item in v['jobs'].items():
        live=False
        if item.get('status')=='running':
            try:
                p=psutil.Process(item['pid'])
                live=p.is_running() and p.status()!=psutil.STATUS_ZOMBIE and str(ROOT) in p.cwd()
            except (psutil.Error,OSError): pass
        queues.append(dict(queue=path.stem,job=job,status=item.get('status'),pid=item.get('pid'),live=live,stale_running=item.get('status')=='running' and not live))
queues=pd.DataFrame(queues)
qsummary=queues.groupby(['queue','status']).size().unstack(fill_value=0).reset_index()
seed42=main[(main.seed==42)&(main.split=='test')]
a20=agg[(agg.tag=='v2_default')&(agg.population=='active')&(agg.K==20)&(agg.metric=='ndcg')&(agg.split=='validation')]
t20=seed42[(seed42.population=='active')&(seed42.K==20)&(seed42.metric=='ndcg')]
comparison=[]
for ds in datasets:
    q=t20[t20.dataset==ds].sort_values('value',ascending=False)
    pf=q[q.method=='PatentFlow'].iloc[0]
    comparison.append(dict(dataset=ds,best_method=q.iloc[0]['method'],best_ndcg20=q.iloc[0].value,PatentFlow=pf.value,rank=int(q.value.rank(method='min',ascending=False).loc[pf.name]),methods=len(q)))
comparison=pd.DataFrame(comparison)
tables={'method_progress':progress,'coverage':coverage,'official_seed_results':main,'official_aggregates':agg[agg.tag=='v2_default'],'component_seed_results':components,'component_aggregates':agg[agg.tag!='v2_default'],'test_seed42_all_metrics':seed42,'test_seed42_comparison':comparison,'queue_summary':qsummary,'queue_jobs':queues}
for metric in ['recall','ndcg','phr','mrr','ap','precision']:
    for split, data in [('test_seed42',seed42),('validation',agg[agg.tag=='v2_default'])]:
        q=data[(data.population=='active')&(data.K==20)&(data.metric==metric)]
        if split=='validation': q=q[q.split=='validation']
        if len(q): tables[split+'_'+metric+'20_matrix']=q.pivot(index='method',columns='dataset',values='value' if split=='test_seed42' else 'mean').reindex(methods).reset_index()
for name,data in tables.items(): data.to_csv(OUT/(name+'.csv'),index=False)
with pd.ExcelWriter(OUT/'实验结果完整汇总.xlsx',engine='openpyxl') as writer:
    for name,data in tables.items(): data.to_excel(writer,sheet_name=name[:31],index=False)
    for sheet in writer.book.worksheets:
        sheet.freeze_panes='B2'; sheet.auto_filter.ref=sheet.dimensions
        for col in sheet.columns:
            sheet.column_dimensions[col[0].column_letter].width=min(48,max(14,len(str(col[0].value))+3))

def md(data):
    headers=list(data.columns)
    rows=['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']
    for row in data.itertuples(index=False,name=None):
        rows.append('| '+' | '.join(f'{v:.6f}' if isinstance(v,float) else str(v) for v in row)+' |')
    return '\n'.join(rows)
now=datetime.now(timezone.utc).isoformat()
missing=coverage[(coverage.split=='validation')&(coverage.n_seeds<3)]
lines=['# NLPIR2026 当前实验结果汇总','',f'核查时间（UTC）：{now}；北京时间为 UTC+8。', '',
'直接读取 results 中当前协议的原始 JSON，验证和测试分开，pilot、迁移验证、计时试验不计入正式结果。默认主结果和所有消融、超参数、memory、层级实验均纳入；保留 active / all_zero_filled 两类总体、全部已记录指标和 K=10/20/30/40。原始文件 SHA256 见 audit.json。', '',
f"主实验验证 {int(progress.validation.sum())}/540；三种子测试 {int(progress.test.sum())}/540；统一 seed=42 测试 {len(t20)}/180。", '',
'单种子测试已齐备，但不能当作三种子均值或显著性证据。传统方法各有三个种子，神经方法各只有 seed=42 测试；TOP/PersonalTOP 的重复输出不代表独立随机试验。均值、样本标准差和样本方差分别列出，n=1 的标准差/方差留空。','',md(progress),'',
'## 未完成验证','',md(missing),'',
'验证优先使用 validation.json；若单独文件缺失但 metrics.json 内含验证结果，则计为已有验证数值。队列完成标志可能与结果数不同，不能仅凭队列状态否定已落盘结果。','',
'## 运行状态','',f"队列任务中实际存活 {int(queues.live.sum())} 条；陈旧 running 记录 {int(queues.stale_running.sum())} 条。检查时未发现项目训练进程。未启动或重启实验。",'',md(queues[queues.stale_running][['queue','job','pid','status']]),'',md(qsummary),'',
'## 最终测试：统一 seed=42，active NDCG@20','',md(comparison),'',
f"PatentFlow 在 {int((comparison['rank']==1).sum())}/12 个视图排名第一。其余视图及差距见上表；不能宣称全数据集或全指标领先。",'',
'## 验证：三个 All 视图 NDCG@20','',md(a20[a20.dataset.str.endswith('_All')][['dataset','method','n_seeds','mean','std']]),'',
'## 组件实验（验证集，NDCG@20）','']
ca=agg[(agg.tag!='v2_default')&(agg.population=='active')&(agg.K==20)&(agg.metric=='ndcg')&(agg.split=='validation')]
for prefix,title in [('ablation_','消融'),('sensitivity_','超参数敏感性'),('memory_','记忆变体'),('level','层级')]:
    q=ca[ca.tag.str.startswith(prefix)]
    lines += [f'### {title}', '',f'已完成 {int(q.n_seeds.sum())} 个种子实验；{len(q)} 个数据集×配置组合。','',md(q[q.dataset.str.endswith('_All')][['dataset','tag','n_seeds','mean','std']]),'']
lines += ['消融需与 ablation_full 比较。移除 anchor/alignment 在三个 All 上并未降低均值，IN_All 移除 structure 反而提升；不支持所有模块均有效。三个 All 的记忆变体最佳均为 memory_single_1095。以上为验证期观察，不能用测试表现重新选参。','',
'## 案例与历史成果','',
'年度轨迹、组织—专利语义、IPC 层级语义的证据见 reports/memory_evolution/；数据和图统计见 reports/model_graph_details_20260920/，输入审计见 reports/raw_audit/。案例为回顾性描述，不计为测试预测结果。历史稿件/PPT 快照保留原样。','',
'## 交付文件','',
'- 实验结果完整汇总.xlsx：方法进度、逐数据集覆盖、全部正式与组件逐种子及聚合指标、seed=42 比较、队列状态和核心指标矩阵。',
'- 同名 CSV：完整精度数据，source 列可回溯原始指标文件。',
'- audit.json：输入指纹、排除清单、重复指标差异和行数。','']
lines += ['## 同一验证结果的文件差异', '', f'发现 {len(conflicts)} 个指标在 validation.json 与 metrics.json 的验证副本中不一致（阈值 1e-10）。本汇总固定优先采用 validation.json，差异及两个来源完整记录于 audit.json；不影响测试结果的来源选择。', '']
(OUT/'当前实验汇总.md').write_text('\n'.join(lines),encoding='utf-8')
audit={'generated_utc':now,'protocol_sha256':protocol,'source_sha256':sources,'excluded':excluded,'duplicate_value_conflicts':conflicts,'rows':{k:len(v) for k,v in tables.items()}}
(OUT/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2))
print(OUT); print(progress.to_string(index=False)); print(missing.to_string(index=False)); print(comparison.to_string(index=False)); print('conflicts',len(conflicts))
