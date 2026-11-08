#!/usr/bin/env python
"""Export reproducible dataset/feasibility tables, figures and numerical commentary."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from patentflow.data import ROOT,atomic_json,file_hash
from patentflow.dataset import DatasetView

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,
                     'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
out=ROOT/'reports/data_analysis';out.mkdir(exist_ok=True)
figures=ROOT/'figures/data_analysis';figures.mkdir(parents=True,exist_ok=True)
def save(fig,name):
    fig.tight_layout()
    for ext in ['pdf','svg','png']:
        fig.savefig(figures/(name+'.'+ext),dpi=250,bbox_inches='tight')
    plt.close(fig)

rows=[];annual=[];semantics=[]
for country in ['CN','IN','US']:
    for section in ['All','B','F','C']:
        name=country+'_'+section
        stats=json.loads((ROOT/'reports/dataset_statistics'/(name+'.json')).read_text())
        probe=json.loads((ROOT/'reports/feasibility'/(name+'.json')).read_text())
        coop=probe['collaboration'];community=probe['behavior_communities']
        row=dict(dataset=name,country=country,section=section,subclasses=stats['subclasses'],
                 candidates=stats['train_candidates'])
        for split,values in stats['splits'].items():
            for key in ['events','active_queries','empty_queries','new_organization_active_queries_excluded',
                        'repeat_fraction','oov_target_count','historically_known_queries']:
                row[split+'_'+key]=values[key]
        row.update(peer_available_fraction=coop['coverage'],peer_lift=coop['macro_coverage_lift'],
                   peer_ci_low=coop['organization_bootstrap_95ci'][0],peer_ci_high=coop['organization_bootstrap_95ci'][1],
                   community_ndcg20=community['centroid_metrics']['ndcg_20'],
                   global_ndcg20=community['global_frequency_metrics']['ndcg_20'])
        row['community_ndcg20_gain']=row['community_ndcg20']-row['global_ndcg20']
        rows.append(row)
        data=DatasetView(name)
        for year in data.manifest['years']:
            counts=data.counts(year)
            annual.append(dict(dataset=name,year=year,events=int(counts.sum()),
                               organizations=int((np.asarray(counts.sum(axis=1)).ravel()>0).sum()),
                               active_ipc=int((np.asarray(counts.sum(axis=0)).ravel()>0).sum())))
    # These probes access training years only; no test-based module selection.
    data=DatasetView(country+'_All');target_year=data.spec['train_end_year'];cutoff=target_year-1
    snap=np.load(ROOT/'data/memory'/country.lower()/'f180_s1095_n1'/('%d.npz'%cutoff))
    known,truth=data.queries(target_year)
    active=np.array([len(t)>0 for t in truth]);orgs=known[active]
    fast,slow=snap['org_fast'][orgs,:100],snap['org_slow'][orgs,:100]
    norm=np.linalg.norm(fast,axis=1)*np.linalg.norm(slow,axis=1)
    drift=1-(fast*slow).sum(axis=1)/np.maximum(norm,1e-8)
    history=data.history(cutoff)[orgs];targets=data.counts(target_year)[orgs].sign()
    repeated=np.asarray(history.sign().multiply(targets).sum(axis=1)).ravel()
    positive=np.asarray(targets.sum(axis=1)).ravel();novel=1-repeated/np.maximum(positive,1)
    valid=norm>1e-10
    rho,p=spearmanr(drift[valid],novel[valid])
    details=pd.DataFrame({'org_id':orgs[valid],'text_fast_slow_cosine_drift':drift[valid],
                          'next_training_year_novel_fraction':novel[valid]})
    details.to_csv(out/(country+'_training_text_drift.csv'),index=False)
    arr=np.load(ROOT/'data/text'/country.lower()/'taxonomy_anchors.npz')
    lookup=dict(zip(arr['codes'],arr['anchors']))
    fixed=np.stack([lookup[c] for c in data.all_leaf_codes])
    dynamic=snap['ipc_slow'][:,:100]
    used=np.array(data.spec['train_candidate_ids'])
    def similarities(values):
        values=values[used];mask=np.linalg.norm(values,axis=1)>1e-8;values=values[mask]
        values=values/(np.linalg.norm(values,axis=1,keepdims=True)+1e-8)
        matrix=values@values.T
        return matrix[np.triu_indices(len(matrix),1)]
    anchored,dyn=similarities(fixed),similarities(dynamic)
    semantics.append(dict(country=country,target_training_year=target_year,n_orgs=int(valid.sum()),
        drift_novel_spearman=float(rho),spearman_p=float(p),
        fixed_anchor_pair_cosine_mean=float(anchored.mean()),dynamic_text_pair_cosine_mean=float(dyn.mean()),
        fixed_anchor_pair_above_095=float((anchored>.95).mean()),dynamic_text_pair_above_095=float((dyn>.95).mean())))
    np.savez_compressed(out/(country+'_ipc_similarity.npz'),fixed=anchored,dynamic_text=dyn)

frame=pd.DataFrame(rows);frame.to_csv(out/'dataset_feasibility.csv',index=False)
timeframe=pd.DataFrame(annual);timeframe.to_csv(out/'annual_counts.csv',index=False)
pd.DataFrame(semantics).to_csv(out/'semantic_probes.csv',index=False)
fig,axes=plt.subplots(1,3,figsize=(10,2.6))
for ax,country in zip(axes,['CN','IN','US']):
    selected=timeframe[timeframe.dataset==country+'_All']
    ax.bar(selected.year,selected.events/1e6,color='#abc9db',edgecolor='#52758b',linewidth=.7)
    ax.set(title=country+' All',xlabel='Publication year',ylabel='Events (million)')
    ax.tick_params(axis='x',rotation=45)
save(fig,'annual_event_counts')
fig,axes=plt.subplots(1,2,figsize=(9,4.4),gridspec_kw={'width_ratios':[1.3,1]})
y=np.arange(len(frame));colors=[{'CN':'#adc9da','IN':'#c7d8b1','US':'#e5c5bd'}[c] for c in frame.country]
axes[0].errorbar(frame.peer_lift,y,xerr=np.array([frame.peer_lift-frame.peer_ci_low,
    frame.peer_ci_high-frame.peer_lift]),fmt='o',color='#496d85',capsize=3)
axes[0].axvline(0,color='.5',ls='--',lw=.8)
axes[0].set(yticks=y,yticklabels=frame.dataset,xlabel='Cooperator coverage lift (95% CI)')
axes[0].invert_yaxis()
axes[1].barh(y,frame.community_ndcg20_gain,color=colors,edgecolor='#666666',hatch='//',linewidth=.6)
axes[1].axvline(0,color='.5',ls='--',lw=.8)
axes[1].set(yticks=y,yticklabels=[],xlabel='Community probe NDCG@20 gain')
axes[1].invert_yaxis()
save(fig,'training_feasibility')
fig,axes=plt.subplots(1,3,figsize=(10,2.7))
for ax,country in zip(axes,['CN','IN','US']):
    sim=np.load(out/(country+'_ipc_similarity.npz'))
    ax.hist(sim['dynamic_text'],bins=40,range=(-1,1),density=True,histtype='step',lw=1.5,label='Historical IPC text')
    ax.hist(sim['fixed'],bins=40,range=(-1,1),density=True,histtype='step',lw=1.5,label='Fixed hierarchy anchor')
    ax.set(title=country,xlabel='IPC pair cosine',ylabel='Density')
axes[-1].legend(fontsize=7)
save(fig,'ipc_semantic_similarity')

lines=['# 原始数据重建后的可行性分析','',
    '本报告由scripts/report_statistics.py生成；数据与图片分别保存在reports/data_analysis和figures/data_analysis。',
    '采用当前冻结的raw-annual-v2。所有模块探针只使用训练时间范围；全期规模统计只描述数据。','',
    '## 数据规模、时序与样本存在性','',
    '| 数据集 | subclass/训练候选 | 训练边 | 验证边 | 测试边 | 验证活跃/空查询 | 验证重复正例比例 |',
    '|---|---:|---:|---:|---:|---:|---:|']
for row in rows:
    lines.append('| {dataset} | {subclasses}/{candidates} | {train_events} | {validation_events} | {test_events} | {validation_active_queries}/{validation_empty_queries} | {validation_repeat_fraction:.3f} |'.format(**row))
lines += ['', '边数是展开后的org–IPC交互次数，不是独立专利记录数。共享专利按申请人与分类数分配消息权重；TOP保留整数频次。',
    '已知组织的空目标占比较高，因此不能通过目标年活跃性筛掉它们。OOV与新组织数量保存在CSV中，主指标只针对历史可查询的组织。',
    '', '## 合作与行为社区的实证依据','']
for row in rows:
    interpretation='区间高于0，支持继续检验' if row['peer_ci_low']>0 else '区间跨越或低于0，证据不足'
    lines.append('- {dataset}：历史合作可用率{peer_available_fraction:.1%}，新分类覆盖宏平均增益{peer_lift:.4f}，95%区间[{peer_ci_low:.4f}, {peer_ci_high:.4f}]，'.format(**row)+interpretation+
                 '；行为社区探针相对全局频次的NDCG@20变化为{community_ndcg20_gain:+.4f}。'.format(**row))
lines += ['', '合作对照仅匹配历史活跃度分桶，是观察性关联，不证明合作导致业务转向。社区探针是32类历史IPC频率聚类，不能视作新模型Fluxon模块已有效的证据。',
    '因此合作必须有缺失mask、可靠性门控及独立消融；社区模块在各视图上单独检查，不假设必然提升。',
    '', '## 语义演化与锚点','']
for value in semantics:
    lines.append('- {country}：{n_orgs}个训练期可观测组织中，快慢文本漂移与下一训练年新类别比例的Spearman相关为{drift_novel_spearman:.4f}；历史IPC文本向量平均两两余弦为{dynamic_text_pair_cosine_mean:.4f}，固定层级锚点为{fixed_anchor_pair_cosine_mean:.4f}。'.format(**value))
lines += ['', '相关分析是训练期探索，Word2Vec使用整个训练期拟合，并非独立留出语料的因果实验。相关弱或为负时，不能宣称语义漂移必然预测新业务。',
    '锚点和动态均值的相似度分布只说明当前表征性质。固定锚点包含共享祖先，其几何有构造因素；最终防趋同效果必须通过去锚点消融与学习后表征检验。',
    '双时间尺度和有界变化权重是可检验设计：对慢变与突变组织分别比较，报告beta=0及单时间尺度对照。当前没有据此宣称该聚合器在所有数据集最优。',
    '', '## 计算与后续验证','',
    '静态充分统计每组超参数仅需一次事件重放，训练读出层的时间与递归baseline分开统计；同时报告预处理成本，不能只比较缓存之后的训练时间。',
    '所有深度方法以验证集选epoch与配置，最终锁定后才评估测试。三种子均值、标准差及按组织bootstrap区间分别保存。']
(ROOT/'docs/feasibility_report.md').write_text('\n'.join(lines)+'\n')
atomic_json(out/'provenance.json',{'protocol_sha256':file_hash(ROOT/'configs/protocol.json'),
    'script_sha256':file_hash(Path(__file__)),'no_test_model_results_consulted':True})
print('Exported 12-view tables, 3 figure families, training semantic probes and detailed report.')
