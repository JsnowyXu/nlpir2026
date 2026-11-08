#!/usr/bin/env python
"""Rebuild the research deck, PDF, figures and notes from a frozen result snapshot."""
import sys,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools/presentation_dependencies'))
from presentation_canvas import Deck,OUT,TEAL,NAVY,BLUE,MINT,LIGHT,CORAL,MUTED
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
font_manager.fontManager.addfont(str(OUT/'assets/NotoSansSC-Regular.ttf'))
plt.rcParams.update({'font.family':font_manager.FontProperties(fname=str(OUT/'assets/NotoSansSC-Regular.ttf')).get_name(),'axes.unicode_minus':False,'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
S=json.loads((OUT/'snapshot/records.json').read_text());DF=pd.read_csv(OUT/'results_numeric.csv')
STATS=pd.read_csv(OUT/'snapshot/dataset_feasibility.csv')
METHODS=S['methods'];DATASETS=S['datasets'];METRICS=S['metrics'];KS=S['ks']
RECORDS=S['records'];COMP=S['components'];ASSETS=OUT/'assets'
STAMP=S['finished_utc'][:19].replace('T',' ')+' UTC'

METRIC_LOOKUP={(r.dataset,r.method,r.metric,int(r.K),r['split']):r for _,r in DF[DF.population=='active'].iterrows()}
def metric(data,method,metric='ndcg',k=20,split='validation'):
    return METRIC_LOOKUP[(data,method,metric,k,split)]
def fmt(row,spread='variance'):
    if row.n==0:return '-'
    if row.n==1:return f'{row["mean"]:.4f} ± -'
    s=row[spread];v='0' if s==0 else (f'{s:.2e}' if s<.001 else f'{s:.6f}')
    return f'{row["mean"]:.4f} ± {v}'
EQCOUNT=0
def eq(p,tex,y=5.6,h=.65,w=11.7,x=.8):
    global EQCOUNT
    import re
    tex=re.sub(r'\\mathcal ([A-Za-z])',r'\\mathcal{\1}',tex)
    tex=tex.replace(r'\frac1n',r'\frac{1}{n}')
    name='equation_%03d.png'%EQCOUNT
    EQCOUNT+=1
    fig=plt.figure(figsize=(12,1));fig.text(.01,.4,'$'+tex+'$',fontsize=23,color='#17324D')
    fig.savefig(ASSETS/name,dpi=200,bbox_inches='tight',pad_inches=.12,transparent=True);plt.close(fig)
    p.image(ASSETS/name,x,y,w,h);p.content.append('数学表达：$'+tex+'$')
def figsave(fig,name):
    for ext in ['png','svg','pdf']:fig.savefig(ASSETS/(name+'.'+ext),dpi=180,bbox_inches='tight')
    plt.close(fig);return ASSETS/(name+'.png')

# Scientific figures use stored numeric results, not generated bitmap measurements.
fig,ax=plt.subplots(figsize=(11.4,5.0));a=np.array([[metric(d,m).n for d in DATASETS] for m in METHODS])
im=ax.imshow(a,cmap=matplotlib.colors.ListedColormap(['#eff2f5','#f5d8b5','#aad6d4','#398c91']),vmin=0,vmax=3,aspect='auto')
ax.set_xticks(range(12));ax.set_xticklabels(DATASETS,rotation=35,ha='right');ax.set_yticks(range(15));ax.set_yticklabels(METHODS)
for i in range(15):
 for j in range(12):ax.text(j,i,f'{a[i,j]}/3',ha='center',va='center',fontsize=8,color='white' if a[i,j]==3 else '#25384A')
ax.set_title('当前正式验证结果覆盖率（三级分类，v2_default）',pad=12)
coverage_plot=figsave(fig,'coverage')

fig,axes=plt.subplots(1,2,figsize=(11.5,4));colors=['#9ac9c6','#b0c6df','#e6c4a5','#cfbddb'];selected=['PatentFlow','PersonalTOP','XGBoost','MENCP']
for ax,m,label in zip(axes,['ndcg','mrr'],['NDCG@20','MRR@20']):
 for i,method in enumerate(selected):
  vals=[metric(d,method,m) for d in ['CN_All','US_All','IN_All']];x=np.arange(3)+(i-1.5)*.19
  ax.bar(x,[v['mean'] for v in vals],.18,yerr=[v['std'] for v in vals],capsize=3,label=method,color=colors[i],edgecolor='#506578',hatch=['//','..','\\\\','xx'][i],linewidth=.5)
 ax.set_xticks(range(3));ax.set_xticklabels(['CN_All','US_All','IN_All']);ax.set_ylim(0,1);ax.set_title(label+' · 验证集');ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
axes[0].legend(fontsize=8,loc='upper left');fig.text(.05,-.02,'柱为3种子均值，误差线为样本标准差；表格另按要求提供均值±方差。',fontsize=9)
main_plot=figsave(fig,'main_comparison')

fig,ax=plt.subplots(figsize=(10.8,3.8));days=np.linspace(0,2190,400)
for tau,color in [(180,'#507EAE'),(1095,'#168A8A')]:ax.plot(days,2**(-days/tau),label=f'半衰期 {tau} 天',color=color,lw=2.5)
ax.axhline(.5,c='#bdc9d3',ls='--');ax.set(xlabel='距离最近更新的天数',ylabel='旧分子/质量的衰减系数',ylim=(0,1.03));ax.legend();ax.grid(alpha=.12)
decay_plot=figsave(fig,'decay')

labels={'ablation_full':'Full','ablation_no_text':'−Text','ablation_no_structure':'−Structure','ablation_no_hierarchy':'−Hierarchy','ablation_no_cooperation':'−Coop','ablation_no_community':'−Community','ablation_no_repeat':'−Repeat','ablation_no_anchor':'−Anchor','ablation_no_alignment':'−Align','ablation_history_only':'History only','ablation_no_text_hierarchy':'−Text+Hierarchy'}
groups={}
for r in COMP:
 v=r['value'];tag=v['config']['tag']
 if v['dataset']=='CN_C' and tag in labels:groups.setdefault(tag,[]).append(v['splits']['validation']['active']['ndcg_20'])
valid=[k for k in labels if len(groups.get(k,[]))==3]
fig,ax=plt.subplots(figsize=(11.5,4));means=[np.mean(groups[k]) for k in valid];stds=[np.std(groups[k],ddof=1) for k in valid]
for i,(m,s) in enumerate(zip(means,stds)):ax.bar(i,m,yerr=s,capsize=3,color=['#b7d7d1','#c1d2e5','#e6cfb8'][i%3],edgecolor='#687886',hatch=['//','..','\\\\'][i%3],linewidth=.7)
ax.set_xticks(range(len(valid)));ax.set_xticklabels([labels[k] for k in valid],rotation=30,ha='right',fontsize=9);ax.set_ylabel('验证 NDCG@20');ax.set_ylim(0,1);ax.set_title('CN_C · 同平台CPU · 每组3种子');ax.grid(axis='y',alpha=.12);ax.set_axisbelow(True)
ablation_plot=figsave(fig,'ablation_cn_c')

sens={}
for r in COMP:
 v=r['value'];tag=v['config']['tag']
 if v['dataset']=='IN_All' and tag in ['sensitivity_lr_0p0003','sensitivity_lr_0p003','ablation_full']:
  sens.setdefault(str(v['config']['lr']),[]).append(v['splits']['validation']['active']['ndcg_20'])
keys=sorted([k for k in sens if len(sens[k])==3],key=float)
fig,ax=plt.subplots(figsize=(10,3.8))
if keys:
 bp=ax.boxplot([sens[k] for k in keys],labels=keys,patch_artist=True,widths=.45)
 for patch in bp['boxes']:patch.set_facecolor('#c3dfd9')
 for i,key in enumerate(keys,1):ax.scatter(np.repeat(i,3)+np.array([-.045,0,.045]),sens[key],color='#17324D',s=25,zorder=3)
ax.set_xlabel('学习率（仅展示已完成3种子的CPU组）');ax.set_ylabel('验证 NDCG@20');ax.set_title('IN_All · 参数敏感性');ax.grid(axis='y',alpha=.12)
sensitivity_plot=figsave(fig,'sensitivity_lr')

D=Deck('PatentFlow_完整汇报')
def two(title,sec,left,right,notes='',subtitle=''):
 p=D.page(title,sec,subtitle,notes);p.card(left[0],left[1],.63,1.99,5.94,4.32);p.card(right[0],right[1],6.78,1.99,5.94,4.32,MINT);return p
def three(title,sec,items,notes='',subtitle=''):
 p=D.page(title,sec,subtitle,notes)
 for i,(a,b) in enumerate(items):p.card(a,b,.63+i*4.11,1.99,3.91,4.32,[LIGHT,MINT,'F8EFE7'][i],17)
 return p

p=D.page('PatentFlow：组织未来专利类别预测','研究汇报 / 当前代码与真实实验','任务、输入、模型、优化、指标、优势与完整结果',
 '本汇报以main_experiment/src/patentflow_main当前实现和冻结结果为准。所有结果按原始raw-annual-v2协议；验证结果不替代测试。实验继续运行，PPT是可追溯的时间快照。概念图由imagegen生成，仅表达研究问题；后续结构图与科学图由代码构建。')
p.image(ASSETS/'concept_overview.png',.65,1.86,8.6,4.92)
p.text('组织 × IPC × 时间',9.40,2.08,3.2,.55,24,TEAL,bold=True)
p.text('12 个数据集\n15 种方法\n3 个随机种子\n4 个截断 K\n6 类排名指标',9.4,2.96,3.0,2.25,22)
p.text('结果冻结：\n'+STAMP,9.4,5.51,3,.65,11,MUTED)
p.note('概念示意图由AI生成；精确计算结构见第13页。')

p=three('汇报路线：从任务到可核查结果','导览',[
 ('01—02  任务与数据','预测什么？\n如何定义历史、候选和标签？\n输入从原始专利如何构造？'),
 ('03—05  模型与评估','七个功能模块及接口。\n得分、loss与参数更新。\n六类指标的逐查询含义。'),
 ('06—07  证据与结果','设计差异和已知局限。\n当前结果、消融与进度。\n附录完整96张结果表。')],notes='正文聚焦解释与证据，附录为12数据集×4个K×验证/测试，共96张表。Excel提供可筛选数值、逐种子记录、均值方差和均值标准差两种显示。')
p.note('全部实验表采用当前已完成结果；没有结果用“-”，不补估计值。')

p=D.page('实验进度：完成、运行与等待分开看','进度',STAMP,'此处冻结的正式结果只统计v2_default；pilot、迁移验证、消融和敏感性不混入基线主表。TOP/PersonalTOP虽然按三个seed留档，但算法本身确定性。')
rows=[]
for m in METHODS:
 rows.append([m,str(sum(int(metric(d,m).n) for d in DATASETS))+'/36',str(sum(int(metric(d,m,split='test').n) for d in DATASETS))+'/36'])
p.table(['方法','验证结果数','测试结果数'],rows,widths=[4.2,3.9,3.9],row_h=.285,size=12)
p.note('主模型及神经baseline尚未执行最终测试；剩余队列继续运行。')

p=two('任务：给定历史，预测下一自然年的IPC集合','01 / 任务',
 ('研究对象','组织 u：公司或机构。\n分类 c：默认IPC第三级subclass。\n历史 H_y：截至 y 年末的可见专利事件。\n输出：组织在 y+1 年涉及的类别排序。'),
 ('预测边界','每个已知组织是一条年度query。\n标签是未来一年出现过的类别集合。\n不预测具体公开专利号或未来文本。\n同一组织可以同时有多个正确类别。'),notes='这是固定预测窗口的多标签排序/集合预测任务。它与EDGPAT研究目标相关，但当前适配严格使用下一自然年，不跳到组织下一次活跃年份。组织名称只能透明筛选企业/机构，原始表不足以证明所有节点均为公司。参考EDGPAT：https://arxiv.org/abs/2308.09780 。')

p=D.page('任务的数学定义：历史、标签与候选','01 / 形式化','L ∈ {1,2,3}；默认 L=3。',
 'U_y是截至y年末已观测组织；C_tr^(L)是训练期实际出现的L级IPC候选。对所有u∈U_y生成query，即使下一年没有专利。真实标签不截断为候选交集：候选外正类在评估分母中保留并计miss。模型只能对训练候选排序，不声称解决新类别冷启动。')
p.text('历史事件包含组织、分类、记录标识、时间与文本向量。',.8,1.99,11.8,.5,20)
eq(p,r'\mathcal{H}_y=\{e=(u,c,p,t,x_p):t\leq T_y\}',2.69,.65)
eq(p,r'Y_u^{y+1}=\{c:\exists p,t\in(T_y,T_{y+1}],\ (u,c,p,t)\in\mathcal{E}\}',3.60,.72)
eq(p,r'\ell_{u,c}^{y+1}=f_\theta(u,c;\mathcal{H}_y),\qquad c\in\mathcal{C}_{\mathrm{tr}}^{(L)}',4.57,.67)
eq(p,r'\widehat{Y}_{u,K}^{y+1}=\operatorname{TopK}_{c\in\mathcal{C}_{\mathrm{tr}}^{(L)}}\ell_{u,c}^{y+1}',5.52,.67)
p.note('候选外正类仍计为未命中；未来新组织不属于当前历史组织query集合，需另行统计。')

p=D.page('时间协议：窗口固定，边界可复查','01 / 时间设置','时间覆盖检查不等于证明数据源已收集所有公开专利。',
 'CN/US观察2010–2019；训练目标年2011–2017，验证2018，测试2019。IN观察2018–2023；训练目标年2019–2021，验证2022，测试2023。每个目标年的输入只读取前一年末memory/计数。固定Word2Vec共享整个训练期；这不使用验证/测试文本拟合，但不应把内部训练探针视为逐时点重新拟合语言模型的在线实验。')
p.table(['语料','观察期','监督训练目标年','验证','测试'],[['CN / US','2010–2019','2011–2017','2018','2019'],['IN','2018–2023','2019–2021','2022','2023']],widths=[2.0,2.6,3.6,1.9,1.9],row_h=.58,size=17)
for i,(label,color) in enumerate([('训练：按年构造历史→下一年标签',BLUE),('验证：选择 epoch 与超参数',TEAL),('测试：配置锁定后只做最终评估',CORAL)]):
 p.rect(.82,3.85+i*.70,.17,.43,color);p.text(label,1.15,3.83+i*.70,10.8,.49,20)

p=two('输入来自四类可追溯信息','02 / 输入',
 ('原始专利记录','组织/申请人列表。\nIPC分类号列表。\n专利公开日期。\n标题与摘要（IN原始表无标题）。\nsource_row是内部记录ID。'),
 ('外部词典与历史上下文','各patent_*下的IPCSection、IPCClass、IPCSubClass。\n过去共同申请形成合作关系。\n历史org–IPC频次和首次出现时间。\n不读取待预测年份的专利文本。'),notes='专利是边上的记录与属性，不是当前主模型独立可训练的第三类图节点。主要节点是organization与IPC。三套语料按国家区分；IN文本是英文，不能表述为三种不同语言。固定IPC词典只用于对应国家语义体系，不从其他baseline预处理目录替代用户指定词典。')

p=D.page('一条多组织、多类别专利如何展开','02 / 事件流','示例仅解释数据构造，不代表真实公司或实验记录。',
 '若记录p具有2个合格组织与3个有效subclass，则生成6条同日org–IPC事件。每条输入消息使用w_p=1/(2×3)=1/6，防止同一专利展开后机械放大。年度频次/TOP仍使用整数事件计数，不把加权memory质量误当独立专利数量。过滤后的多个组织同专利共现可更新彼此最近合作列表。')
p.card('原始记录 p','组织：u1、u2\n类别：c1、c2、c3\n日期：t\n文本：title + abstract',.7,2.05,3.3,3.65)
p.line(4.15,3.8,4.85,3.8,TEAL,2,True)
p.card('展开后的6条事件','u1→c1，u1→c2，u1→c3\nu2→c1，u2→c2，u2→c3\n每条共享同一文本与时间\n基础权重均为 1/6',5.0,2.05,4.1,3.65,MINT,17)
p.line(9.25,3.8,9.65,3.8,TEAL,2,True)
p.card('时序组织','按日排序\n同日先读后写\n保存稳定ID映射\n年度生成标签',9.80,2.05,2.80,3.65,LIGHT,17)
eq(p,r'w_p=\frac{1}{|U_p|\,|C_p|}',5.90,.53)

p=three('清洗、索引与12个视图的依据','02 / 数据构造',[
 ('记录与身份','姓名做NFKC、大小写及空格归一。\n不做模糊公司合并。\n精确去重与日期检查留审计。\n无法识别的申请人单独隔离。'),
 ('IPC与事件','词典外码保留原始记录，隔离主图。\n映射org、subclass、年份与合作索引。\n相同日期不能依赖输入行顺序。\n原始公开编号缺失不自行补造。'),
 ('数据视图','每国All + 训练subclass种类数最多的三个section。\n当前均为B、F、C。\n共12个视图。\n选择依据不是事件总量最多。')],notes='时间窗口外记录保留审计。词典外分类可能来自旧版体系，不能一概判断是假码。节点全集映射可能大于某个视图的实际活跃节点/类别数；幻灯片数据表采用视图统计，不将映射表行数与训练候选数混同。')

p=D.page('12个数据集：规模、候选和查询分布','02 / 数据统计','事件为展开后的org–IPC交互；query为组织×目标年。',
 '三国主图总组织数分别CN182347、US185339、IN40683；这些包含观察期后来出现的组织，不等于验证可查询组织数。表中active/empty是验证目标集合非空/为空的query数。主结果按active宏平均，同时保留all_zero_filled。')
rows=[]
for data in DATASETS:
 r=STATS[STATS.dataset==data].iloc[0]
 rows.append([data,int(r.candidates),f'{int(r.train_events):,}',f'{int(r.validation_events):,}',f'{int(r.test_events):,}',f'{int(r.validation_active_queries):,} / {int(r.validation_empty_queries):,}'])
p.table(['数据集','候选','训练边','验证边','测试边','验证 active / empty'],rows,widths=[1.45,.77,2.03,1.78,1.77,4.3],row_h=.34,size=12)
p.note('CN/US验证查询各91,534 / 150,537；IN为27,943。section视图同样保留全部历史已知组织。')

p=two('文本与IPC锚点：只用训练期拟合语义','02 / 文本及全局语义',
 ('专利向量 x_p ∈ R^100','每国All训练期训练skip-gram Word2Vec。\n维度100；window=5；min_count=5；negative=5；5轮。\n词频也只来自训练期。\n标题+摘要以SIF加权均值编码。'),
 ('固定分类锚点 a_c ∈ R^100','编码分类自身描述及section/class祖先描述。\n取均值后归一化。\n原始锚点保存为不更新的buffer。\n同一国家section视图共享All训练词向量。'),notes='SIF权重q_w=a/(a+f_w)，a=.001，f_w为训练词频。缺词全部落空时用零向量，不生成不存在的标题。IPC描述没有加入Word2Vec的训练句子，只由已训练词向量编码。共享国家All预训练是公开设置，不应称为每个section分别独立预训练。最多512 token，中文jieba、英文正则，max_final_vocab=300000；这些属于预处理配置。IPC全局树通过祖先描述均值进入锚点，当前主网络没有在树上执行可学习消息传播。')
eq(p,r'x_p=\frac{\sum_w q_wv_w}{\sum_wq_w},\quad q_w=\frac{0.001}{0.001+f_w}',5.72,.5)

p=D.page('在线读出到底接收哪些张量？','02 / 模型接口','B：组织query batch；C：训练候选数量；d=100。',
 '主模型不在每个训练batch重新读取原始文本或重放全事件图。MemoryFeatures读取年度快慢缓存，组装节点、合作和历史pair特征；固定anchor和候选结构embedding保存在模型中。合作mask还检查该组织在当前section是否有历史状态。')
p.table(['张量','形状','构造与含义'],[
 ['org_raw','B × 400','fast[text100,structure100] + slow[text100,structure100]'],
 ['ipc_raw','C × 400','IPC快慢缓存；一级/二级按slow质量聚合叶级状态'],
 ['peer_raw','B × 3 × 400','最近三个历史合作方在相同截止年的状态'],
 ['peer_mask','B × 3','合作存在且在当前视图有有效历史状态'],
 ['reliability','B × 8','质量、时距、快慢一致性、合作数及活动量'],
 ['pair_features','B × C × 16','组织-候选历史频次、近期变化及父类占比'],
 ['raw_anchors','C × 100','固定IPC描述层级锚点；作为模型buffer'],
 ],widths=[2.15,2.10,7.85],row_h=.48,size=14)

p=D.page('整体架构：七个功能模块，两阶段计算','03 / 模型结构','阶段A：一次因果缓存；阶段B：可学习读出。',
 '按功能划分7个模块，不等于网络只有7层。M1属于预先构建的固定时序统计；M2–M7构成训练中的神经读出。三种基础得分来自文本动态语义、结构身份、全局层级语义；合作、社区与历史频次作为额外加性分数。全局锚点不会被时间更新覆盖。')
boxes=[(.75,2.15,2.35,1.12,'历史专利事件','文本100 + 结构身份100',LIGHT),(.75,4.15,2.35,1.20,'固定IPC层级锚点','自身描述 + 祖先描述',MINT),
 (3.55,2.15,2.45,1.12,'M1 双时间尺度memory','年度org / IPC 状态400',LIGHT),
 (6.45,1.99,2.52,1.02,'M2 组织双通道','z_sem / z_str ∈ R^100',MINT),
 (6.45,3.32,2.52,1.02,'M3 IPC锚点读出','h_score / anchor / e_c',MINT),
 (9.47,2.25,3.00,1.57,'M4 三分数可靠性融合','基础 logit：B × C',LIGHT),
 (3.55,5.13,2.45,1.10,'M5 合作增强','最近3伙伴 + mask + gate',LIGHT),
 (6.45,5.13,2.52,1.10,'M6 行为社区','32原型，top3路由缓存',LIGHT),
 (9.47,5.13,3.00,1.10,'M7 历史频次残差','16维特征MLP + 类别bias',LIGHT)]
for x,y,w,h,t,b,c in boxes:
 p.rect(x,y,w,h,c,rounded=True);p.text(t,x+.1,y+.12,w-.2,.36,15,TEAL,bold=True);p.text(b,x+.1,y+.59,w-.2,h-.63,11)
p.rect(9.47,4.06,3.00,.69,MINT,rounded=True)
p.text('Σ 基础 + 合作 + 社区 + 历史',9.57,4.23,2.80,.40,13,TEAL,bold=True)
for a,b in [((3.1,2.7),(3.55,2.7)),((6,2.6),(6.45,2.6)),((6,2.95),(6.45,3.68)),((3.1,4.7),(6.45,4.0)),((8.97,2.55),(9.47,2.88)),((8.97,3.81),(9.47,3.41)),((10.98,3.82),(10.98,4.06)),((4.8,5.13),(9.8,4.75)),((7.7,5.13),(10.8,4.75)),((10.98,5.13),(11.9,4.75))]:p.line(*a,*b,BLUE,.9,True)
p.text('输出：最终分数 ℓ → sigmoid解释 / Top-K排序 → 次年真实集合评估',.84,6.49,11.8,.43,15,NAVY)

p=D.page('M1：加权双尺度记忆如何更新','03 / M1','输入：同日事件与日初历史；输出：每节点 fast/slow 的200维均值、质量和age。',
 '消息先在同一天全部从旧状态生成，再按节点汇总提交，避免行顺序影响。org消息由专利文本和固定随机IPC身份组成；IPC消息由文本和日初组织slow结构记忆组成。固定随机身份来自seed42，与IPC语义锚点不同。新组织结构均值为0时余弦按0处理。变化权重依据结构相似度，不是直接由文本漂移决定。memory从零状态开始，固定IPC描述锚点在读出时注入，而非用来初始化事件memory。')
eq(p,r'm_{u,e}=[x_p;R_c],\qquad m_{c,e}=[x_p;M_{u,\mathrm{str}}^{s}(t^-)]',2.0,.6)
eq(p,r'w_e=\frac{1+\beta[1-\cos(M_{u,\mathrm{str}}^s(t^-),R_c)]}{|U_p|\,|C_p|},\quad\beta=1',2.92,.66)
eq(p,r'S_v^\tau(t)=2^{-\Delta t/\tau}S_v^\tau(t^-)+\sum_{e\in E_v(t)}w_em_{v,e}',3.94,.61)
eq(p,r'Z_v^\tau(t)=2^{-\Delta t/\tau}Z_v^\tau(t^-)+\sum_{e\in E_v(t)}w_e,\quad M_v^\tau=S_v^\tau/\max(Z_v^\tau,\epsilon)',4.91,.66)
p.note('τ_fast=180天，τ_slow=1095天；τ与β当前固定，改变它们需重建缓存。')

p=D.page('M1的作用与效率边界','03 / M1','快尺度响应近期变化，慢尺度保留长期背景。',
 '衰减分子和质量相同，单独均值在没有新事件时不会自动衰减到零。因此可靠性输入同时提供衰减质量和距最后事件的age。主要数值更新约O(E·200)，还存在按日分组/去重开销；读出训练另有全候选B×C计算和C²几何损失。不能将一次预计算的复杂度当成整个训练复杂度。缓存使用更多磁盘，400维状态也不自动比所有TGN实现更省内存。')
p.image(decay_plot,.70,1.92,8.8,4.52)
p.card('需要实测的取舍','可复用缓存，减少逐epoch事件重放。\n不建立全时间BPTT图。\n消息参数不可学习。\n成本转移到预计算与读出。',9.69,2.0,2.93,4.24,MINT,17)
p.note('本图是衰减函数示意，不是实验拟合曲线。')

p=D.page('M2：组织文本与结构分开编码','03 / M2','输入 m_u ∈ R^400；输出 z_sem、z_str ∈ R^100。',
 '从fast/slow状态分别拼接两组200维输入。投影为Linear→GELU→LayerNorm→Dropout(.1)，再L2归一化。结构输入是过去IPC身份随机投影的加权统计，而非原生TGN的GRU隐藏状态。结构候选e_c是单独学习的IPC身份embedding，结构分数没有直接与ipc_raw做点积。')
p.card('文本通道','[fast_text100 ; slow_text100]\n捕捉历史专利内容的分布变化。',.75,2.04,5.85,1.64,MINT,17)
p.card('结构通道','[fast_structure100 ; slow_structure100]\n捕捉组织历史类别交互profile。',6.80,2.04,5.80,1.64,LIGHT,17)
eq(p,r'z_u^{\mathrm{sem}}=N(P_{\mathrm{sem}}([M_{u,x}^{f};M_{u,x}^{s}]))',4.0,.60)
eq(p,r'z_u^{\mathrm{str}}=N(P_{\mathrm{str}}([M_{u,R}^{f};M_{u,R}^{s}]))',4.91,.60)
p.note('N表示L2归一化；P表示可学习投影。此处不把随机身份投影称为IPC语义。')

p=D.page('M3：每次读出都重新注入固定语义锚点','03 / M3','输入：IPC动态400维状态 + 固定100维描述锚点；输出：100维IPC评分表征。',
 'raw_anchors是不可训练buffer，但W_A可学习。动态IPC状态经过P_D投影。评分路径先分别归一化静态和动态分支，再对static+.5dynamic归一化。固定锚点保证存在持续语义路径，但不能从公式直接推出类别永不趋同，仍需检查相似度分布与消融。')
eq(p,r'a_c=N\left(\operatorname{mean}_{j\in\{c,\operatorname{anc}(c)\}}\operatorname{Embed}(d_j)\right)',2.13,.73)
eq(p,r'\bar a_c=N(W_Aa_c),\qquad\bar d_c=N(P_D(m_c))',3.30,.66)
eq(p,r'h_c^{\mathrm{score}}=N(\bar a_c+0.5\bar d_c)',4.44,.69)
p.card('固定什么？','分类原始描述锚点 a_c 保持不变。',.8,5.10,5.68,1.22,MINT,14)
p.card('学习什么？','描述投影 W_A 与动态读出 P_D。',6.73,5.10,5.68,1.22,LIGHT,14)

p=D.page('M4：三种基础分数与可靠性融合','03 / M4','输入：组织/IPC编码与8维可靠性；输出：基础候选分数 B×C。',
 '三分数分别为语义组织与锚定动态IPC点积、结构组织与可学习类别身份点积、组织双通道和与静态锚点点积。融合权重是每个组织一个3维softmax，对该组织所有候选共享，不是候选特定注意力。缩放η=exp(logit_scale)且上限30，初始化10。')
eq(p,r's^{\mathrm{sem}}_{uc}=z_u^{\mathrm{sem}\top}h_c^{\mathrm{score}},\qquad s^{\mathrm{str}}_{uc}=z_u^{\mathrm{str}\top}N(e_c)',2.10,.70)
eq(p,r's^{\mathrm{hier}}_{uc}=N(z_u^{\mathrm{sem}}+z_u^{\mathrm{str}})^\top\bar a_c',3.22,.64)
eq(p,r'\alpha_u=\operatorname{softmax}(W_\alpha r_u+b_\alpha),\qquad \ell^0_{uc}=\eta\sum_{b=1}^{3}\alpha_{u,b}s^b_{uc}',4.32,.76)
p.text('作用：利用历史质量与一致性调整三路信息的权重；不是固定等权相加。',.85,5.64,11.7,.66,20)

p=D.page('8维可靠性：全部由截止日前历史构造','03 / M4输入细节','它控制基础融合、合作gate与社区gate。',
 '具体构造与MemoryFeatures一致。第6项为原合作索引的非缺失比例；实际合作贡献还需要peer_mask检查伙伴在当前视图有非零状态。没有伙伴时gate贡献严格为零；禁用合作也会屏蔽该可靠性项。所有时间单位与缩放是当前代码实际常数。')
p.table(['分量','当前实现'],[
 ['r1 / r2','log(1 + fast_mass)/10；log(1 + slow_mass)/10'],
 ['r3','min(slow_age / 3650, 3)'],
 ['r4','cos(fast_text, slow_text)'],
 ['r5','cos(fast_structure, slow_structure)'],
 ['r6','历史合作索引非缺失个数 / 3'],
 ['r7','log(1 + 历史org事件数) / 10'],
 ['r8','max(cutoff_year − first_year + 1, 0) / 20'],
 ],widths=[2.1,10.0],row_h=.51,size=17)

p=D.page('M5：最近三个合作方的带mask增强','03 / M5','输入：3个伙伴的400维状态 + 有效性mask；输出：合作候选分数。',
 '合作来源是同一过去专利记录的共同申请者。使用伙伴在相同截止年的当前视图memory，而不是读取未来合作关系。编码后按有效伙伴取均值并归一化，与IPC评分表征做点积；可靠性sigmoid gate决定增量。合作缺失时贡献为0，不用全局虚拟伙伴填补。')
eq(p,r'p_u=N\left(\frac{\sum_{v\in P_u}b_{uv}N(z_v^{\mathrm{sem}}+z_v^{\mathrm{str}})}{\max(1,\sum_vb_{uv})}\right)',2.17,.88)
eq(p,r'g_u^{\mathrm{coop}}=\sigma(w_{\mathrm{coop}}^\top r_u+b_{\mathrm{coop}})\,\mathbf{1}[\sum_v b_{uv}>0]',3.55,.78)
eq(p,r'\Delta\ell^{\mathrm{coop}}_{uc}=\eta\,g_u^{\mathrm{coop}}\,p_u^\top h_c^{\mathrm{score}}',4.91,.72)
p.note('可解释为历史伙伴上下文；观察性增益不等于合作导致新业务的因果证据。')

p=D.page('M6：行为社区是可学习路由的历史缓存','03 / M6','32个可学习原型；每个组织只路由到top3；temperature=0.5。',
 '组织结构200维profile通过community_query映射，与归一化原型求余弦。选择top3，归一化权重。每年/epoch从全部非零历史组织状态刷新32×400缓存，index_add汇总避免稠密org×org图；刷新处no_grad，缓存统计不回传梯度。预测时router和原型仍经选中路由获得梯度。该机制不是原版IDGP热度模型，不能声称发现真实产业社区。')
eq(p,r'\pi_{uq}=\operatorname{softmax}_{q\in\mathrm{Top3}}\left(N(W_Qm_{u,\mathrm{str}})^\top N(v_q)/T\right)',2.08,.80)
eq(p,r'B_q=\operatorname{stopgrad}\left(\frac{\sum_u\pi_{uq}m_u}{\max(\epsilon,\sum_u\pi_{uq})}\right)',3.29,.78)
eq(p,r'z_u^{\mathrm{com}}=N\left(P_{\mathrm{com}}(\sum_q\pi_{uq}B_q)\right)',4.47,.63)
eq(p,r'\Delta\ell^{\mathrm{com}}_{uc}=\eta\,g_u^{\mathrm{com}}\,(z_u^{\mathrm{com}})^\top h_c^{\mathrm{score}}',5.50,.61)

p=two('M7：保留强历史先验，避免收益归因错误','03 / M7',
 ('16维因果pair特征','4项历史/近年/上年/衰减计数。\n组织份额、近期份额、近期变化、交互间隔。\n全局份额、近期份额和全局变化。\n组织事件数、多样性、年龄。\nclass与section历史份额。'),
 ('MLP残差与类别bias','输入：B × C × 16。\nLayerNorm(16) → Linear(16,64) → GELU → Linear(64,1)。\n乘可学习repeat_scale。\n最后加每个候选的可学习bias。'),notes='频次特征只用截止年和更早数据。其中带half_life=2的历史计数按年衰减，与事件memory的180/1095天尺度不同。应单独消融repeat分支，不能把频次先验带来的收益归因于语义或社区。关闭hierarchy评分分支并不移除pair_features中的父类份额，因此该消融的作用范围必须按代码描述。')
eq(p,r'\Delta\ell^{\mathrm{hist}}_{uc}=\gamma\operatorname{MLP}(f_{uc}),\qquad \gamma_{\mathrm{init}}=1',5.71,.51)

p=D.page('完整pipeline：从缓存到Top-K预测','03 / 端到端工作流','“端到端工作流”不等于“全事件端到端反向传播”。',
 '训练循环：加载国家/section数据和候选；读取各输入年的memory；每epoch、每训练年刷新社区缓存；随机选择最多2万个组织query；组装所有候选特征；前向得到logits；计算联合loss更新读出；年末固定权重验证；保存验证NDCG20最优checkpoint。重建缓存的时间和Word2Vec预训练时间应独立计入总成本。')
steps=[('准备输入','训练期Word2Vec / IPC锚点 / 因果快慢缓存'),('读取年度历史','y年末org、IPC、伙伴、频次与可靠性'),('刷新社区','只聚合当前年度已观测历史状态'),('计算候选分数','M2—M7生成完整 B×C logits'),('训练或推断','训练：联合loss；推断：稳定排序Top-K'),('验证与保存','按验证NDCG@20选优；测试等待配置锁定')]
for i,(a,b) in enumerate(steps):
 y=1.99+i*.71;p.rect(.78,y,.42,.42,TEAL,rounded=True);p.text(str(i+1),.89,y+.045,.20,.28,14,'FFFFFF',record=False)
 p.text(a,1.48,y,2.20,.45,19,NAVY,bold=True);p.text(b,3.77,y+.01,8.66,.47,17)

p=D.page('网络输出：候选logits，而非单一类别','04 / 输出与任务','主输出为 ℓ ∈ R^(B×C)。',
 '每个org-candidate分数综合了三路基础分数、合作增量、社区增量、历史残差和类别bias。sigmoid单调，因此按logits或sigmoid输出排序一致。加权BCE下sigmoid不能未经校准就解释成严格概率；当前任务不使用0.5阈值生成标签，而是Top-K排序。多个分类可同时为真。')
eq(p,r'\ell_{uc}=\ell^0_{uc}+\Delta\ell^{\mathrm{coop}}_{uc}+\Delta\ell^{\mathrm{com}}_{uc}+\gamma\operatorname{MLP}(f_{uc})+b_c',2.21,.86)
eq(p,r'q_{uc}=\sigma(\ell_{uc}),\qquad \widehat{Y}_{u,K}=\operatorname{TopK}_c\ell_{uc}',3.62,.80)
p.card('输入标签','a_uc=1 表示c在下一自然年出现；否则为0。',.80,5.0,5.72,1.25,MINT,17)
p.card('任务落地','用真实未来集合核对候选排序，而非分类准确率。',6.77,5.0,5.72,1.25,LIGHT,17)

p=D.page('Loss ①：加权BCE处理多标签与不均衡','04 / 训练目标','类别正例权重只按完整监督训练query计算。',
 'N_tr是所有训练目标年的历史已知组织query总数；n_c为其中该类出现的query数，按二元标签sign计数，不是patent条数。pos_weight=clip((N_tr−n_c)/(n_c+1),1,30)。一个batch所有候选共同计算BCE，包含空目标query；权重并非根据当前随机采样batch临时估计。')
eq(p,r'w_c^+=\operatorname{clip}\left(\frac{N_{\mathrm{tr}}-n_c}{n_c+1},1,30\right)',2.30,.76)
eq(p,r'\mathcal L_{\mathrm{BCE}}=-\frac{1}{BC}\sum_{u,c}\left[w_c^+a_{uc}\log\sigma(\ell_{uc})+(1-a_{uc})\log(1-\sigma(\ell_{uc}))\right]',3.67,1.01)
p.text('作用：保留多标签监督，降低热门类别对优化的支配；空目标query仍贡献负例项。',.87,5.42,11.6,.95,20)

p=D.page('Loss ②：listwise排名项与几何保持项','04 / 训练目标','默认总loss = BCE + 0.5×rank + 0.1×geometry。',
 'rank只在batch中候选正例集合非空的query上计算。将正例均匀归一化，再与候选log_softmax做交叉熵，不是对六个指标分别直接求导。geometry比较候选表示余弦矩阵与固定原始锚点余弦矩阵；当前正则使用的h_reg归一化顺序与评分h_score不同，下一页明确展示。')
eq(p,r'\mathcal L_{\mathrm{rank}}=-\frac{1}{|\mathcal B_+|}\sum_{u\in\mathcal B_+}\sum_c\frac{a_{uc}}{\sum_ja_{uj}}\log\operatorname{softmax}(\ell_u)_c',2.17,.88)
eq(p,r'\mathcal L_{\mathrm{geo}}=\frac{1}{C^2}\left\|H_{\mathrm{reg}}H_{\mathrm{reg}}^\top-A_0A_0^\top\right\|_F^2',3.58,.77)
eq(p,r'\mathcal L=\mathcal L_{\mathrm{BCE}}+0.5\mathcal L_{\mathrm{rank}}+0.1\mathcal L_{\mathrm{geo}}',4.94,.74)
p.note('几何项保持类别间关系；它不是将所有类别拉向同一公共祖先。')

p=D.page('实现核对：评分、正则与导出向量有差异','04 / 当前代码边界','此处如实描述当前实现；本次汇报没有修改训练模型或重算成绩。',
 '当前forward动态投影先归一化，但anchor_regularization使用未经L2归一化的P_D输出；export_embeddings还使用未经归一化的W_Aa。因此正则并非直接约束评分路径完全相同的h_score，导出向量也不一定等于评分IPC向量。导出的组织向量是N(z_sem+z_str)，不包含合作/社区/频次残差，不能仅用导出向量点积复原全部logits。后续若统一表达式，必须用新配置重跑与比较，不能悄悄替换已有成绩。')
eq(p,r'h_c^{\mathrm{score}}=N\left(N(W_Aa_c)+0.5N(P_D(m_c))\right)',2.19,.73)
eq(p,r'h_c^{\mathrm{reg}}=N\left(N(W_Aa_c)+0.5P_D(m_c)\right)',3.43,.73)
eq(p,r'h_c^{\mathrm{export}}=N\left(W_Aa_c+0.5P_D(m_c)\right)',4.68,.73)
p.note('演化向量是辅助分析输出；不能等同于模型全部决策路径。')

p=three('如何优化：梯度范围、验证选择与后续调参','04 / 优化',[
 ('训练中的梯度','AdamW，lr=0.001。\nweight_decay=0.0001。\nbatch=256；梯度裁剪5。\n最多30轮，patience=5。\n每训练年最多2万query。'),
 ('不会反向更新的部分','已训练Word2Vec不在读出训练中更新。\n预计算memory固定。\nτ和β固定。\n社区缓存刷新no_grad。\n路由及神经读出可学习。'),
 ('选择与继续优化','按验证NDCG@20选择最佳epoch。\n只用验证集调超参数。\n检查各模块三种子消融。\n锁定配置后统一最终测试。\n不按测试集寻找最优参数。')],notes='优化的是候选联合loss；MRR/Recall等为离散排序评估，当前没有直接优化每个指标。快慢/novelty变体重建缓存后再训练；有18组All视图一级/二级预测，需注意候选数量变化导致指标难度不同。resume目前从best而非完整最后状态恢复，不承诺中断后逐位等于不间断训练。')

p=D.page('六类指标：回答六个不同问题','05 / 评估指标','所有指标为“先按组织query计算，再宏平均”；K=10、20、30、40。',
 '展示的AP列是逐query AP@K再宏平均，因此也可称MAP@K。PHR是至少命中一个真实类别的query比例。MRR采用第一个相关结果的倒数，不是对所有相关标签的排名倒数取平均。Precision@K不是多标签分类accuracy。')
p.table(['指标','它衡量什么','在当前任务中的含义'],[
 ['Recall@K','覆盖多少真实类别','未来实际类别中，有多少进入推荐列表'],
 ['Precision@K','推荐列表有多准','推荐的K_eff个类别中有多少为真'],
 ['MRR@K','第一个正确答案有多靠前','组织最快能看到一个正确未来类别的位置'],
 ['AP / MAP@K','多个正确类别是否靠前','每次命中的前缀精确率，按真实标签数归一'],
 ['NDCG@K','排序的折损相关性','越靠前命中贡献越大，再与理想排序比较'],
 ['PHR@K','至少命中一个的组织比例','Top-K是否包含任意真实未来类别'],
 ],widths=[1.9,3.4,6.8],row_h=.62,size=16)

p=D.page('Recall、Precision与PHR的计算','05 / 指标公式','h_i = 1 表示第 i 个推荐类别属于真实集合 Y_u；k = min(K,C)。',
 '对于非空Y_u，Recall分母是真实集合大小，包括不在候选中的正类；Precision分母为实际候选截断数k。候选少于K时不会topk越界。空目标的逐query指标全部置0，用于all_zero_filled，主active平均则排除空目标但没有改变查询构造。')
eq(p,r'R_u@K=\frac{\sum_{i=1}^kh_i}{|Y_u|},\qquad P_u@K=\frac{\sum_{i=1}^kh_i}{k}',2.18,.85)
eq(p,r'\mathrm{PHR}_u@K=\mathbf{1}\left[\sum_{i=1}^kh_i>0\right]',3.70,.79)
p.text('同样命中1个类别：真实集合越大，Recall越低；列表越长，Precision越低；PHR只记一次成功。',.85,5.10,11.65,1.08,21)

p=D.page('MRR、AP与NDCG的计算','05 / 指标公式','二元相关性：一个分类在下一年是否出现，不按该类专利数量加权。',
 'MRR没有命中时为0。AP按min(k,|Y|)归一，包含候选外正类；NDCG理想DCG同样按min(k,|Y|)计算，因此漏掉OOV正类不能无代价消失。代码稳定排序按候选顺序处理相同分数，不根据真实标签打破平局。')
eq(p,r'\mathrm{MRR}_u@K=\max_{1\leq i\leq k}\frac{h_i}{i}',2.02,.62)
eq(p,r'\mathrm{AP}_u@K=\frac{\sum_{i=1}^{k}h_i\left(\frac{1}{i}\sum_{j=1}^{i}h_j\right)}{\min(k,|Y_u|)}',3.09,.89)
eq(p,r'\mathrm{NDCG}_u@K=\frac{\sum_{i=1}^{k}h_i/\log_2(i+1)}{\sum_{i=1}^{\min(k,|Y_u|)}1/\log_2(i+1)}',4.55,1.00)

p=D.page('用一个小例子解释：命中数相同，排序也重要','05 / 指标示例','教学示例使用K=3；正式实验仍为K=10/20/30/40。',
 '真实集合{A,B,C}，候选排序[D,B,A,E]，Top3为[D,B,A]，hit=[0,1,1]。Recall=2/3，Precision=2/3，MRR=1/2，AP=(1/2+2/3)/3=7/18，NDCG=(1/log2(3)+1/log2(4))/(1+1/log2(3)+1/log2(4))≈.5307，PHR=1。把正确类别提前可以提升MRR/AP/NDCG，但在命中数量不变时Recall/Precision/PHR可不变。')
p.card('真实标签 Y','{ A, B, C }',.8,2.0,3.2,1.5,MINT,23)
p.card('预测 Top3','[ D, B, A ]',4.32,2.0,3.6,1.5,LIGHT,23)
p.card('命中向量','[ 0, 1, 1 ]',8.24,2.0,4.25,1.5,LIGHT,23)
p.table(['Recall','Precision','MRR','AP','NDCG','PHR'],[['0.6667','0.6667','0.5000','0.3889','0.5307','1.0000']],x=.8,y=4.22,widths=[1.95]*6,row_h=.60,size=19)
p.note('这组数值仅用于解释公式，不是某个真实数据集的实验成绩。')

p=two('从query指标到结果表：两层聚合','05 / 汇总口径',
 ('第一层：组织query宏平均','active：只对真实集合非空query平均。\nall_zero_filled：所有历史已知组织平均，空目标为0。\n候选外正例计miss。\n未来新组织不混入已知组织评估。'),
 ('第二层：跨种子统计','种子42、2026、3407。\n表格按要求显示均值 ± 样本方差。\nn=1只有均值，方差为“-”。\nn=0整个单元为“-”。\nTOP等确定性方法方差0。'),notes='均值μ=(1/n)Σ_s M_s；样本方差s²=Σ_s(M_s−μ)²/(n−1)。这不是置信区间，方差单位是指标比例的平方。Excel额外提供均值±标准差，原始数值保留更多精度。当前表中的n=2不是三seed完成。神经测试没有结果必须留空，不能从validation搬过去。')
eq(p,r'\bar M=\frac1n\sum_{s=1}^{n}M_s,\qquad s^2=\frac{\sum_s(M_s-\bar M)^2}{n-1}',5.74,.48)

p=three('要解决的关键问题与对应机制','06 / 问题—机制',[
 ('非平稳历史','公司业务与类别使用随时间变化。\n双时间尺度保存近期与长期上下文。\n变化加权降低完全重复事件支配。\n是否有效由memory敏感性验证。'),
 ('结构与语义不完全一致','同结构频次不一定同技术语义。\n文本和随机身份结构分通道。\n固定IPC描述锚点提供语义参考。\n不把层级本身当首次提出的创新。'),
 ('稀疏历史与强重复先验','合作方可补充局部信息，缺失则mask。\n行为社区共享相似组织上下文。\n频次残差保留强先验。\n每个分支必须独立检验。')],notes='所谓优势首先是模型设计预期，不能由动机图直接推导实验胜利。合作是观察性关系，不是因果干预；community是学习路由缓存而非真实社群标签。')

p=D.page('与现有研究比较：差异应准确表述','06 / 方法定位','现有文献的能力不能被简化掉；当前优势尚需完整对照。',
 'EDGPAT已经具有时间memory、层级消息传播和静态/动态/层级预测，不应描述为只用结构。TGN是通用连续时间图memory框架；当前实现通过年度任务适配比较。DyGFormer与RepeatMixer具有历史交互和共现/重复建模能力，不应假定它们完全不考虑历史。当前差异集中在历史专利文本、不可变描述锚点、快慢充分统计、合作与社区缓存以及统一年度全候选协议。参考：https://arxiv.org/abs/2308.09780；https://arxiv.org/abs/2006.10637；https://arxiv.org/abs/2303.13047；https://arxiv.org/abs/2405.17473 。')
p.table(['研究/对照','已有重点','当前设计的差异与边界'],[
 ['EDGPAT','动态memory + IPC层级','文本接入、固定描述锚点与快慢缓存；层级不是新概念'],
 ['TGN / TGN-edgetype','事件消息、memory、图算子','以预计算统计替代逐epoch递归重放；损失端到端性减弱'],
 ['DyGFormer / RepeatMixer','历史交互、共现或重复','原任务适配年度全候选；不预设其必然劣于本方法'],
 ['PersonalTOP / 树模型','强重复和统计先验','明确保留频次残差并消融；避免把频次收益误归语义'],
 ['MENCP / IDGP适配','原任务文本分类 / 热度','只能宣称当前代码的任务适配对比，不等于原论文原任务'],
 ],widths=[2.28,3.1,6.72],row_h=.81,size=14)

p=D.page('可行性证据：支持继续检验，不替代消融','06 / 数据证据','以下统计来自训练期探针；没有用神经模型测试结果选设计。',
 'All视图的合作新类别覆盖增益CN .11934、US .16292、IN .05706，给出bootstrap区间。IN_B和IN_F区间跨0，意味着不能无条件相信合作分支。训练快慢文本漂移与下一训练年新标签比例的Spearman相关是CN−.21491、US−.05583、IN−.18580，数据不支持简单的越漂移越创新假设。当前随机或频率聚类probe也不是学习社区模块效果本身。')
p.table(['All视图','合作覆盖增益','95%区间','文本漂移—新类别相关'],[
 ['CN_All','+0.11934','[0.10752, 0.13055]','−0.21491'],
 ['US_All','+0.16292','[0.14863, 0.17401]','−0.05583'],
 ['IN_All','+0.05706','[0.03602, 0.07794]','−0.18580'],
 ],widths=[2.1,2.8,3.7,3.5],row_h=.63,size=17)
p.text('合作项保留mask和gate；文本变化只能作为状态信息，不能直接声称是创新程度。',.83,5.00,11.65,1.04,22,TEAL)
p.note('数值依据：reports/data_analysis；完整统计见docs/feasibility_report.md。')

p=D.page('当前结果：NDCG提升不等于所有指标领先','07 / 实验结果','All视图；正式默认配置；各显示组均为3种子。',
 'CN_All PatentFlow验证NDCG20 .566516，相较XGBoost .517509；US_All .742010，相较XGBoost .717852；IN_All .567601，相较XGBoost .538884。当前三国All在已完成3seed的对照中NDCG表现较好，但US与IN的MRR低于PersonalTOP，且部分强基线仍未完成。不能宣称所有数据集、所有指标、所有基线均已胜出，更不能把验证提升称为最终泛化结论。')
p.image(main_plot,.62,1.84,12.05,4.66)
p.note('图上误差线是样本标准差；完整实验表按用户要求显示样本方差。')

p=D.page('All视图验证NDCG@20：当前可核查数值','07 / 结果摘要','完整15方法×全部K与全部指标见附录；此页选取已完成方法。',
 '所有值均由冻结的逐seed验证JSON重新计算。报告的±是样本方差，不是标准差。主模型n=3；未完成方法没有填补估算。比较只代表当前默认配置下的结果，没有从大量敏感性配置中挑最高分混入主表。')
show=['PatentFlow','PersonalTOP','XGBoost','RF','MENCP','IDGP']
p.table(['方法','CN_All','US_All','IN_All'],[[m]+[fmt(metric(d,m)) for d in ['CN_All','US_All','IN_All']] for m in show],widths=[2.0,3.36,3.36,3.36],row_h=.61,size=16)
p.note('MENCP为目录方法标识，对应MENPC活跃源码分支适配；不等于完整原论文复现。')

p=D.page('当前完整程度：不能把缺失当成0','07 / 结果覆盖','每个单元为已完成种子数 / 3。',
 '图中0/3代表无已完成正式验证结果，表格用“-”。1/3只有均值、没有样本方差；2/3按已有种子计算，但明确属于部分结果。即使已开始训练，只要validation.json未完整落盘，也不计入成绩。')
p.image(coverage_plot,.70,1.79,11.96,4.91)

p=D.page('已有消融：先展示同平台、完整三种子','07 / 消融','CN_C，验证集；全模型与消融均在CPU运行。',
 '只展示每组已完成3seed的CPU消融，不用GPU默认模型替代CPU对照。每组分数直接读取ablation_*输出；完整组件数据保存在冻结快照。条形图呈现均值、样本标准差与淡色条纹。当前关闭的是代码定义的分支/输入通道，例如hierarchy消融仍可能保留频次特征中的父类份额，因此不能将它解释为删除全部体系结构信息。')
p.image(ablation_plot,.7,1.88,11.95,4.68)
p.note('这是当前可用消融示例；其余数据集等待各组完成后再做完整比较。')

p=D.page('已有敏感性：保留每个种子的分布','07 / 参数敏感性','IN_All学习率；图中只纳入已完成三种子的CPU组。',
 '箱线图的三个原始点分别来自42、2026、3407，不是伪造更多重复实验。默认CPU完整组未完成时不补入默认GPU结果。三点分布只能做初步稳定性观察，不足以替代充分置信区间或显著性检验。其他维度包括dim、community数量、temperature、dropout、rank/alignment权重、weight_decay、每年query预算，以及需重建缓存的时间尺度/novelty。')
p.image(sensitivity_plot,.76,1.92,11.75,4.46)
p.note('所有超参数选择只看验证集；最终测试不参与搜索。')

p=three('已有结论、尚未证明与下一步','07 / 结论边界',[
 ('目前可以确认','完整数据、12视图和统一评估可运行。\n主模型默认配置3seed全部验证完成。\nAll视图NDCG优于部分完成对照。\n三国checkpoint迁移核验一致。'),
 ('目前不能确认','不能宣称全面超越所有baseline。\nIN/US的MRR仍有差距。\n固定锚点不自动证明永不坍塌。\n没有最终神经模型测试结果。\n未完成公平独占资源效率对比。'),
 ('后续顺序','补齐强基线、消融与敏感性。\n核验/统一读出、正则、导出定义。\n在验证集确定模型与参数。\n锁配置后最终测试。\n再完成演化case与论文。')],notes='主实验已经拆到main_experiment，后续主训练从独立目录启动；正在执行的实验不因制作PPT而暂停。若修正归一化定义，须新tag重跑并保留旧成绩。本次PPT没有修改当前模型。')

p=D.page('参考、实现和结果定位','资料索引','引用区分原研究能力与本项目任务适配。',
 '主模型依据main_experiment/src/patentflow_main/{model,train,neural_data,features,dataset,metrics}.py；数据构造依据scripts/build_memory.py、scripts/train_word2vec.py与configs/protocol.json。参考文献的预印本DOI不应冒充会议DOI；本汇报只引用能够核查的原文链接。')
refs=[('EDGPAT','Event-based Dynamic Graph Representation Learning for Patent Application Trend Prediction','https://arxiv.org/abs/2308.09780'),('TGN','Temporal Graph Networks for Deep Learning on Dynamic Graphs','https://arxiv.org/abs/2006.10637'),('DyGFormer','Towards Better Dynamic Graph Learning: New Architecture and Unified Library','https://arxiv.org/abs/2303.13047'),('RepeatMixer','Repeat-Aware Neighbor Sampling for Dynamic Graph Learning','https://arxiv.org/abs/2405.17473')]
for i,(a,b,c) in enumerate(refs):
 y=2.00+i*1.08;p.text(a,.8,y,2.2,.4,18,TEAL,bold=True);p.text(b,3.04,y,9.4,.54,14);p.text(c,3.04,y+.55,9.4,.3,11,MUTED)
p.note('完整数值与逐seed来源：实验结果完整表.xlsx、results_numeric.csv、snapshot/records.json。')

BODY_COUNT=len(D.pages)
p=D.page('附录：完整实验表格的阅读方式','附录 / 表格规则','12数据集 × 4个K × 2个split = 96张表；每张15种方法、6项指标。',
 '顺序为验证/测试，各按CN、US、IN及All/B/F/C展开，每数据集K10/20/30/40。所有主模型与神经baseline采用v2_default、level3；传统baseline是同协议下自己的验证选参结果。均值与方差按已有seed计算；1seed方差不存在，0seed为“-”。神经测试未执行，必须留“-”。主表按active query，all_zero_filled在Excel数值页另存。')
p.card('列的含义','方法 | n/3 | MRR | Recall | AP | NDCG | PHR | Precision\n指标的K由每页标题指定。\n每个数值为：均值 ± 样本方差。',.82,2.03,11.65,2.10,MINT,22)
p.card('不要混读','验证与测试年份不同，不能横向替换。\n不同K、国家、section或level的难度不同。\n三个seed只覆盖该模型训练随机性；共享预训练输入不提供独立语料重复。',.82,4.48,11.65,1.79,LIGHT,20)
for split,label in [('validation','验证集'),('test','测试集')]:
 for data in DATASETS:
  year=(2022 if split=='validation' else 2023) if data.startswith('IN') else (2018 if split=='validation' else 2019)
  for k in KS:
   rows=[]
   for method in METHODS:
    first=metric(data,method,k=k,split=split)
    rows.append([method,f'{int(first.n)}/3']+[fmt(metric(data,method,m,k,split)) for m in METRICS])
   p=D.page(f'{data}  ·  {label} {year}  ·  K={k}', '附录 / 完整结果',
     'active query宏平均；单元格=均值 ± 样本方差；n为已完成seed数。',
     f'{data} {split} K={k}，level3。冻结时间{STAMP}。只聚合v2_default正式记录，不挑选消融或超参数最佳值。n=0用-占位；n=1只给均值、方差用-。AP列为宏平均AP@K。测试缺失不能用验证结果替代。')
   p.table(['方法','n','MRR','Recall','AP','NDCG','PHR','Precision'],rows,widths=[1.78,.52]+[1.63]*6,row_h=.285,size=10.9)
   p.note('TOP/PersonalTOP为确定性方法；“±0”不代表总体不确定性为0。完整精度与标准差另见Excel。')
D.save()
# Record slide navigation and page provenance, independent of the live results tree.
(OUT/'slide_index.json').write_text(json.dumps({'body_slides':BODY_COUNT,'table_appendix_slides':96,'total_slides':len(D.pages),'snapshot_utc':S['finished_utc'],'titles':[p.title for p in D.pages]},ensure_ascii=False,indent=2))
print('Created slides',len(D.pages),'body',BODY_COUNT,'table appendix',96)
