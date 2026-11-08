#!/usr/bin/env python
"""Freeze existing results and aggregate literal mean ± sample variance."""
import sys,json,hashlib,argparse
from pathlib import Path
from datetime import datetime,timezone
from collections import defaultdict
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools/presentation_dependencies'))
import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font,PatternFill,Alignment
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output-dir', type=Path, default=ROOT/'reports/presentation_20260919')
args=parser.parse_args()
OUT=args.output_dir.resolve();OUT.mkdir(parents=True,exist_ok=True)
(OUT/'snapshot').mkdir(exist_ok=True)
DATASETS=[c+'_'+s for c in ['CN','US','IN'] for s in ['All','B','F','C']]
METHODS=['PatentFlow','EDGPAT','MENCP','RepeatMixer','TGAT','DyGFormer','TAMI','IDGP','TGN','TGN_edgetype','TOP','PersonalTOP','SVM','XGBoost','RF']
METRICS=['mrr','recall','ap','ndcg','phr','precision'];SEEDS={42,2026,3407}
protocol=hashlib.sha256((ROOT/'configs/protocol.json').read_bytes()).hexdigest()
start=datetime.now(timezone.utc).isoformat();records=[];components=[];seen=set();rejected=[]
paths=sorted((ROOT/'results/baselines').glob('*/*/metrics.json'))
for top in ['main','deep']:
 base=ROOT/'results'/top
 pattern='*/v2_default/s*' if top=='main' else '*/*/v2_default/s*'
 for folder in sorted(base.glob(pattern)):
  p=folder/('metrics.json' if (folder/'metrics.json').exists() else 'validation.json')
  if p.exists():paths.append(p)
for p in paths:
 raw=p.read_bytes();v=json.loads(raw);cfg=v.get('config',{})
 if v.get('protocol_sha256')!=protocol or v.get('dataset') not in DATASETS or v.get('seed') not in SEEDS or v.get('level',cfg.get('level',3))!=3:
  rejected.append(str(p.relative_to(ROOT)));continue
 method=v.get('method');key=(v['dataset'],method,v['seed'])
 if method not in METHODS:rejected.append(str(p.relative_to(ROOT)));continue
 if key in seen:raise ValueError('Duplicate official run '+str(key))
 seen.add(key);records.append(dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(raw).hexdigest(),value=v))
for p in sorted((ROOT/'results/main').glob('*/*/s*/validation.json')):
 v=json.loads(p.read_bytes());tag=v.get('config',{}).get('tag','')
 if v.get('protocol_sha256')==protocol and tag.startswith(('ablation_','sensitivity_','memory_','level')):
  components.append(dict(path=str(p.relative_to(ROOT)),value=v))
snapshot={'started_utc':start,'finished_utc':datetime.now(timezone.utc).isoformat(),'protocol_sha256':protocol,'records':records,'components':components,'excluded_paths':rejected,'methods':METHODS,'datasets':DATASETS,'metrics':METRICS,'ks':[10,20,30,40],'aggregation':'sample variance ddof=1 across available seeds; n=1 variance missing; n=0 all missing; no pilot or best-over-tags selection'}
(OUT/'snapshot/records.json').write_text(json.dumps(snapshot,ensure_ascii=False,indent=2))
(OUT/'snapshot/progress.json').write_bytes((ROOT/'state/progress.json').read_bytes())
source_paths=['main_experiment/src/patentflow_main/'+x+'.py' for x in ['model','train','neural_data','features','dataset','metrics','evaluate']]+['scripts/build_memory.py','scripts/train_word2vec.py','src/patentflow/moments.py','configs/protocol.json']
(OUT/'snapshot/code_hashes.json').write_text(json.dumps({p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in source_paths},indent=2))
bykey=defaultdict(list);seedrows=[];coverage=[]
for record in records:
 v=record['value']
 for split,obj in v['splits'].items():
  if split not in ['validation','test']:continue
  coverage.append(dict(dataset=v['dataset'],method=v['method'],seed=v['seed'],split=split,path=record['path'],n_queries=obj['n_queries'],n_active=obj['n_active'],n_empty=obj['n_empty'],n_candidates=obj['n_candidates'],oov_positive_count=obj['oov_positive_count'],positive_count=obj['positive_count']))
  for pop in ['active','all_zero_filled']:
   for k in [10,20,30,40]:
    for metric in METRICS:
     value=obj[pop].get(metric+'_'+str(k))
     if value is None:continue
     key=(v['dataset'],v['method'],split,pop,k,metric)
     bykey[key].append((v['seed'],float(value)))
     seedrows.append(dict(zip(['dataset','method','split','population','K','metric'],key),seed=v['seed'],value=value,path=record['path']))
rows=[]
for data in DATASETS:
 for method in METHODS:
  for split in ['validation','test']:
   for pop in ['active','all_zero_filled']:
    for k in [10,20,30,40]:
     for metric in METRICS:
      key=(data,method,split,pop,k,metric);vals=bykey[key];a=np.array([v for _,v in vals])
      rows.append(dict(zip(['dataset','method','split','population','K','metric'],key),n=len(a),seeds=','.join(str(s) for s,_ in sorted(vals)),mean=float(a.mean()) if len(a) else None,variance=float(a.var(ddof=1)) if len(a)>1 else None,std=float(a.std(ddof=1)) if len(a)>1 else None))
df=pd.DataFrame(rows);df.to_csv(OUT/'results_numeric.csv',index=False)
pd.DataFrame(seedrows).to_csv(OUT/'results_per_seed.csv',index=False)
pd.DataFrame(coverage).to_csv(OUT/'run_coverage.csv',index=False)
def formatted(mean,spread):
 if pd.isna(mean):return '-'
 if pd.isna(spread):return f'{mean:.4f} ± -'
 s='0' if spread==0 else (f'{spread:.2e}' if spread<.001 else f'{spread:.6f}')
 return f'{mean:.4f} ± {s}'
wide={}
for spread in ['variance','std']:
 tmp=df.copy();tmp['display']=[formatted(a,b) for a,b in zip(tmp['mean'],tmp[spread])]
 wide[spread]=tmp.pivot(index=['dataset','method','split','population','K','n'],columns='metric',values='display').reset_index()
 wide[spread].to_csv(OUT/f'results_mean_{spread}.csv',index=False)
wb=Workbook();ws=wb.active;ws.title='说明'
notes=[['字段','规则'],['时间范围',start+' 至 '+snapshot['finished_utc']],['协议',protocol],['主表','三级分类；v2_default；按已有种子聚合；不择优混用消融/调参tag'],['显示格式','按用户要求：均值 ± 样本方差；方差分母n−1；不是置信区间'],['种子','42, 2026, 3407；n字段标记当前完整结果数'],['缺失','n=0：-；n=1：均值 ± -；n=2：部分结果，不能称3种子完成'],['指标人群','均值方差/标准差表采用active；原始数值另含all_zero_filled'],['验证/测试','完全分开；未执行神经模型测试不填验证值'],['统计量','先每个seed按组织query宏平均，再跨seed求均值和样本方差'],['确定性方法','TOP与PersonalTOP三seed相同，方差0不表示估计不确定性为0'],['单位','指标为0~1原始比例；方差单位为比例的平方'],['AP与PHR','AP=各查询AP@K的宏平均；PHR=至少命中1类的查询比例'],['MENCP','保留目录/日志方法名；实际对应MENPC论文的给定活跃代码分支适配'],['空白界限','模型全部候选得分；不是负采样评估；未报告阈值0.5分类准确率']]
for row in notes:ws.append(row)
def sheet(name,frame):
 ws=wb.create_sheet(name);ws.append(list(frame.columns))
 for row in frame.itertuples(index=False,name=None):ws.append([None if isinstance(v,float) and not np.isfinite(v) else v for v in row])
 ws.freeze_panes='C2';ws.auto_filter.ref=ws.dimensions
 return ws
for split,label in [('validation','验证'),('test','测试')]:
 for spread,label2 in [('variance','均值方差'),('std','均值标准差')]:
  z=wide[spread];z=z[(z.split==split)&(z.population=='active')]
  z=z[['dataset','method','K','n']+METRICS]
  sheet(label2+'_'+label,z)
sheet('原始数值',df);sheet('逐种子',pd.DataFrame(seedrows));sheet('运行覆盖',pd.DataFrame(coverage));sheet('数据集',pd.read_csv(ROOT/'reports/data_analysis/dataset_feasibility.csv'))
for ws in wb:
 for cell in ws[1]:cell.fill=PatternFill('solid',fgColor='16324F');cell.font=Font(color='FFFFFF',bold=True)
 for col in ws.columns:
  letter=col[0].column_letter;ws.column_dimensions[letter].width=22 if letter not in ['A','B'] else 24
 for row in ws.iter_rows(min_row=2):
  for cell in row:
   if isinstance(cell.value,float):cell.number_format='0.00000000'
ws=wb['说明'];ws.column_dimensions['B'].width=100
wb.save(OUT/'实验结果完整表.xlsx')
print('Frozen runs',len(records),'component runs',len(components),'numeric cells',len(df),'timestamp',snapshot['finished_utc'])
for method in METHODS:
 print(method,sum(r['value']['method']==method for r in records))
