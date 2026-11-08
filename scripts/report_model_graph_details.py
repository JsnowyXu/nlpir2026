"""Read-only graph statistics for the actual processed event streams and views."""
import sys,json
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools/presentation_dependencies')]
from patentflow.dataset import DatasetView
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=ROOT/'reports/model_graph_details_20260920';OUT.mkdir(exist_ok=True)
raws=[];graphs=[];annual=[];splits=[];degrees=[]
for country in ['cn','us','in']:
 a=json.loads((ROOT/f'reports/raw_audit/{country}.json').read_text());m=json.loads((ROOT/f'data/processed/{country}/manifest.json').read_text())
 e=np.load(ROOT/f'data/processed/{country}/events.npz');tax=json.loads((ROOT/'reports/taxonomy_audit.json').read_text())[country]
 raws.append(dict(country=country.upper(),raw_rows=a['rows_read'],exact_duplicate_rows=a['duplicates_exact_content'],exact_duplicate_rate=a['duplicates_exact_content']/a['rows_read'],retained_after_raw_cleaning=a['retained_unique_records'],raw_clean_org_names=a['organizations'],raw_clean_subclass_strings=a['subclasses'],raw_title_missing=a['title_missing'],raw_abstract_missing=a['abstract_missing'],raw_joint_records=a['joint_applicant_records'],main_patent_records=sum(m['record_counts_by_year'].values()),main_joint_records=sum(m['joint_records_by_year'].values()),allocated_ipc_ids=m['n_subclasses'],taxonomy_sections=tax['counts_by_level']['1'],taxonomy_classes=tax['counts_by_level']['2'],taxonomy_subclasses=tax['counts_by_level']['3'],train_start=m['observation_start_year'],train_end=m['train_end_year'],validation_year=m['validation_year'],test_year=m['test_year']))
 for sec in ['All','B','F','C']:
  name=country.upper()+'_'+sec;d=DatasetView(name);mask=np.isin(e['dst'],d.spec['subclass_ids']);u=e['src'][mask];v=e['dst'][mask];y=e['year'][mask];pid=e['patent_row'][mask]
  pair=np.unique(u.astype(np.int64)*m['n_subclasses']+v);nu=len(np.unique(u));nc=len(np.unique(v));ne=len(u);npair=len(pair)
  graph=dict(dataset=name,organization_id_pool=m['n_organizations'],edge_incident_org_nodes=nu,ipc_nodes=nc,total_incident_nodes=nu+nc,candidates=len(d.candidate_ids),patent_records=len(np.unique(pid)),temporal_edges=ne,unique_org_ipc_pairs=npair,extra_repeat_edges=ne-npair,event_repeat_rate=1-npair/ne,bipartite_density=npair/(nu*nc),mean_org_unique_degree=npair/nu,mean_ipc_unique_degree=npair/nc,mean_org_event_degree=ne/nu,mean_ipc_event_degree=ne/nc)
  graphs.append(graph)
  for typ,values in [('org_event_degree',np.bincount(u)),('ipc_event_degree',np.bincount(v)),('org_unique_degree',np.bincount(pair//m['n_subclasses'])),('ipc_unique_degree',np.bincount(pair%m['n_subclasses']))]:
   values=values[values>0];degrees.append(dict(dataset=name,degree_type=typ,minimum=values.min(),median=np.median(values),p90=np.quantile(values,.9),p99=np.quantile(values,.99),maximum=values.max(),mean=values.mean()))
  if sec=='All':
   hist=np.bincount(u);hist=hist[hist>0];xs=np.unique(hist);ys=np.array([(hist>=x).mean() for x in xs]);plt.loglog(xs,ys,label=country.upper())
  for year in m['years']:
   mm=y==year;orgs,truth=d.queries(year);act=sum(bool(len(t)) for t in truth)
   annual.append(dict(dataset=name,year=year,patent_records=len(np.unique(pid[mm])),events=int(mm.sum()),incident_org_nodes=len(np.unique(u[mm])),incident_ipc_nodes=len(np.unique(v[mm])),known_queries=len(orgs),active_queries=act,empty_queries=len(orgs)-act))
  st=json.loads((ROOT/f'reports/dataset_statistics/{name}.json').read_text())
  for split,sv in st['splits'].items():
   splits.append(dict(dataset=name,split=split,**{k:(','.join(map(str,val)) if isinstance(val,list) else val) for k,val in sv.items() if not isinstance(val,dict)}))
plt.xlabel('Organization event degree');plt.ylabel('P(degree >= x)');plt.legend();plt.grid(alpha=.2);plt.tight_layout();plt.savefig(OUT/'organization_degree_ccdf.png',dpi=200);plt.savefig(OUT/'organization_degree_ccdf.pdf');plt.close()
tables={'原始数据':pd.DataFrame(raws),'12视图图统计':pd.DataFrame(graphs),'年度统计':pd.DataFrame(annual),'划分和标签重复':pd.DataFrame(splits),'度分布':pd.DataFrame(degrees),'合作社区探针':pd.read_csv(ROOT/'reports/data_analysis/dataset_feasibility.csv')}
for name,df in tables.items():df.to_csv(OUT/(name+'.csv'),index=False)
with pd.ExcelWriter(OUT/'三国数据与图统计.xlsx',engine='openpyxl') as writer:
 for name,df in tables.items():
  df.to_excel(writer,sheet_name=name,index=False);ws=writer.book[name];ws.freeze_panes='B2';ws.auto_filter.ref=ws.dimensions
  for col in ws.columns:ws.column_dimensions[col[0].column_letter].width=25
fig,axs=plt.subplots(1,3,figsize=(13,3.4))
for ax,c in zip(axs,['CN','US','IN']):
 z=tables['年度统计'];z=z[z.dataset==c+'_All'];ax.bar(z.year,z.events,color='#a7c8d9');ax.set_title(c+' All');ax.set_xlabel('Year');ax.set_ylabel('Temporal edges')
fig.tight_layout();fig.savefig(OUT/'annual_edges.png',dpi=200);fig.savefig(OUT/'annual_edges.pdf');plt.close(fig)
print(tables['原始数据'].to_string(index=False));print(tables['12视图图统计'].to_string(index=False));print(tables['划分和标签重复'][['dataset','split','events','active_queries','empty_queries','repeat_fraction','oov_target_count']].to_string(index=False))
