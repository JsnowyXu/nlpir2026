#!/usr/bin/env python
import sys,json,hashlib,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools/presentation_dependencies'))
import numpy as np,pandas as pd
import fitz
from pptx import Presentation
from PIL import Image,ImageDraw
OUT=ROOT/'reports/presentation_20260919';snapshot=json.loads((OUT/'snapshot/records.json').read_text())
rows=pd.read_csv(OUT/'results_numeric.csv')
assert len(rows)==12*15*2*2*4*6
assert not rows.duplicated(['dataset','method','split','population','K','metric']).any()
assert rows.loc[rows.n==0,['mean','variance','std']].isna().all().all()
assert rows.loc[rows.n==1,['variance','std']].isna().all().all()
assert np.allclose(rows.loc[rows.n>=2,'variance'],rows.loc[rows.n>=2,'std']**2)
# Recompute every numeric cell directly from the frozen seed JSONs.
lookup={}
for r in snapshot['records']:
 v=r['value']
 for split,metrics in v['splits'].items():
  for population in ['active','all_zero_filled']:
   for key,value in metrics[population].items():
    name,k=key.rsplit('_',1);lookup.setdefault((v['dataset'],v['method'],split,population,int(k),name),[]).append(value)
for row in rows.itertuples(index=False):
 values=lookup.get((row.dataset,row.method,row.split,row.population,row.K,row.metric),[])
 assert len(values)==row.n
 if values:assert abs(np.mean(values)-row.mean)<1e-13
 if len(values)>1:assert abs(np.var(values,ddof=1)-row.variance)<1e-13
code_hashes=json.loads((OUT/'snapshot/code_hashes.json').read_text())
for relative,digest in code_hashes.items():
 source=ROOT/relative;assert hashlib.sha256(source.read_bytes()).hexdigest()==digest
 dest=OUT/'snapshot/code'/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
prs=Presentation(OUT/'PatentFlow_完整汇报.pptx');doc=fitz.open(OUT/'PatentFlow_完整汇报.pdf')
index=json.loads((OUT/'slide_index.json').read_text());assert len(prs.slides)==len(doc)==index['total_slides']==140
assert not json.loads((OUT/'layout_audit.json').read_text())['overflow']
assert len([t for t in index['titles'] if 'K=' in t])==96
from fontTools.ttLib import TTFont
cmap=TTFont(OUT/'assets/NotoSansSC-Regular.ttf').getBestCmap()
missing_glyphs=sorted({c for slide in prs.slides for shape in slide.shapes if shape.has_text_frame for c in shape.text if not c.isspace() and ord(c) not in cmap})
assert not missing_glyphs,missing_glyphs
bbox_issues=[]
for i,slide in enumerate(prs.slides):
 for shape in slide.shapes:
  if shape.left<0 or shape.top<0 or shape.left+shape.width>prs.slide_width+50 or shape.top+shape.height>prs.slide_height+50:bbox_issues.append((i+1,shape.name))
assert not bbox_issues,bbox_issues[:10]
previews=OUT/'previews';previews.mkdir(exist_ok=True)
selected=[0,4,8,12,13,17,22,26,28,31,35,38,43,44,92,139]
canvas=Image.new('RGB',(1600,1040),'#e9eef3');draw=ImageDraw.Draw(canvas)
for i,pageno in enumerate(selected):
 pix=doc[pageno].get_pixmap(matrix=fitz.Matrix(1.4,1.4),alpha=False)
 path=previews/f'slide_{pageno+1:03d}.png';pix.save(path)
 image=Image.open(path);image.thumbnail((390,220))
 x=(i%4)*400+5;y=(i//4)*260+5;canvas.paste(image,(x,y));draw.text((x+3,y+225),f'Slide {pageno+1:03d}',fill='#17324D')
canvas.save(previews/'contact_sheet.png')
# Short PDF includes the substantive presentation only; complete PPT/PDF retains all tables.
short=fitz.open();short.insert_pdf(doc,from_page=0,to_page=index['body_slides']-1);short.save(OUT/'PatentFlow_正文43页.pdf');short.close()
# Also retain exact vector versions of key schematic pages.
for i in [4,5,10,11,12,13,22,23,25,26]:
 (OUT/'assets'/f'slide_{i+1:03d}_vector.svg').write_text(doc[i].get_svg_image())
report={'all_numeric_cells_recomputed':len(rows),'missing_and_partial_seed_rules_passed':True,'variance_equals_std_squared':True,'missing_glyphs':missing_glyphs,'slides':len(doc),'editable_ppt_shapes':sum(len(s.shapes) for s in prs.slides),'full_result_tables':96,'out_of_bounds_shapes':bbox_issues,'source_hashes_verified':len(code_hashes),'snapshot_utc':snapshot['finished_utc']}
for page,name in [(12,'model_architecture'),(22,'annual_pipeline'),(4,'task_formalization'),(13,'weighted_memory'),(23,'ranking_output')]:
    doc[page].get_pixmap(matrix=fitz.Matrix(2,2),alpha=False).save(OUT/'assets'/(name+'.png'))
    (OUT/'assets'/(name+'.svg')).write_text(doc[page].get_svg_image())
    single=fitz.open();single.insert_pdf(doc,from_page=page,to_page=page);single.save(OUT/'assets'/(name+'.pdf'));single.close()
(OUT/'audit_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps(report,indent=2))
