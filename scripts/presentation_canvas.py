"""Small dual renderer: editable PPT shapes/text plus matching vector PDF."""
import sys,math,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools/presentation_dependencies'))
from pptx import Presentation
from pptx.util import Inches,Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE,MSO_CONNECTOR
from pptx.enum.text import MSO_ANCHOR
from pptx.oxml.xmlchemy import OxmlElement
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import HexColor
from PIL import Image

OUT=ROOT/'reports/presentation_20260919'
FONT=OUT/'assets/NotoSansSC-Regular.ttf'
pdfmetrics.registerFont(TTFont('NotoSC',str(FONT)))
NAVY='17324D';TEAL='168A8A';BLUE='507EAE';INK='25384A';MUTED='60758A';LIGHT='F1F5F8';MINT='E6F3EF';CORAL='D97C61'
W,H=13.333333,7.5

def wrap(text,width,size):
    result=[]
    for paragraph in str(text).split('\n'):
        line=''
        for ch in paragraph:
            if pdfmetrics.stringWidth(line+ch,'NotoSC',size)>width*72*.96 and line:
                result.append(line);line=ch
            else:line+=ch
        result.append(line)
    return result

class Deck:
    def __init__(self,stem):
        self.prs=Presentation();self.prs.slide_width=Inches(W);self.prs.slide_height=Inches(H)
        self.pdf=canvas.Canvas(str(OUT/(stem+'.pdf')),pagesize=(W*72,H*72))
        self.pdf.setTitle('PatentFlow：当前实现与实验结果汇报')
        self.stem=stem;self.pages=[];self.overflow=[]
    def page(self,title,section='',subtitle='',notes=''):
        if self.pages:self.pdf.showPage()
        s=Page(self,title,section,subtitle,notes);self.pages.append(s);return s
    def save(self):
        self.pdf.save()
        for p in self.pages:p.slide.notes_slide.notes_text_frame.text=p.title+'\n'+p.notes+'\n\n'+'\n\n'.join(p.content)
        self.prs.save(OUT/(self.stem+'.pptx'))
        md=['# PatentFlow 完整汇报逐页讲稿','', '本文件与PPT同源生成；实验表中的±为样本方差，具体口径见统计说明页。','']
        for i,p in enumerate(self.pages,1):
            md+=['## 第%d页｜%s'%(i,p.title),'',p.notes,'','\n\n'.join(p.content),'']
        (OUT/(self.stem+'_逐页讲稿.md')).write_text('\n'.join(md))
        (OUT/'layout_audit.json').write_text(json.dumps({'slides':len(self.pages),'overflow':self.overflow},ensure_ascii=False,indent=2))
        if self.overflow:raise RuntimeError('Layout overflow: '+str(self.overflow[:5]))

class Page:
    def __init__(self,deck,title,section,subtitle,notes):
        self.deck=deck;self.slide=deck.prs.slides.add_slide(deck.prs.slide_layouts[6]);self.title=title;self.notes=notes;self.content=[]
        self.rect(0,0,W,H,'FFFFFF');self.rect(0,0,.14,H,TEAL)
        self.text(section.upper(),.6,.23,11.8,.24,10,TEAL,bold=True,record=False)
        self.text(title,.6,.67,12.1,.66,27,NAVY,bold=True,record=False)
        if subtitle:self.text(subtitle,.63,1.31,12,.46,12,MUTED,record=True)
        self.line(.6,7.12,12.72,7.12,'D7E1E8',.7)
        self.text('PatentFlow  /  raw-annual-v2  /  当前实现与冻结结果',.62,7.19,10.8,.18,8.4,MUTED,record=False)
        self.text('%03d'%(len(deck.pages)+1),12.05,7.17,.62,.23,10,TEAL,record=False)
        self.slide.notes_slide.notes_text_frame.text=title+'\n'+notes
    def rect(self,x,y,w,h,fill=LIGHT,stroke=None,rounded=False):
        sh=self.slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE,Inches(x),Inches(y),Inches(w),Inches(h))
        sh.fill.solid();sh.fill.fore_color.rgb=RGBColor.from_string(fill)
        if stroke:sh.line.color.rgb=RGBColor.from_string(stroke);sh.line.width=Pt(.65)
        else:sh.line.fill.background()
        c=self.deck.pdf;c.setFillColor(HexColor('#'+fill));c.setStrokeColor(HexColor('#'+(stroke or fill)));c.setLineWidth(.65)
        if rounded:c.roundRect(x*72,(H-y-h)*72,w*72,h*72,7,fill=1,stroke=bool(stroke))
        else:c.rect(x*72,(H-y-h)*72,w*72,h*72,fill=1,stroke=bool(stroke))
    def text(self,text,x,y,w,h,size=18,color=INK,bold=False,record=True,align='left'):
        original=size
        while True:
            lines=wrap(text,w,size)
            if len(lines)*size*1.28<=h*72 or size<=8:break
            size-=.5
        if len(lines)*size*1.28>h*72+1:self.deck.overflow.append(dict(slide=self.title,text=str(text)[:80]))
        if record:self.content.append(str(text))
        sh=self.slide.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h));tf=sh.text_frame;tf.clear();tf.word_wrap=False
        tf.margin_top=tf.margin_bottom=tf.margin_left=tf.margin_right=0
        for i,line in enumerate(lines):
            p=tf.paragraphs[0] if i==0 else tf.add_paragraph();p.text=line;p.space_before=Pt(0);p.space_after=Pt(0);p.line_spacing=Pt(size*1.28)
            if align=='center':
                from pptx.enum.text import PP_ALIGN
                p.alignment=PP_ALIGN.CENTER
            for r in p.runs:
                r.font.name='Noto Sans SC';r.font.size=Pt(size);r.font.bold=bold;r.font.color.rgb=RGBColor.from_string(color)
                ea=OxmlElement('a:ea');ea.set('typeface','Noto Sans SC');r._r.get_or_add_rPr().append(ea)
        c=self.deck.pdf;c.setFillColor(HexColor('#'+color));c.setFont('NotoSC',size)
        for i,line in enumerate(lines):
            xx=x*72
            if align=='center':xx+=(w*72-pdfmetrics.stringWidth(line,'NotoSC',size))/2
            c.drawString(xx,(H-y)*72-size*1.02-i*size*1.28,line)
        return size
    def line(self,x1,y1,x2,y2,color=BLUE,width=1.3,arrow=False):
        sh=self.slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT,Inches(x1),Inches(y1),Inches(x2),Inches(y2));sh.line.color.rgb=RGBColor.from_string(color);sh.line.width=Pt(width)
        c=self.deck.pdf;c.setStrokeColor(HexColor('#'+color));c.setLineWidth(width);c.line(x1*72,(H-y1)*72,x2*72,(H-y2)*72)
        if arrow:
            angle=math.atan2(y2-y1,x2-x1);length=.085
            for a in [angle+2.65,angle-2.65]:self.line(x2,y2,x2+length*math.cos(a),y2+length*math.sin(a),color,width)
    def image(self,path,x,y,w,h):
        path=Path(path);im=Image.open(path);iw,ih=im.size;r=min(w/iw,h/ih);ww,hh=iw*r,ih*r;xx=x+(w-ww)/2;yy=y+(h-hh)/2
        self.slide.shapes.add_picture(str(path),Inches(xx),Inches(yy),width=Inches(ww),height=Inches(hh))
        self.deck.pdf.drawImage(str(path),xx*72,(H-yy-hh)*72,ww*72,hh*72,mask='auto')
    def card(self,title,body,x,y,w,h,color=LIGHT,size=18):
        self.rect(x,y,w,h,color,rounded=True);self.text(title,x+.18,y+.14,w-.36,.5,19,TEAL,bold=True)
        self.text(body,x+.18,y+.77,w-.36,h-.91,size,INK)
    def table(self,headers,rows,x=.62,y=1.96,widths=None,row_h=.29,size=11):
        widths=widths or [(12.08)/len(headers)]*len(headers)
        for i,row in enumerate([headers]+rows):
            xx=x
            for j,value in enumerate(row):
                bg=NAVY if i==0 else MINT if str(row[0])=='PatentFlow' else ('F1F5F8' if i%2 else 'FFFFFF')
                self.rect(xx,y+i*row_h,widths[j],row_h,bg,'DEE6EB')
                self.text(str(value),xx+.06,y+i*row_h+.035,widths[j]-.12,row_h-.045,size,'FFFFFF' if i==0 else INK,bold=(i==0),record=False)
                xx+=widths[j]
        self.content.append('| '+' | '.join(map(str,headers))+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+'\n'.join('| '+' | '.join(map(str,r))+' |' for r in rows))
    def note(self,text):self.text(text,.65,6.60,12,.35,11,MUTED)
