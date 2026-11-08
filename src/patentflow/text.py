"""Country-specific tokenization without fitted validation/test vocabulary."""
import re

WORD = re.compile(r"[a-zA-Z][a-zA-Z'-]+|[\u4e00-\u9fff]+")
STOP = set('the a an and or of in on to for with by is are be as this that from at it its into using used said'.split())


def tokenize(text,lang,max_tokens=512):
    if lang=='cn':
        import jieba
        tokens=jieba.lcut(text.lower(),cut_all=False)
        return [t.strip() for t in tokens if WORD.fullmatch(t.strip()) and t.strip() not in STOP][:max_tokens]
    return [w for w in WORD.findall(text.lower()) if w not in STOP][:max_tokens]


def tokenize_row(args):
    row_id,year,title,abstract,lang=args
    return row_id,year,tokenize(title+' '+abstract,lang)
