"""Input normalization. Deliberately does not trust legacy remapped IDs."""
import ast
import csv
import hashlib
import json
import re
import unicodedata
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
IPC_RE = re.compile(r"(?<![A-Z0-9])([A-H]\s*\d{2}\s*[A-Z])", re.I)


def normalize_name(value):
    value = unicodedata.normalize("NFKC", str(value)).strip()
    return re.sub(r"\s+", " ", value).casefold().strip(' "\'')


def applicants(value):
    """Parse IN's Python list, then explicit semicolon separators, never commas.

    Commas are legal company name characters (e.g. MAGIC LEAP, INC).
    Names are not fuzzily merged: false merges distort collaboration statistics.
    """
    raw = str(value).strip()
    values = [raw]
    if raw.startswith("[") and raw.endswith("]"):
        try:
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, (list, tuple)):
                values = [str(x) for x in parsed]
        except (ValueError, SyntaxError):
            pass
    return sorted({normalize_name(x) for v in values for x in re.split(r"[;；]", v)
                   if normalize_name(x) not in ("", "nan", "none", "null")})


def subclasses(value):
    return sorted({re.sub(r"\s+", "", x).upper() for x in IPC_RE.findall(str(value))})


def dates(values):
    """Explicit formats support the installed pandas 1.5, and reject bad dates."""
    s = values.astype(str).str.strip()
    compact = s.str.match(r"^\d{8}$")
    slash = s.str.match(r"^\d{1,2}/\d{1,2}/\d{4}$")
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    out.loc[compact] = pd.to_datetime(s[compact], format="%Y%m%d", errors="coerce")
    out.loc[slash] = pd.to_datetime(s[slash], format="%d/%m/%Y", errors="coerce")
    iso = ~compact & ~slash
    out.loc[iso] = pd.to_datetime(s[iso].str.slice(0, 10), format="%Y-%m-%d", errors="coerce")
    return out


def signature(row, lang):
    # Exact values, not fuzzy matches, prevent accidental patent identity merges.
    fields = [str(row.get("publicationDate", "")), str(row.get("summary", ""))]
    if lang == "cn":
        fields.append(str(row.get("title", "")))
    return hashlib.sha256(json.dumps(fields, ensure_ascii=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    tmp.replace(path)


def taxonomy_paths(lang):
    root = ROOT / "datasets" / ("patent_" + lang)
    paths = []
    for name in ["IPCSection", "IPCClass", "IPCSubClass"]:
        candidates = [root / (name + suffix + ".csv") for suffix in ["", "_US"]]
        found = [p for p in candidates if p.exists()]
        if len(found) != 1:
            raise ValueError("Expected exactly one authoritative taxonomy file for %s: %s" % (name, found))
        paths.append(found[0])
    return paths


def load_taxonomy(lang):
    result = {}
    for level, path in enumerate(taxonomy_paths(lang), 1):
        with path.open(encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                if "Class" in row:
                    code, description = row["Class"].strip(), row["Name"].strip()
                else:
                    parts = row["Class Name"].split(" ", 1)
                    code, description = parts[0], parts[1].strip(' "') if len(parts) > 1 else ""
                if code in result:
                    raise ValueError("Duplicate taxonomy code: " + code)
                result[code] = {"code": code, "level": level, "description": description,
                                "parent": code[:3] if level == 3 else code[:1] if level == 2 else None}
    for entry in result.values():
        if entry["parent"] and entry["parent"] not in result:
            raise ValueError("Missing taxonomy parent for " + entry["code"])
    return result
