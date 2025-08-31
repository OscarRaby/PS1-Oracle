#!/usr/bin/env python3
import json, sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sym = json.loads((root/"data/symbols.json").read_text())
mp  = json.loads((root/"data/neutral_map.json").read_text())

ok = True
# A) neutral2sdk tokens must exist in symbols.json
allowed = set(sym["functions"]) | set(sym.get("literals", [])) | set(sym.get("enums", {}).keys())
for k, vals in mp.get("neutral2sdk", {}).items():
    for v in vals:
        if v not in allowed:
            print(f"[ERR] neutral2sdk[{k}] -> '{v}' not in symbols.json")
            ok = False

# B) lex2neutral categories must exist in neutral2sdk
cats = set(mp.get("neutral2sdk", {}).keys())
for term, tags in mp.get("lex2neutral", {}).items():
    missing = set(tags) - cats
    if missing:
        print(f"[ERR] lex2neutral['{term}'] has unknown categories: {sorted(missing)}")
        ok = False

sys.exit(0 if ok else 1)