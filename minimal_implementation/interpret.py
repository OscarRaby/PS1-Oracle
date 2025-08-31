import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

def load_json(name: str):
    with open(DATA / name, "r", encoding="utf-8") as f:
        return json.load(f)

SYMBOLS = load_json("symbols.json")
REL: Dict[str, Dict[str, List[str]]] = load_json("relations.json")
try:
    MAP = load_json("neutral_map.json")
except FileNotFoundError:
    MAP = {"lex2neutral": {}, "neutral2sdk": {}}

def tokenize(text: str) -> List[str]:
    return [t for t in re.split(r"[^a-z0-9_]+", text.lower()) if t]

def to_neutral(tokens: List[str]) -> Set[str]:
    neutral = set()
    for t in tokens:
        for n in MAP.get("lex2neutral", {}).get(t, []):
            neutral.add(n)
    return neutral

def neutral_to_sdk(neutral: Set[str]) -> Set[str]:
    sdk = set()
    for n in neutral:
        for w in MAP.get("neutral2sdk", {}).get(n, []):
            if w in SYMBOLS.get("functions", []) or w in SYMBOLS.get("literals", []) or w in SYMBOLS.get("enums", {}):
                sdk.add(w)
    return sdk

def topo_require_chain(targets: List[str]) -> List[str]:
    """Produce a minimal acyclic order that respects 'requires' edges in REL."""
    seq: List[str] = []
    seen: Set[str] = set()

    def add_with_requires(w: str):
        if w in seen:
            return
        for r in REL.get(w, {}).get("requires", []):
            add_with_requires(r)
        if w not in seen:
            seq.append(w)
            seen.add(w)

    for w in targets:
        add_with_requires(w)
    return seq

def build_state_tableau(sdk_words: Set[str]) -> Tuple[List[str], List[str]]:
    observable, callable_ = [], []
    for w in sorted(sdk_words):
        if w in REL and REL[w].get("yields"):
            # Show the domain symbol (e.g., PAD_STATE_* or Cdl*)
            domain = REL[w]["yields"][0]
            observable.append(f"{w} → {domain}")
        elif w in SYMBOLS.get("functions", []) or w in SYMBOLS.get("literals", []):
            callable_.append(w)
        elif w in SYMBOLS.get("enums", {}):
            # enums appear implicitly via observable rows; no extra row here
            pass
    return observable, callable_

def interpret_event(text: str) -> Dict:
    tokens = tokenize(text)
    neutral = to_neutral(tokens)
    sdk_words = neutral_to_sdk(neutral)

    # A tiny "desired path" that fits the slice, if available in sdk_words
    desired = ["PadInit", "PadGetState", "PAD_STATE_*", "VSync(0)"]
    # Keep desired items only if present/representable in current set or enums
    desired_present = []
    for w in desired:
        if w == "PAD_STATE_*":
            if "PAD_STATE_*" in SYMBOLS.get("enums", {}):
                desired_present.append(w)
        elif (w in sdk_words) or (w in SYMBOLS.get("literals", [])):
            desired_present.append(w)

    api_sentence = topo_require_chain(desired_present)

    observable, callable_ = build_state_tableau(sdk_words)
    bag = sorted(list(sdk_words))

    # Unrepresentable: tokens that didn't map to any neutral tag
    mapped_any = set()
    for t in tokens:
        if MAP.get("lex2neutral", {}).get(t):
            mapped_any.add(t)
    unrep = sorted([t for t in set(tokens) if t not in mapped_any])

    return {
        "api_sentence": api_sentence,
        "state_tableau": {
            "observable": observable,
            "callable": callable_
        },
        "bag_of_api": bag,
        "unrepresentable": unrep
    }

if __name__ == "__main__":
    # q = " ".join(sys.argv[1:]) or "a cup of coffee is about to spill on you"
    q = " ".join(sys.argv[1:]) or "a PadInit"
    result = interpret_event(q)
    print(json.dumps(result, indent=2, ensure_ascii=False))
