#!/usr/bin/env python3
import argparse, json, os, re, sys
from pathlib import Path
from typing import Dict, List, Iterable, Set

ROOT = Path(__file__).resolve().parents[1]  # repo root (assumes tools/…)
NEUTRAL_MAP = ROOT / "data" / "neutral_map.json"
BACKUP = ROOT / "data" / "neutral_map.backup.json"

# ---------- Providers ----------

class SynonymProvider:
    def suggest(self, term: str, sense_hint: str = "") -> List[str]:
        raise NotImplementedError

class LocalHeuristicsProvider(SynonymProvider):
    """Offline: simple morphology + tiny static set."""
    _tiny = {
        "spill": ["spilt","spilling","spills","spilled","splashed"],
        "pause": ["pausing","paused","pauses","halt","break"],
        "disconnect": ["disconnected","disconnecting","unplug","unplugged"],
        "shake": ["shook","shaking","jolt","bump","bumped"]
    }
    def suggest(self, term: str, sense_hint: str = "") -> List[str]:
        t = term.lower()
        base = set(self._tiny.get(t, []))
        # naive morphological variants
        if re.search(r"[a-z]e$", t):
            base.update([t+"d", t[:-1]+"ing", t+"s"])
        else:
            base.update([t+"ed", t+"ing", t+"s"])
        return sorted(base)

class LMStudioProvider(SynonymProvider):
    """
    Uses a locally-running LM Studio server (OpenAI-compatible).
    Default endpoint: http://localhost:1234/v1/chat/completions
    No API key required by default.
    """
    def __init__(self,
                 model: str = "lmstudio",
                 base_url: str = "http://localhost:1234",
                 temperature: float = 0.2,
                 max_tokens: int = 256):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)

    def suggest(self, term: str, sense_hint: str = "") -> List[str]:
        import requests
        prompt = (
            "You are expanding a controlled lexicon for a research project. "
            "Return ONLY a JSON array (no preface text) of English synonyms or morphological variants "
            f"for the word '{term}' in the specific sense: '{sense_hint or 'project-defined sense'}'. "
            "Rules: 1) lowercase single tokens preferred; short tokens only; "
            "2) no punctuation or emojis; 3) avoid multi-word phrases unless unavoidable; "
            "4) do NOT invent new senses; 5) 8–15 items."
        )
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "Return only valid JSON. No commentary."},
                {"role": "user", "content": prompt}
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        url = f"{self.base_url}/v1/chat/completions"
        # LM Studio typically doesn't require Authorization; keep header minimal.
        resp = requests.post(url, headers={"Content-Type": "application/json"}, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        print("DEBUG: LM Studio response:", data)  # Add this line
        if "choices" not in data:
            raise RuntimeError(f"LM Studio response missing 'choices': {data}")
        content = data["choices"][0]["message"]["content"]
        # LM Studio models may still wrap JSON with text—be tolerant:
        try:
            arr = json.loads(content)
            if isinstance(arr, dict):
                # if the model returns an object, try common keys
                arr = arr.get("results") or arr.get("synonyms") or []
        except json.JSONDecodeError:
            # last-resort: extract quoted strings
            arr = re.findall(r'"([^"]+)"', content)
        return [a.strip().lower() for a in arr if isinstance(a, str)]

# ---------- Helpers ----------

def load_map(path: Path) -> Dict:
    if not path.exists():
        return {"lex2neutral":{}, "neutral2sdk":{}}
    return json.loads(path.read_text(encoding="utf-8"))

TOKEN_RX = re.compile(r"^[a-z][a-z0-9_-]*$")

def sanitize(cands: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for c in cands:
        tok = c.strip().lower()
        if not tok or len(tok) < 3:
            continue
        if not TOKEN_RX.match(tok):
            continue
        if tok in seen:
            continue
        seen.add(tok)
        out.append(tok)
    return out

def propose(provider: SynonymProvider, term: str, sense_hint: str="") -> List[str]:
    return sanitize(provider.suggest(term, sense_hint))

def interactive_select(term: str, sense: List[str], props: List[str], existing_keys: Set[str]) -> List[str]:
    print(f"\n=== Term: '{term}' → tags {sense}")
    if not props:
        print("No proposals.")
        return []
    print("Proposed additions (type indexes or ranges like '1 3-5', ENTER=none, 'all'=everything; 'q' quits):")
    for i, p in enumerate(props, 1):
        mark = "(exists)" if p in existing_keys else ""
        print(f"  {i:2d}. {p} {mark}")
    sel = input("> ").strip().lower()
    if sel in ("q","quit","exit"):
        sys.exit(1)
    if sel == "" or sel == "none":
        return []
    if sel == "all":
        return [p for p in props if p not in existing_keys]
    picks: Set[int] = set()
    for token in sel.split():
        if "-" in token:
            a,b = token.split("-",1)
            if a.isdigit() and b.isdigit():
                for k in range(int(a), int(b)+1):
                    picks.add(k)
        elif token.isdigit():
            picks.add(int(token))
    chosen = []
    for k in sorted(picks):
        if 1 <= k <= len(props):
            cand = props[k-1]
            if cand not in existing_keys:
                chosen.append(cand)
    return chosen

def merge_lex2neutral(mapobj: Dict, new_pairs: Dict[str, List[str]]):
    valid_neutrals = set(mapobj.get("neutral2sdk", {}).keys())
    for k, tags in new_pairs.items():
        if not set(tags).issubset(valid_neutrals):
            raise ValueError(f"Refusing to add '{k}' → {tags}: contains non-existent neutral tags.")
        existing = mapobj["lex2neutral"].get(k, [])
        merged = sorted(set(existing).union(tags))
        mapobj["lex2neutral"][k] = merged
    return mapobj

# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser(description="Suggest and review lex2neutral expansions (LM Studio local LLM).")
    ap.add_argument("--term", help="Seed lexeme already mapped in lex2neutral; if omitted, batch over all keys.", default=None)
    ap.add_argument("--sense", help="Optional sense hint for the model.", default="")
    ap.add_argument("--provider", choices=["local","lmstudio"], default="lmstudio")
    ap.add_argument("--model", help="LM Studio model name", default="google/gemma-3n-e4b")
    ap.add_argument("--base-url", help="LM Studio base URL", default="http://localhost:1234")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--apply", action="store_true", help="Write changes to neutral_map.json (with backup).")
    ap.add_argument("--data", default=str(NEUTRAL_MAP), help="Path to neutral_map.json")
    args = ap.parse_args()

    path = Path(args.data)
    nm = load_map(path)
    lex2neutral = nm.get("lex2neutral", {})
    if not lex2neutral:
        print("No lex2neutral entries found; seed at least one before expansion.")
        sys.exit(2)

    if args.provider == "local":
        provider = LocalHeuristicsProvider()
    else:
        provider = LMStudioProvider(model=args.model, base_url=args.base_url,
                                    temperature=args.temperature, max_tokens=args.max_tokens)

    seeds = [args.term] if args.term else sorted(lex2neutral.keys())
    existing_keys = set(lex2neutral.keys())
    to_add: Dict[str, List[str]] = {}

    for seed in seeds:
        sense_tags = lex2neutral.get(seed, [])
        props = propose(provider, seed, args.sense or ", ".join(sense_tags))
        chosen = interactive_select(seed, sense_tags, props, existing_keys)
        for c in chosen:
            to_add[c] = sense_tags

    if not to_add:
        print("\nNothing selected. Exiting.")
        return

    print("\nPlanned additions:")
    for k, v in to_add.items():
        print(f"  {k:>16s}  →  {v}")

    if args.apply:
        BACKUP.write_text(json.dumps(nm, indent=2, ensure_ascii=False), encoding="utf-8")
        merged = merge_lex2neutral(nm, to_add)
        path.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n✅ Updated {path} (backup at {BACKUP})")
    else:
        print("\n(dry run) Use --apply to write changes.")

if __name__ == "__main__":
    main()
