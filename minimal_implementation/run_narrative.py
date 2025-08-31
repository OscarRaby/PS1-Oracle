import json, re, requests, sys, argparse
from pathlib import Path
from typing import List, Dict, Set, Tuple

# ----------------- Config -----------------
LMSTUDIO_URL = "http://localhost:1234/v1/chat/completions"
# MODEL = "openai/gpt-oss-20b"   # change to your LM Studio model name
MODEL = "google/gemma-3n-e4b"   # change to your LM Studio model name
DATA_ROOT = Path(__file__).resolve().parent / "data"

# ----------------- Load data -----------------
def _load(name: str) -> Dict:
    p = DATA_ROOT / name
    if not p.exists():
        raise FileNotFoundError(f"Missing data file: {p}")
    return json.loads(p.read_text(encoding="utf-8"))

SYMBOLS  = _load("symbols.json")
REL      = _load("relations.json")
NEUT_MAP = _load("neutral_map.json")
PASSAGES = _load("manual_passages.json")

# ----------------- Interpreter -----------------
def _tokenize(text: str) -> List[str]:
    return [t for t in re.split(r"[^a-z0-9_]+", text.lower()) if t]

def interpret_event(text: str) -> Dict:
    tokens = _tokenize(text)

    # NL -> neutral
    neutral = set()
    for t in tokens:
        neutral.update(NEUT_MAP.get("lex2neutral", {}).get(t, []))

    # neutral -> SDK tokens (gate by symbols.json)
    sdk = set()
    for n in neutral:
        for w in NEUT_MAP.get("neutral2sdk", {}).get(n, []):
            if w in SYMBOLS.get("functions", []) or w in SYMBOLS.get("literals", []) or w in SYMBOLS.get("enums", {}):
                sdk.add(w)

    mapped_terms = {t for t in tokens if NEUT_MAP.get("lex2neutral", {}).get(t)}
    unrepresentable = sorted(set(tokens) - mapped_terms)

    return {
        "bag_of_api": sorted(sdk),
        "unrepresentable": unrepresentable
    }

# ----------------- Requires closure -----------------
def requires_of(tok: str) -> List[str]:
    return REL.get(tok, {}).get("requires", []) or []

def expand_with_requires(tokens: List[str]) -> List[str]:
    out, seen = [], set()
    def add(w: str):
        if w in seen: return
        for r in requires_of(w): add(r)
        seen.add(w); out.append(w)
    for t in tokens: add(t)
    return out

def requires_set(tokens: Set[str]) -> Set[str]:
    return set(expand_with_requires(list(tokens))) - set(tokens)

# ----------------- Passage selection -----------------
def select_passages_event_minimal(activated_tokens: Set[str], max_quotes: int = 4) -> List[Dict]:
    T = set(expand_with_requires(list(activated_tokens)))  # A ∪ Req(A)
    prereq_only = requires_set(activated_tokens)           # Req(A) \ A
    chosen, seen_ids = [], set()
    # Non-backbone first
    for p in PASSAGES:
        if p["id"] in seen_ids or p.get("role") == "backbone": continue
        if any(tok in T for tok in p.get("tokens", [])):
            chosen.append({"id": p["id"], "text": p["text"]}); seen_ids.add(p["id"])
            if len(chosen) >= max_quotes: return chosen
    # Backbone only for actual prerequisites
    for p in PASSAGES:
        if p["id"] in seen_ids or p.get("role") != "backbone": continue
        if any(tok in prereq_only for tok in p.get("tokens", [])):
            chosen.append({"id": p["id"], "text": p["text"]}); seen_ids.add(p["id"])
            if len(chosen) >= max_quotes: break
    return chosen

# ----------------- LM Studio call -----------------
def call_lmstudio(payload: Dict, debug: bool=False) -> Tuple[str, Dict]:
    try:
        r = requests.post(LMSTUDIO_URL, json=payload, timeout=90)
        r.raise_for_status()
        j = r.json()
        content = j["choices"][0]["message"]["content"]
        if debug:
            print("\n[DEBUG] Raw LM Studio content:\n", content, "\n", flush=True)
        return content, j
    except Exception as e:
        print(f"[ERROR] LM Studio request failed: {e}", file=sys.stderr, flush=True)
        raise

def ask_lmstudio(allowed_tokens: List[str], quotes: List[Dict], event_brief: str, debug: bool=False) -> Dict:
    system = (
        "You are formatting a literal, first-person narrative from a PlayStation (PS1) SDK perspective. "
        "Only use the SDK tokens provided in AllowedTokens. "
        "Do not claim anything not supported by Passages. "
        "Cite claims like [id]. Keep it concise (90–130 words). Use first-person 'I'. "
        "Return ONLY JSON with keys: narrative, citationsUsed, tokensUsed. "
        "Do NOT cite any passage id except those in the provided Passages."
    )
    user = {
        "EventBrief": event_brief,
        "AllowedTokens": allowed_tokens,
        "Passages": quotes,
        "NarrativeTask": (
            "Describe the event strictly in SDK terms (AllowedTokens). "
            "Omit notions not present in Passages. Cite each factual claim like [pad.state.p12]. "
            "Do NOT invent or cite passage ids that are not in the provided Passages."
        )
    }
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)}
        ],
        "temperature": 0.2,
        "max_tokens": 400
    }
    raw, _ = call_lmstudio(payload, debug=debug)

    # Parse JSON from model
    try:
        out = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.S)
        if not m:
            out = {"narrative":"", "citationsUsed":[], "tokensUsed":[], "__parse_error":"no-json-object-found", "__raw":raw}
        else:
            try:
                out = json.loads(m.group(0))
            except Exception as e:
                out = {"narrative":"", "citationsUsed":[], "tokensUsed":[], "__parse_error":str(e), "__raw":raw}

    # Guardrail: Remove any citations not in provided_ids from output
    provided_ids = {q["id"] for q in quotes}
    if "citationsUsed" in out:
        out["citationsUsed"] = [cid for cid in out["citationsUsed"] if cid in provided_ids]
    # Optionally, also remove from narrative
    if "narrative" in out and provided_ids:
        # Remove bracketed citations not in provided_ids
        def citation_replacer(match):
            cid = match.group(1)
            return f"" if cid not in provided_ids else match.group(0)
        out["narrative"] = re.sub(r"\[([a-zA-Z0-9_.-]+)\]", citation_replacer, out["narrative"])
    return out

# ----------------- Lint -----------------
def lint_output(out: Dict, allowed_tokens: Set[str], provided_ids: List[str], banned_terms: List[str] | None = None):
    ok, issues = True, []
    used_caps = set(re.findall(r"\b[A-Z][A-Za-z0-9_]*\b", out.get("narrative","")))
    for t in used_caps:
        if t not in allowed_tokens and t not in {"I","PS1"}:
            ok = False; issues.append(f"unallowed token: {t}")
    cids = set(out.get("citationsUsed", []))
    valid_ids = set(provided_ids)
    if not cids:
        ok = False; issues.append("no citations used")
    for cid in cids:
        if cid not in valid_ids:
            ok = False; issues.append(f"bad citation id: {cid}")
    for term in (banned_terms or []):
        if re.search(rf"\b{re.escape(term)}\b", out.get("narrative",""), re.I):
            ok = False; issues.append(f"unrepresentable term present: {term}")
    return ok, issues

# ----------------- Orchestrator -----------------
def generate_narrative(event_text: str, debug: bool=False) -> Dict:
    print("[1/5] Interpreting event…", flush=True)
    interp = interpret_event(event_text)
    activated = set(interp["bag_of_api"])
    banned = interp["unrepresentable"]

    print(f"      Activated tokens: {sorted(list(activated))}", flush=True)
    print(f"      Unrepresentable:  {banned}", flush=True)

    # Guardrail: If no SDK tokens are activated or all words are unrepresentable, return fallback narrative
    if not activated or (len(banned) == len(_tokenize(event_text))):
        fallback = {
            "narrative": (
                "No part of the PlayStation SDK vocabulary can be used to describe this event. "
                "The input does not map to any known SDK concepts or tokens."),
            "citationsUsed": [],
            "tokensUsed": [],
            "__ok": True,
            "__activated_tokens": [],
            "__allowed_tokens": [],
            "__provided_ids": [],
            "__fallback": True
        }
        print("[2/5] No SDK tokens activated. Returning fallback narrative.", flush=True)
        return fallback

    print("[2/5] Expanding prerequisites…", flush=True)
    allowed_tokens = expand_with_requires(list(activated))
    print(f"      AllowedTokens: {allowed_tokens}", flush=True)

    print("[3/5] Selecting passages (event-minimal)…", flush=True)
    quotes = select_passages_event_minimal(activated, max_quotes=4)
    provided_ids = [q["id"] for q in quotes]
    print(f"      Passages: {provided_ids}", flush=True)

    print("[4/5] Asking local LLM…", flush=True)
    out = ask_lmstudio(allowed_tokens, quotes, event_text, debug=debug)

    print("[5/5] Linting output…", flush=True)
    ok, issues = lint_output(out, set(allowed_tokens), provided_ids, banned)
    out["__ok"] = ok
    if not ok:
        out["__issues"] = issues
    out["__activated_tokens"] = sorted(list(activated))
    out["__allowed_tokens"] = allowed_tokens
    out["__provided_ids"] = provided_ids
    return out

# ----------------- CLI -----------------
def main():
    ap = argparse.ArgumentParser(description="Generate event-minimal PS1 narrative from user input.")
    ap.add_argument("--debug", action="store_true", help="Print raw LM Studio content and step logs.")
    ap.add_argument("--event", type=str, help="Event text. If omitted, prompt interactively.")
    args = ap.parse_args()

    event_text = args.event or input("Enter the event description: ").strip()
    try:
        result = generate_narrative(event_text, debug=args.debug)
        print("\n=== RESULT JSON ===", flush=True)
        print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
        print("\n=== NARRATIVE ===", flush=True)
        print(result.get("narrative","<no narrative>"), flush=True)
    except Exception as e:
        print(f"[FATAL] {e}", file=sys.stderr, flush=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
