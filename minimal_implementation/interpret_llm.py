import json, re, requests
from pathlib import Path

# ----------------- Config -----------------
LMSTUDIO_URL = "http://localhost:1234/v1/chat/completions"
# MODEL = "openai/gpt-oss-20b"   # set to your local model name
MODEL = "google/gemma-3n-e4b"   # set to your local model name

DATA_ROOT = Path("minimal_implementation/data")
PASSAGES_PATH = DATA_ROOT / "manual_passages.json"
RELATIONS_PATH = DATA_ROOT / "relations.json"

# ----------------- Data loading -----------------
def load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))

def load_passages():
    return load_json(PASSAGES_PATH)

REL = load_json(RELATIONS_PATH)  # relations graph (token -> {"requires":[...]})


# ----------------- Require-closure helpers -----------------
def requires_of(tok: str) -> list[str]:
    return REL.get(tok, {}).get("requires", []) or []

def expand_with_requires(tokens: list[str]) -> list[str]:
    """Return tokens plus their prerequisites in a topo-respecting order (duplicates removed)."""
    out, seen = [], set()
    def add(w: str):
        if w in seen:
            return
        for r in requires_of(w):
            add(r)
        seen.add(w)
        out.append(w)
    for t in tokens:
        add(t)
    return out

def requires_set(tokens: set[str]) -> set[str]:
    """Req(A) \\ A"""
    expanded = set(expand_with_requires(list(tokens)))
    return expanded - set(tokens)


# ----------------- Passage selection (EVENT–MINIMAL) -----------------
def select_passages_event_minimal(activated_tokens: set[str], all_passages: list[dict], max_quotes: int = 4) -> list[dict]:
    """
    Pick passages tied to activated tokens and *only the prerequisites actually used*.
    Avoid 'backbone' unless one of its tokens is an included prerequisite.
    """
    T = set(expand_with_requires(list(activated_tokens)))  # A ∪ Req(A)
    prereq_only = requires_set(activated_tokens)           # prerequisites used here

    chosen, seen_ids = [], set()

    # 1) Prefer non-backbone passages tied to T
    for p in all_passages:
        if p["id"] in seen_ids or p.get("role") == "backbone":
            continue
        if any(tok in T for tok in p.get("tokens", [])):
            chosen.append({"id": p["id"], "text": p["text"]})
            seen_ids.add(p["id"])
            if len(chosen) >= max_quotes:
                return chosen

    # 2) Add backbone passages only if they explain an actual prerequisite (in prereq_only)
    for p in all_passages:
        if p["id"] in seen_ids or p.get("role") != "backbone":
            continue
        if any(tok in prereq_only for tok in p.get("tokens", [])):
            chosen.append({"id": p["id"], "text": p["text"]})
            seen_ids.add(p["id"])
            if len(chosen) >= max_quotes:
                break

    return chosen


# ----------------- LLM call -----------------
def ask_lmstudio(allowed_tokens: list[str], quotes: list[dict], event_brief: str):
    system = (
        "You are formatting a literal, first-person narrative from a PlayStation (PS1) SDK perspective. "
        "Only use the SDK tokens provided in AllowedTokens. "
        "Do not claim anything not supported by Passages. "
        "Cite claims like [id]. Keep it concise (90–130 words). Use first-person 'I'. "
        "Return ONLY JSON with keys: narrative, citationsUsed, tokensUsed."
    )
    user = {
        "EventBrief": event_brief,
        "AllowedTokens": allowed_tokens,
        "Passages": quotes,
        "NarrativeTask": (
            "Describe the event strictly in SDK terms (AllowedTokens). "
            "Omit notions not present in Passages. Cite each factual claim like [gpu.reset.p47]."
        )
    }
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)}
        ],
        "temperature": 0.0,
        "max_tokens": 400
    }
    r = requests.post(LMSTUDIO_URL, json=payload, timeout=60)
    r.raise_for_status()
    content = r.json()["choices"][0]["message"]["content"]

    # tolerate non‑JSON wrapper
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", content, re.S)
        return json.loads(m.group(0)) if m else {"narrative":"", "citationsUsed":[], "tokensUsed":[]}


# ----------------- Lint -----------------
def lint_output(out, allowed_tokens: set[str], provided_ids: list[str], banned_terms=None):
    ok, issues = True, []

    # Token whitelist (CamelCase-ish heuristic)
    used_caps = set(re.findall(r"\b[A-Z][A-Za-z0-9_]*\b", out.get("narrative", "")))
    for t in used_caps:
        if t not in allowed_tokens and t not in {"I", "PS1"}:
            ok = False; issues.append(f"unallowed token: {t}")

    # Citations must be drawn from provided passages
    cids = set(out.get("citationsUsed", []))
    valid_ids = set(provided_ids)
    if not cids:
        ok = False; issues.append("no citations used")
    for cid in cids:
        if cid not in valid_ids:
            ok = False; issues.append(f"bad citation id: {cid}")

    # Banned terms (UNREP leakage)
    for term in (banned_terms or []):
        if re.search(rf"\b{re.escape(term)}\b", out.get("narrative",""), re.I):
            ok = False; issues.append(f"unrepresentable term present: {term}")

    return ok, issues


# ----------------- Orchestration -----------------
def narrative_from_event(event_text: str, tokens_from_interpreter: list[str], unrep_terms: list[str]):
    """
    tokens_from_interpreter: event-activated tokens (e.g., interpreter['bag_of_api'])
    unrep_terms: interpreter['unrepresentable'] for leakage checks
    """
    passages = load_passages()

    # A) Activated tokens for this event
    activated = set(tokens_from_interpreter)

    # B) Allowed tokens = activated ∪ prerequisites (ordered)
    allowed_tokens = expand_with_requires(list(activated))

    # C) Event-minimal passage selection
    quotes = select_passages_event_minimal(activated_tokens=activated, all_passages=passages, max_quotes=4)
    provided_ids = [q["id"] for q in quotes]

    # D) Ask local LLM
    llm_out = ask_lmstudio(allowed_tokens, quotes, event_text)

    # E) Lint
    ok, issues = lint_output(llm_out, set(allowed_tokens), provided_ids, unrep_terms)
    llm_out["__ok"] = ok
    if not ok:
        llm_out["__issues"] = issues

    return llm_out


# ----------------- Example -----------------
if __name__ == "__main__":
    # Example activated set; in your pipeline, use interpreter['bag_of_api']
    tokens = ["PadGetState", "PAD_STATE_*", "CdGetError", "Cdl*"]  # no longer always the full loop
    unrep = ["coffee", "spill", "liquid"]

    event_text = input("Enter the event description: ")
    result = narrative_from_event(event_text, tokens, unrep)
    print(json.dumps(result, indent=2, ensure_ascii=False))