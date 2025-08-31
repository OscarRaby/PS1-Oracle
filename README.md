# PlayStation 1 SDK Event-to-SDK Narrative Toolkit

This project provides a research and tooling framework for mapping natural language descriptions of events to PlayStation 1 SDK (Psy-Q) API concepts. It enables you to interpret user-described events (e.g., "controller disconnected", "game paused") and translate them into relevant PS1 SDK tokens, documentation passages, and technical narratives.

## Features
- **Lexicon Expansion:** Suggest and manage mappings from natural language terms to neutral tags and SDK tokens.
- **Event Interpretation:** Map user event descriptions to SDK tokens using curated mappings and LLMs.
- **Passage Selection:** Select relevant documentation snippets (passages) for SDK tokens.
- **Narrative Generation:** Use an LLM to generate a first-person SDK-style narrative about an event, citing relevant passages.
- **Validation:** Ensure all mappings and references are consistent with the actual SDK API surface (as indexed from header files).

## TODO
- As a POC, the system currently uses hard-coded language to SDK mappings from a minimal subset of functions. Further subsets of the SDK will be included.

## Project Structure

```
minimal_implementation/
    interpret_llm.py         # Main event-to-narrative pipeline
    run_narrative.py         # CLI for running the pipeline
    data/
        headers_index.csv    # Index of all SDK functions/macros (from headers)
        symbols.json         # Curated SDK vocabulary (functions, enums, literals)
        relations.json       # Dependency graph between SDK tokens
        neutral_map.json     # Maps NL terms → neutral tags → SDK tokens
        manual_passages.json # Documentation snippets for SDK tokens
    tools/
        suggest_lex2neutral.py   # Lexicon expansion tool
        validate_sdk_json.py     # Mapping validation tool
        parse_psyq_sdk.py        # Header index generator (support)
```

## How It Works
1. **Describe an event** (e.g., "controller disconnected").
2. The pipeline interprets the event, maps it to SDK tokens, expands with prerequisites, and selects relevant documentation passages.
3. An LLM generates a technical narrative, citing real SDK docs and focusing on aspects like transience, learning, humanity, work, and physicality.
4. The output is validated for correctness and completeness.

## Getting Started

### Prerequisites
- Python 3.8+
- [LM Studio](https://lmstudio.ai/) running locally (for LLM-based features)
- All data files present in `minimal_implementation/data/`

### Installation
1. Clone this repository.
2. Install required Python packages:
   ```cmd
   pip install requests
   ```
3. (Optional) Install and run LM Studio, and download a compatible LLM model (e.g., `google/gemma-3n-e4b`).

### Running the Pipeline

#### Interactive Narrative Generation
Run the main script and enter an event description when prompted:
```cmd
python minimal_implementation/run_narrative.py
```

#### Batch or Programmatic Use
You can import and use `interpret_llm.py` in your own scripts, or call its functions directly.

#### Lexicon Expansion
To suggest and review new mappings from natural language to neutral tags:
```cmd
python minimal_implementation/tools/suggest_lex2neutral.py --term pause --apply
```

#### Validation
To check that all tokens/enums/literals in your JSON mappings exist in the SDK headers:
```cmd
python minimal_implementation/tools/validate_sdk_json.py
```

## Data Files
- **symbols.json:** List of SDK functions, enums, and literals used for mapping.
- **relations.json:** Defines dependencies between SDK tokens.
- **neutral_map.json:** Maps natural language terms to neutral tags and then to SDK tokens.
- **manual_passages.json:** Contains documentation snippets for SDK tokens, used for narrative generation and citation.
- **headers_index.csv:** Index of all function prototypes and macros in the Psy-Q SDK headers (used for validation).

## Customization
- You can expand the vocabulary and mappings by editing the JSON files in `data/` or using the provided tools.
- To add new documentation passages, update `manual_passages.json`.

## Support
For questions or contributions, please open an issue or submit a pull request.

---

**This project bridges the gap between plain English and the technical world of PlayStation programming, making the PS1 SDK more accessible and explorable.**

