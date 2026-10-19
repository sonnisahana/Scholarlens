# ScholarLens — Grounded RAG Assistant for Scholarships

ScholarLens is a Python-based research prototype that helps students explore scholarship schemes using natural-language questions. It focuses on Karnataka State Scholarship Portal (SSP) material and selected central scholarship schemes.

The goal is to make scholarship information easier to understand by combining document retrieval, rule-based eligibility checks, language-model explanations, source citations, and claim verification.

> **Disclaimer:** ScholarLens is an informational prototype, not an official scholarship eligibility decision system. Scheme availability, eligibility requirements, income limits, deadlines, and other conditions must be confirmed against the latest official notifications.

## Key Features

- **Natural-language queries:** Ask questions about scholarships and eligibility in English, Kannada, or mixed-language phrasing.
- **Document ingestion:** Extract sentence-level text from scholarship PDFs, with OCR support for scanned documents when configured.
- **Hybrid retrieval:** Retrieve relevant passages using lexical and dense retrieval, with reranking when enabled.
- **Rule-based eligibility:** Apply structured scholarship rules to profile information rather than delegating eligibility decisions entirely to an LLM.
- **Grounded generation:** Generate explanations using retrieved evidence and sentence-level source identifiers.
- **Claim verification:** Check citations, numerical consistency, and textual entailment when the relevant verifier is enabled.
- **Scheme catalogue:** Read Karnataka SSP entries from a master registry.
- **Evaluation tools:** Support retrieval, extraction, eligibility, and groundedness evaluation.

## Architecture

```text
Student question
       |
       v
Profile extraction / intent detection
       |
       v
Retrieval and structured rule evaluation
       |
       v
Evidence-grounded answer generation
       |
       v
Citation and claim verification
       |
       v
Answer with supporting sources
```

The rule engine evaluates structured eligibility conditions. Retrieval supplies relevant document passages, and the LLM explains the evidence. The verifier checks claims against cited text; it does not guarantee that the underlying documents or rules are current or correct.

A source identifier may look like:

```text
SCHEME-ID:p4:s12
```

This represents a scheme or document identifier, page 4, sentence 12. The exact identifier format depends on the ingested document.

## Technology Stack

- Python 3.10+
- Streamlit
- PyMuPDF and Tesseract OCR
- BM25 and dense retrieval
- Sentence Transformers and scikit-learn
- Gemini API through `google-genai`
- Optional Anthropic API or local Ollama integration
- Pytest

## Project Structure

```text
scholarlens/
├── app/
│   └── streamlit_app.py
├── data/
│   ├── gold_set/
│   ├── rules/
│   ├── raw_pdfs/
│   ├── processed/
│   ├── schemes_registry.csv
│   └── schemes_registry_master.csv
├── src/
│   └── scholarlens/
│       ├── config.py
│       ├── ingest.py
│       ├── retrieval.py
│       ├── llm.py
│       ├── profile.py
│       ├── rules.py
│       ├── build_rules.py
│       ├── generate.py
│       ├── verify.py
│       ├── pipeline.py
│       └── evaluate.py
├── tests/
├── .env.example
├── .gitignore
├── pyproject.toml
├── requirements.txt
└── README.md
```

Generated processed data, local caches, results, virtual environments, and source PDF files are excluded from version control by default. You may need to obtain the appropriate source documents and regenerate processed files locally.

## 1. Prerequisites

Install:

- Python 3.10 or later
- Git
- An internet connection for package installation, model downloads, and API use
- An API key for the selected hosted LLM provider, or a configured local Ollama model

For scanned PDFs, install the Tesseract OCR executable separately and ensure it is accessible to the application. Kannada OCR additionally requires the appropriate Kannada language data.

## 2. Installation

Run these commands from the project root.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

### macOS or Linux

```bash
python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

## 3. Configure the LLM

Create a local environment file from the example:

```powershell
Copy-Item .env.example .env
```

Open `.env` and configure the provider and model.

Example for Gemini:

```dotenv
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_actual_api_key
LLM_MODEL=gemini-3.5-flash
```

Use a model identifier supported by your API account and the installed SDK.

Other supported configurations include Anthropic and Ollama, subject to the application's implementation and the dependencies installed for the selected provider.

**Security:** Never commit `.env`, API keys, tokens, or other credentials. `.env.example` should contain placeholders only.

## 4. Prepare the Data

The project uses two registries:

- `data/schemes_registry.csv`: the source registry used by the ingestion pipeline.
- `data/schemes_registry_master.csv`: the master catalogue used by catalogue queries.

Review registry entries and confirm the source, scheme name, issuing authority, applicable academic year, and official notification before relying on them.

Obtain official scholarship guidelines from the relevant government portals or department websites. Check document permissions and licensing before redistributing PDFs.

### Ingest documents

Place the required source PDFs under `data/raw_pdfs/`, following the filenames expected by the registry, then run:

```powershell
python -m scholarlens.ingest
```

The pipeline generates processed sentence-level and chunk-level data under `data/processed/`.

Inspect the extracted text carefully, particularly for scanned documents, multilingual tables, multi-column layouts, and OCR errors. Poor extraction can lead to incorrect retrieval and answers.

### Build structured rules

Run:

```powershell
python -m scholarlens.build_rules
```

Review generated rules against the original official documents. Correct inaccurate values, eligibility conditions, exclusions, and source references before marking a rule as verified.

Do not treat automatically generated rules as ground truth until they have been independently checked.

## 5. Run ScholarLens

### Command-line query

```powershell
python -m scholarlens.pipeline "What scholarships are available for SC students in Karnataka?"
```

Another example:

```powershell
python -m scholarlens.pipeline "What is the income limit for SC scholarships?"
```

An answer may be unavailable when the indexed evidence does not support the requested information. This is preferable to inventing an unsupported threshold.

### Streamlit interface

```powershell
streamlit run app/streamlit_app.py
```

Streamlit will display a local URL in the terminal. Open it in your browser to interact with the application.

The first run may download embedding, reranking, or verification models, depending on the configuration. Model downloads require internet access and sufficient disk space and memory.

## 6. Run Tests

Run the test suite:

```powershell
python -m pytest -q
```

The current baseline is **15 passing tests** in the local development environment. Test coverage and results may change as the project evolves.

Tests passing does not establish that every scholarship entry is current, that OCR is error-free, or that every answer is factually correct.

## 7. Evaluation

The repository includes evaluation utilities and example gold-set files.

Example commands:

```powershell
python -m scholarlens.evaluate --task retrieval
python -m scholarlens.evaluate --task extraction
python -m scholarlens.evaluate --task eligibility
python -m scholarlens.evaluate --task eligibility --oracle-profile
python -m scholarlens.evaluate --task groundedness
python -m scholarlens.evaluate --task all --limit 10
```

Confirm the available task names and required data against the current implementation before running a full evaluation.

For a meaningful evaluation:

1. Build a representative set of student profiles and information-seeking questions.
2. Independently label expected eligibility outcomes using official documents.
3. Record relevant source sentence identifiers for retrieval queries.
4. Include borderline income values, missing information, conflicting conditions, out-of-scope queries, and Kannada or mixed-language inputs.
5. Compare retrieval, extraction, eligibility, and groundedness performance.
6. Manually audit incorrect answers and verifier decisions.
7. Report the dataset, annotation process, metrics, and limitations transparently.

Keep generated evaluation outputs under `results/`. They are excluded from version control by default.

## 8. Current Limitations

- **Document currency:** Scholarship conditions can change between academic years. Current validity must be confirmed from official notifications.
- **OCR and table extraction:** Complex layouts and mixed Kannada-English documents may be extracted incorrectly.
- **Registry completeness:** The master registry is a catalogue source, not proof that every listed scheme is active or applicable.
- **Rule quality:** Eligibility decisions depend on the accuracy and completeness of structured rules.
- **Retrieval and verification:** Citation and entailment checks reduce some unsupported claims but cannot guarantee correctness.
- **Model and hardware requirements:** Dense retrieval, reranking, and NLI verification may require model downloads and substantial memory.
- **Language support:** Multilingual input does not guarantee equally accurate retrieval or generation across languages.
- **Official decisions:** The application cannot replace the relevant scholarship authority's rules, verification, or final decision.

## Roadmap

- Improve scholarship metadata and document provenance.
- Expand and independently validate the eligibility rule set.
- Add regression tests for category filtering and unsupported eligibility questions.
- Automate tests using GitHub Actions.
- Package the application with Docker.
- Evaluate retrieval and answer grounding against a manually validated gold set.
- Deploy a demo with appropriate secrets management and resource limits.

## Responsible Use

Use official government notifications as the final authority for scholarship requirements. Do not enter unnecessary personal or sensitive information. Verify all consequential eligibility decisions with the relevant scholarship portal or department.

## License

No license has been specified yet. Until a license is added, reuse and redistribution are not automatically permitted. Confirm the licensing of any source documents and third-party assets before publishing them.
