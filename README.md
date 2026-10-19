
# ScholarLens — Grounded RAG Assistant for Scholarships

ScholarLens is a Python-based retrieval-augmented generation (RAG) assistant designed to help students explore scholarship information, understand eligibility requirements, and locate supporting evidence in scholarship documents.

The project combines document ingestion, retrieval, evidence-grounded responses, structured eligibility evaluation, and optional claim verification. It is an academic prototype, not an official scholarship decision system.

## Features

- **Document ingestion:** Extract text from PDF scholarship documents using PyMuPDF, with optional OCR support.
- **Retrieval comparison:** Compare TF-IDF, BM25, dense semantic retrieval, hybrid retrieval, and hybrid retrieval with reranking.
- **Evidence-grounded answers:** Retrieve relevant document passages to support generated responses.
- **Eligibility evaluation:** Support structured scholarship eligibility checks and profile-based evaluation.
- **Claim verification:** Check citations, numerical consistency, and textual entailment when the relevant verifier is enabled.
- **Scheme catalogue:** Read Karnataka SSP entries from a master registry.
- **Evaluation tools:** Evaluate retrieval, profile extraction, eligibility decisions, and groundedness.
- **Automated testing:** Run unit tests locally and through GitHub Actions.

## Architecture

```text
Student question
       |
       v
Profile extraction / intent detection
       |
       v
Document retrieval
(TF-IDF / BM25 / Dense / Hybrid)
       |
       v
Optional reranking
       |
       v
Evidence-grounded answer generation
       |
       v
Citation and claim verification
       |
       v
Answer with supporting evidence
```

The exact path depends on the configured provider, enabled components, and question type.

## Technology Stack

- Python
- Streamlit
- PyMuPDF
- scikit-learn
- BM25 retrieval
- Sentence Transformers
- Google Gemini through `google-genai`
- Optional Anthropic or Ollama integrations
- Pytest
- GitHub Actions

Availability of individual features depends on installed dependencies, model downloads, and configuration.

## Project Structure

```text
scholarlens/
├── .github/
│   └── workflows/
│       └── tests.yml
├── data/
│   ├── gold_set/
│   ├── processed/
│   └── raw/
├── src/
│   └── scholarlens/
│       ├── config.py
│       ├── evaluate.py
│       ├── retrieval.py
│       ├── utils.py
│       └── ...
├── tests/
├── app/
│   └── streamlit_app.py
├── results/
├── .env.example
├── .gitignore
├── pyproject.toml
├── requirements.txt
└── README.md
```

Some directories and files are generated locally or may not be present until the corresponding pipeline stage is run.

## Requirements

- Python 3.10 or later
- Git
- Internet access for dependency installation and model downloads
- An API key for the selected hosted language model, if required
- Sufficient RAM and disk space for the selected embedding and reranking models

## Installation

Clone the repository:

```bash
git clone https://github.com/sonnisahana/Scholarlens.git
cd Scholarlens
```

Create and activate a virtual environment.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

If PowerShell blocks virtual-environment activation, use the Python executable directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Configuration

Create a local `.env` file using `.env.example` as a reference.

Configure the provider and model settings supported by the current implementation. For hosted providers, add the required API key to your local `.env` file.

**Never commit `.env` or expose API keys in source code, screenshots, logs, or public repositories.** The `.env.example` file should contain placeholders only.

Keep provider names and model identifiers consistent with those supported by the installed SDK and project configuration.

## Prepare the Corpus

Place the scholarship documents in the expected raw-data directory and run the ingestion stage:

```powershell
python -m scholarlens.ingest
```

The ingestion pipeline creates processed data, including chunks and sentence-level records used by retrieval and evaluation.

Check the project configuration for the exact input paths, output locations, and document formats supported by the current version.

## Run the Application

Launch the Streamlit application:

```powershell
streamlit run app/streamlit_app.py
```

Open the local URL printed by Streamlit in your browser.

The available features depend on the local corpus, configured language model, installed dependencies, and enabled verification components.

## Run Tests

Run the automated test suite from the project root:

```powershell
python -m pytest -q
```

**Development baseline:** 15 tests passed in the previously recorded local run. Re-run the command after changes to verify the current state.

GitHub Actions is configured to run tests when changes are pushed to the main branch or a pull request targets it.

Passing tests verifies the tested behaviours; it does not establish that every scholarship condition is current or that all generated answers are factually correct.

## Evaluation

ScholarLens includes evaluation utilities for retrieval, extraction, eligibility, and groundedness.

### Retrieval pilot

A preliminary retrieval comparison was conducted using 10 queries for the PM-USP Central Sector Scheme of Scholarship for College and University Students (CSSS).

Run the retrieval evaluation with:

```powershell
python -m scholarlens.evaluate --task retrieval --limit 10
```

### Preliminary results

| Retrieval method | Recall@1 | Recall@3 | Recall@5 | MRR | nDCG@1 | nDCG@3 | nDCG@5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| TF-IDF | 0.600 | 0.900 | 1.000 | 0.753 | 0.600 | 0.776 | 0.815 |
| BM25 | 0.600 | 0.800 | 0.900 | 0.725 | 0.600 | 0.688 | 0.754 |
| Dense | 0.600 | 0.800 | 1.000 | 0.728 | 0.600 | 0.674 | 0.783 |
| Hybrid | 0.700 | 1.000 | 1.000 | 0.817 | 0.700 | 0.824 | 0.848 |
| Hybrid + reranking | 0.800 | 1.000 | 1.000 | 0.900 | 0.800 | 0.926 | 0.926 |

In this pilot, hybrid retrieval with reranking achieved the highest scores across all reported metrics. The results suggest that combining retrieval methods and reranking may improve the ordering of relevant evidence for this query set.

### Metric definitions

- **Recall@k:** Fraction of gold-relevant sentence IDs retrieved within the top k results.
- **MRR (Mean Reciprocal Rank):** Average reciprocal rank of the first relevant result.
- **nDCG@k:** Ranking-quality metric that gives greater weight to relevant results appearing earlier.

### Evaluation limitations

This is a **10-query pilot covering one scholarship scheme**, not a comprehensive benchmark. The gold labels were constructed for initial pipeline evaluation and have not undergone independent annotation or validation.

The scores should therefore be treated as preliminary engineering results, not as evidence of general performance across scholarship documents or student populations. A stronger research evaluation would require more queries across multiple schemes, independently reviewed relevance labels, error analysis, and repeated reproducible experiments.

### Other evaluation tasks

The evaluation module also provides commands for other tasks:

```powershell
python -m scholarlens.evaluate --task extraction
python -m scholarlens.evaluate --task eligibility
python -m scholarlens.evaluate --task eligibility --oracle-profile
```

Check the current implementation and required input files before running these tasks. Retrieval scores alone do not measure eligibility accuracy, answer groundedness, or the correctness of final scholarship advice.

Evaluation outputs are written under `results/` and are excluded from version control by default. The retrieval pilot metrics are documented above so they remain visible in the repository.

## Limitations

- **Document currency:** Scholarship eligibility conditions, deadlines, and award amounts can change. Consult current official notifications.
- **OCR and extraction:** Complex PDF layouts, tables, and multilingual text may be extracted incorrectly.
- **Corpus completeness:** Results depend on the documents ingested into the system.
- **Eligibility rules:** Decisions depend on the completeness and accuracy of the structured rules and user-provided information.
- **Retrieval quality:** Relevant evidence may be missed or ranked incorrectly.
- **Claim verification:** Verification can identify some unsupported claims but cannot guarantee factual correctness.
- **Language support:** Multilingual input does not guarantee equal retrieval or generation accuracy across languages.
- **Model requirements:** Embedding, reranking, and verification models may require downloads, memory, and computational resources.
- **Official decisions:** ScholarLens is an academic prototype and does not replace official scholarship portals or decisions made by scholarship authorities.

## Roadmap

- Expand and independently validate the retrieval gold set across multiple schemes.
- Improve document provenance, metadata, and source currency checks.
- Expand eligibility rules and test borderline cases.
- Evaluate extraction accuracy and groundedness against manually reviewed examples.
- Add regression tests for unsupported and ambiguous eligibility questions.
- Improve multilingual retrieval and document extraction.
- Explore deployment with suitable resource limits and secrets management.

## Responsible Use

Use ScholarLens to explore scholarship information and locate relevant source passages. Before acting on an eligibility result, verify the conditions, deadlines, and application process against the relevant official scholarship notification or portal.

Do not treat generated answers as official decisions or submit sensitive personal information unless the deployment has appropriate privacy and security safeguards.

## License

Add the applicable license before redistributing the project. Unless a license is included, reuse and redistribution permissions may be limited.

---

**Project:** ScholarLens  
**Purpose:** Academic prototype for evidence-grounded scholarship information retrieval and assistance.
