import json
from pathlib import Path


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_corpus():
    """Returns (chunks, sid2sent). Run `python -m scholarlens.ingest` first."""
    from . import config
    chunks = read_jsonl(config.CHUNKS)
    if not chunks:
        raise FileNotFoundError("No chunks found. Run `python -m scholarlens.ingest` first.")
    sents = read_jsonl(config.SENTENCES)
    return chunks, {s["sid"]: s for s in sents}
