"""JSONL prompt storage: read/write prompt entries."""

import json
from pathlib import Path
from typing import Iterator


def save_prompts(prompts: Iterator[dict], path: Path) -> int:
    """Write prompts to a JSONL file. Returns count written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(path, "w") as f:
        for entry in prompts:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            count += 1
    return count


def load_prompts(path: Path) -> list[dict]:
    """Read all prompts from a JSONL file."""
    prompts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line))
    return prompts


def get_pending_prompts(path: Path, completed_ids: set[str]) -> list[dict]:
    """Load prompts and filter out already-completed ones."""
    return [p for p in load_prompts(path) if p["prompt_id"] not in completed_ids]
