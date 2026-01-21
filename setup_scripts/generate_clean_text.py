#!/usr/bin/env python
"""
Generate clean text files from BioRED BioC JSON files.

For each document, creates a text file with format:
    <title>

    <body>
"""

import json
from pathlib import Path
from typing import Any, Dict, List


def load_bioc_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "documents" in data:
        return data
    elif isinstance(data, list):
        return {"documents": data}
    else:
        raise ValueError(f"Unsupported JSON structure in {path}")


def extract_title_and_body(passages: List[Dict[str, Any]]) -> Dict[str, str]:
    if not passages:
        return {"title": "", "body": ""}

    passages_sorted = sorted(passages, key=lambda p: p.get("offset", 0))

    title = (passages_sorted[0].get("text") or "").strip()
    body_parts = [
        (p.get("text") or "").strip() for p in passages_sorted[1:] if p.get("text")
    ]
    body = "\n\n".join(part for part in body_parts if part)

    return {"title": title, "body": body}


def write_document_text(doc_id: str, title: str, body: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{doc_id}.txt"

    if body:
        text = f"{title}\n\n{body}".strip() + "\n"
    else:
        text = (title or "").strip() + "\n"

    with out_path.open("w", encoding="utf-8") as f:
        f.write(text)


def process_split(input_path: Path, output_dir: Path) -> int:
    bioc = load_bioc_json(input_path)
    documents = bioc.get("documents", [])
    count = 0

    for doc in documents:
        doc_id = str(doc.get("id", "")).strip()
        if not doc_id:
            continue

        passages = doc.get("passages", [])
        title_body = extract_title_and_body(passages)

        write_document_text(
            doc_id=doc_id,
            title=title_body["title"],
            body=title_body["body"],
            output_dir=output_dir,
        )
        count += 1

    return count


def main():
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    data_dir = project_dir / "data"
    clean_text_dir = project_dir / "clean_text"

    splits = [
        ("Train.BioC.JSON", "train"),
        ("Dev.BioC.JSON", "dev"),
        ("Test.BioC.JSON", "test"),
    ]

    print("Generating clean text files from BioRED...")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {clean_text_dir}")
    print()

    for input_file, split_name in splits:
        input_path = data_dir / input_file
        output_path = clean_text_dir / split_name

        if not input_path.exists():
            print(f"Skipping {split_name}: {input_file} not found")
            print(f"  Run download_biored.py first to download the dataset")
            continue

        count = process_split(input_path, output_path)
        print(f"Processed {split_name}: {count} documents -> {output_path}")

    print()
    print("Generation complete!")


if __name__ == "__main__":
    main()
