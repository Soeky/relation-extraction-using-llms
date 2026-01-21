#!/usr/bin/env python
"""
Generate gold relation JSON files from BioRED BioC JSON files.

For each document, creates a JSON file with format:
{
    "doc_id": "<string>",
    "title": "<string>",
    "body": "<string>",
    "entities": [...],
    "relations": [...]
}
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def load_bioc_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "documents" in data:
        return data
    elif isinstance(data, list):
        return {"documents": data}
    else:
        raise ValueError(f"Unsupported JSON structure in {path}")


def extract_title_and_body(passages: List[Dict[str, Any]]) -> Tuple[str, str]:
    if not passages:
        return "", ""

    passages_sorted = sorted(passages, key=lambda p: p.get("offset", 0))

    title = (passages_sorted[0].get("text") or "").strip()
    body_parts = [
        (p.get("text") or "").strip() for p in passages_sorted[1:] if p.get("text")
    ]
    body = "\n\n".join(part for part in body_parts if part)

    return title, body


def collect_entities(passages: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    entities_by_id: Dict[str, Dict[str, Any]] = {}

    for passage_idx, passage in enumerate(passages):
        passage_offset = passage.get("offset", 0)
        annotations = passage.get("annotations", []) or []

        for ann in annotations:
            infons = ann.get("infons", {}) or {}
            identifier = str(infons.get("identifier", "")).strip()
            ent_type = str(infons.get("type", "")).strip()
            text = (ann.get("text") or "").strip()
            locations = ann.get("locations", []) or []

            if not identifier:
                continue

            if identifier not in entities_by_id:
                entities_by_id[identifier] = {
                    "id": identifier,
                    "type": ent_type,
                    "mentions": [],
                }

            for loc in locations:
                char_offset = loc.get("offset", 0)
                length = loc.get("length", len(text))

                entities_by_id[identifier]["mentions"].append({
                    "text": text,
                    "passage_index": passage_idx,
                    "passage_offset": passage_offset,
                    "char_offset": char_offset,
                    "length": length,
                })

    return entities_by_id


def collect_relations(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    relations_raw = doc.get("relations", []) or []
    relations: List[Dict[str, Any]] = []

    for rel in relations_raw:
        rel_id = str(rel.get("id", "")).strip()
        infons = rel.get("infons", {}) or {}

        entity1 = str(infons.get("entity1", "")).strip()
        entity2 = str(infons.get("entity2", "")).strip()
        rel_type = str(infons.get("type", "")).strip()
        novel_flag = str(infons.get("novel", "")).strip()

        if not entity1 or not entity2 or not rel_type:
            continue

        relations.append({
            "id": rel_id,
            "head_id": entity1,
            "tail_id": entity2,
            "type": rel_type,
            "novel": novel_flag,
        })

    return relations


def process_split(input_path: Path, output_dir: Path) -> int:
    bioc = load_bioc_json(input_path)
    documents = bioc.get("documents", [])

    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    for doc in documents:
        doc_id = str(doc.get("id", "")).strip()
        if not doc_id:
            continue

        passages = doc.get("passages", []) or []
        title, body = extract_title_and_body(passages)
        entities_by_id = collect_entities(passages)
        relations = collect_relations(doc)

        entity_ids = set(entities_by_id.keys())
        relations_filtered = [
            r for r in relations
            if (r["head_id"] in entity_ids and r["tail_id"] in entity_ids)
        ]

        record = {
            "doc_id": doc_id,
            "title": title,
            "body": body,
            "entities": list(entities_by_id.values()),
            "relations": relations_filtered,
        }

        out_path = output_dir / f"{doc_id}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)

        count += 1

    return count


def main():
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    data_dir = project_dir / "data"
    gold_relations_dir = project_dir / "gold_relations"

    splits = [
        ("Train.BioC.JSON", "train"),
        ("Dev.BioC.JSON", "dev"),
        ("Test.BioC.JSON", "test"),
    ]

    print("Generating gold relation files from BioRED...")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {gold_relations_dir}")
    print()

    for input_file, split_name in splits:
        input_path = data_dir / input_file
        output_path = gold_relations_dir / split_name

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
