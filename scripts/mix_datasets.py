import argparse
import json
import math
from pathlib import Path
from typing import List, Dict


def load_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def write_jsonl(path: Path, records: List[Dict]) -> None:
    path.write_text("\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n", encoding="utf-8")


def normalize_record(record: Dict) -> Dict:
    netlist = record.get("input", {}).get("netlist", [])
    output = record.get("output", {})
    modules = output.get("modules", [])
    replaced = output.get("replaced_netlist") or netlist
    return {
        "input": {"netlist": netlist},
        "output": {
            "modules": modules,
            "replaced_netlist": replaced,
        },
    }


def interleave_records(positives: List[Dict], negatives: List[Dict]) -> List[Dict]:
    if not negatives:
        return positives.copy()

    total = len(positives) + len(negatives)
    neg_total = len(negatives)
    neg_index = 0
    pos_index = 0
    mixed: List[Dict] = []

    for idx in range(total):
        expected_neg = math.floor(((idx + 1) * neg_total) / total)
        if expected_neg > neg_index and neg_index < neg_total:
            mixed.append(negatives[neg_index])
            neg_index += 1
        else:
            if pos_index >= len(positives):
                mixed.append(negatives[neg_index])
                neg_index += 1
            else:
                mixed.append(positives[pos_index])
                pos_index += 1

    return mixed


def main() -> None:
    parser = argparse.ArgumentParser(description="Interleave positive and negative datasets evenly")
    parser.add_argument("--positives", type=Path, required=True, help="Positive samples in JSONL format")
    parser.add_argument("--negatives", type=Path, required=True, help="Negative samples in JSONL format")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL path for the mixed dataset")

    args = parser.parse_args()

    positives = [normalize_record(record) for record in load_jsonl(args.positives)]
    negatives = [normalize_record(record) for record in load_jsonl(args.negatives)]

    mixed = interleave_records(positives, negatives)
    write_jsonl(args.output, mixed)


if __name__ == "__main__":
    main()
