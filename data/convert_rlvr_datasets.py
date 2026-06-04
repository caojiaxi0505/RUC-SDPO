from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import requests

HF_ROWS_URL = "https://datasets-server.huggingface.co/rows"

DATASET_SPECS = {
    "openr1_math": ("open-r1/OpenR1-Math-220k", "all", "train"),
    "taco": ("DONG19/TACO", "ALL", "train"),
    "scienceqa": ("derek-thomas/ScienceQA", "default", "train"),
}

HF_COLUMNS = {
    "openr1_math": ["problem", "answer", "problem_type", "question_type", "source", "uuid"],
    "taco": ["question", "starter_code", "input_output", "difficulty", "source", "name", "time_limit"],
    "scienceqa": [
        "image",
        "question",
        "choices",
        "answer",
        "hint",
        "subject",
        "topic",
        "category",
        "grade",
    ],
}

DEFAULT_CODE_TEST_LIMIT = -1  # <= 0 means keep all TACO tests.
DEFAULT_TIMEOUT = 1.0


def json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def write_jsonl_row(handle, row: dict[str, Any]) -> None:
    handle.write(json_dumps(row) + "\n")


def stable_is_val(source_id: str, val_ratio: float, seed: int) -> bool:
    """Deterministic hash split; stable across streaming/backends if source_id is stable."""
    if val_ratio <= 0:
        return False
    key = f"{seed}:{source_id}".encode("utf-8")
    value = int.from_bytes(hashlib.md5(key).digest()[:8], "big") / float(1 << 64)
    return value < val_ratio


def make_verl_row(
    *,
    data_source: str,
    prompt: str,
    ability: str,
    ground_truth: str,
    source_dataset: str,
    source_id: str,
    split: str,
    extra_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    info = {
        "split": split,
        "index": source_id,
        "source_dataset": source_dataset,
        "source_id": source_id,
    }
    if extra_info:
        info.update(extra_info)
    return {
        "data_source": data_source,
        "prompt": [{"role": "user", "content": prompt}],
        "ability": ability,
        "reward_model": {"style": "rule", "ground_truth": ground_truth},
        "extra_info": info,
    }


# ----------------------------- dataset loading -----------------------------


def request_json(
    session: requests.Session,
    *,
    dataset: str,
    config: str,
    split: str,
    offset: int,
    length: int,
    retries: int = 12,
) -> dict[str, Any]:
    params = {"dataset": dataset, "config": config, "split": split, "offset": offset, "length": length}
    for attempt in range(retries):
        try:
            response = session.get(HF_ROWS_URL, params=params, timeout=60)
            if response.status_code == 200:
                return response.json()
            message = f"HTTP {response.status_code}: {response.text[:300]}"
            sleep_s = min(180, 20 * (attempt + 1)) if response.status_code == 429 else min(60, 2**attempt)
        except Exception as exc:
            message = repr(exc)
            sleep_s = min(60, 2**attempt)

        if attempt == retries - 1:
            raise RuntimeError(f"Failed to fetch {dataset}/{config}/{split} offset={offset}: {message}")
        print(f"retry fetch {dataset}/{config}/{split} offset={offset}: {message}; sleep={sleep_s}s")
        time.sleep(sleep_s)

    raise AssertionError("unreachable")


def iter_rows_api(
    *,
    dataset: str,
    config: str,
    split: str,
    page_size: int,
    max_samples: int | None,
    request_sleep: float,
) -> Iterable[dict[str, Any]]:
    session = requests.Session()
    first_page = request_json(session, dataset=dataset, config=config, split=split, offset=0, length=page_size)
    total = int(first_page.get("num_rows_total", 0))
    if max_samples is not None:
        total = min(total, max_samples)

    yielded = 0
    for page in [first_page]:
        for wrapped in page.get("rows", []):
            if yielded >= total:
                return
            yielded += 1
            yield wrapped

    for offset in range(page_size, total, page_size):
        if request_sleep > 0:
            time.sleep(request_sleep)
        page = request_json(
            session,
            dataset=dataset,
            config=config,
            split=split,
            offset=offset,
            length=min(page_size, total - offset),
        )
        for wrapped in page.get("rows", []):
            if yielded >= total:
                return
            yielded += 1
            yield wrapped


def iter_hf_datasets(
    *,
    dataset_name: str,
    dataset: str,
    config: str,
    split: str,
    max_samples: int | None,
    decode_images: bool,
) -> Iterable[dict[str, Any]]:
    from datasets import Image, load_dataset

    ds = load_dataset(dataset, config, split=split, streaming=True)

    if not decode_images and "image" in (getattr(ds, "column_names", None) or []):
        ds = ds.cast_column("image", Image(decode=False))

    columns = HF_COLUMNS.get(dataset_name)
    if columns and hasattr(ds, "select_columns"):
        ds = ds.select_columns(columns)

    for idx, row in enumerate(ds):
        if max_samples is not None and idx >= max_samples:
            return
        yield {"row_idx": idx, "row": row}


def iter_dataset_rows(
    *,
    dataset_name: str,
    dataset: str,
    config: str,
    split: str,
    page_size: int,
    max_samples: int | None,
    request_sleep: float,
    backend: str,
) -> Iterable[dict[str, Any]]:
    if backend not in {"auto", "hf", "rows"}:
        raise ValueError(f"Unsupported backend: {backend}")

    if backend in {"auto", "hf"}:
        try:
            yield from iter_hf_datasets(
                dataset_name=dataset_name,
                dataset=dataset,
                config=config,
                split=split,
                max_samples=max_samples,
                decode_images=dataset_name != "scienceqa",
            )
            return
        except Exception as exc:
            if backend == "hf":
                raise
            print(f"{dataset_name}: hf backend unavailable ({exc}); falling back to rows API")

    yield from iter_rows_api(
        dataset=dataset,
        config=config,
        split=split,
        page_size=page_size,
        max_samples=max_samples,
        request_sleep=request_sleep,
    )


# ----------------------------- shared helpers ------------------------------


def truncated_columns(wrapped: dict[str, Any]) -> set[str]:
    cells = wrapped.get("truncated_cells") or []
    out: set[str] = set()
    for cell in cells:
        if isinstance(cell, dict):
            name = cell.get("column_name") or cell.get("column")
            if name:
                out.add(str(name))
        elif isinstance(cell, str):
            out.add(cell)
    return out


def parse_float(value: Any, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, int | float):
        return float(value)
    text = str(value).strip()
    if not text:
        return default
    try:
        return float(text.split()[0])
    except Exception:
        return default


def option_letter(index: int) -> str:
    if not 0 <= index < 26:
        raise ValueError("answer_index_out_of_range")
    return chr(ord("A") + index)


# ----------------------------- prompt builders -----------------------------


def make_math_prompt(problem: str) -> str:
    return "\n".join(
        [
            "Solve the following math problem.",
            "Show your reasoning clearly.",
            "Put only the final answer in the form \\boxed{...}.",
            "",
            "Problem:",
            problem.strip(),
        ]
    )


def parse_taco_signature(starter_code: str) -> str | None:
    if "def " not in starter_code:
        return None
    try:
        return "def " + starter_code.split("def ", 1)[1].split("Input\n", 1)[0].strip()
    except Exception:
        return None


def make_taco_prompt(question: str, *, test_type: str, fn_name: str, starter_code: str) -> str:
    parts = [
        "Write a correct Python solution for the following programming problem.",
        "Return only one Python code block in the format ```python ... ```.",
        "Do not include explanations outside the code block.",
        "",
        "Problem:",
        question.strip(),
    ]

    signature = parse_taco_signature(starter_code)
    if test_type == "functional":
        parts.extend(["", "This is a function-call problem, not a stdin/stdout problem."])
        if signature and (not fn_name or f"def {fn_name}" in signature):
            parts.extend(
                [
                    "Your solution must implement the following function signature:",
                    f"```python\n{signature}\n```",
                    "Do not change the function name or signature.",
                ]
            )
        else:
            parts.append(f"Your solution must define a function named `{fn_name}`.")
        parts.append("Do not read from stdin or print the final answer unless the problem explicitly requires it.")
    else:
        parts.extend(
            [
                "",
                "This is a stdin/stdout problem.",
                "Write a complete Python program that reads from standard input and writes to standard output.",
            ]
        )

    return "\n".join(parts)


def make_scienceqa_prompt(row: dict[str, Any]) -> str:
    choices = row.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("missing_choices")

    lines = [
        "Answer the following science multiple-choice question.",
        "You may reason briefly, but the final answer must be exactly in this format: <answer>X</answer>.",
        "X must be one of the option letters.",
        "",
    ]

    hint = str(row.get("hint", "") or "").strip()
    if hint:
        lines.extend(["Context:", hint, ""])

    lines.extend(["Question:", str(row.get("question", "")).strip(), "", "Choices:"])
    for idx, choice in enumerate(choices):
        lines.append(f"{option_letter(idx)}. {choice}")
    return "\n".join(lines)


# ----------------------------- converters ----------------------------------


def convert_openr1_math_row(wrapped: dict[str, Any], split: str) -> dict[str, Any]:
    row = wrapped["row"]
    problem = str(row.get("problem", "")).strip()
    answer = str(row.get("answer", "")).strip()
    if not problem:
        raise ValueError("missing_problem")
    if not answer:
        raise ValueError("missing_answer")

    source_id = f"openr1-math-{row.get('uuid') or wrapped.get('row_idx')}"
    return make_verl_row(
        data_source="openr1_math",
        prompt=make_math_prompt(problem),
        ability="math",
        ground_truth=answer,
        source_dataset="openr1_math",
        source_id=source_id,
        split=split,
        extra_info={
            "problem_type": row.get("problem_type", ""),
            "question_type": row.get("question_type", ""),
            "source": row.get("source", ""),
        },
    )


def build_taco_tests(input_output: str, *, time_limit: Any, max_code_tests: int) -> tuple[str, dict[str, Any]]:
    tests = json.loads(input_output)
    if not isinstance(tests, dict):
        raise ValueError("input_output_not_dict")

    inputs = tests.get("inputs")
    outputs = tests.get("outputs")
    if not isinstance(inputs, list) or not isinstance(outputs, list):
        raise ValueError("missing_inputs_or_outputs")
    if not inputs or not outputs:
        raise ValueError("empty_inputs_or_outputs")
    if len(inputs) != len(outputs):
        raise ValueError("input_output_length_mismatch")

    original_count = len(inputs)
    if max_code_tests > 0 and original_count > max_code_tests:
        inputs = inputs[:max_code_tests]
        outputs = outputs[:max_code_tests]

    fn_name = str(tests.get("fn_name", "") or "").strip()
    test_type = "functional" if fn_name else "stdin"
    timeout = parse_float(time_limit, DEFAULT_TIMEOUT)
    out_tests = {
        "inputs": inputs,
        "outputs": outputs,
        "testtype": test_type,
        "fn_name": fn_name,
        "time_limit": timeout,
    }
    info = {
        "testtype": test_type,
        "fn_name": fn_name,
        "time_limit": timeout,
        "num_tests_original": original_count,
        "num_tests_kept": len(inputs),
        "test_limit_applied": int(original_count != len(inputs)),
    }
    return json_dumps(out_tests), info


def convert_taco_row(wrapped: dict[str, Any], split: str, max_code_tests: int) -> dict[str, Any]:
    if {"question", "input_output"} & truncated_columns(wrapped):
        raise ValueError("truncated_required_cell")

    row = wrapped["row"]
    question = str(row.get("question", "")).strip()
    if not question:
        raise ValueError("missing_question")

    tests_json, test_info = build_taco_tests(
        str(row.get("input_output", "")),
        time_limit=row.get("time_limit"),
        max_code_tests=max_code_tests,
    )
    starter_code = str(row.get("starter_code", "") or "")
    source_id = f"taco-{wrapped.get('row_idx')}"

    return make_verl_row(
        data_source="taco",
        prompt=make_taco_prompt(
            question,
            test_type=test_info["testtype"],
            fn_name=test_info["fn_name"],
            starter_code=starter_code,
        ),
        ability="code",
        ground_truth=tests_json,
        source_dataset="taco",
        source_id=source_id,
        split=split,
        extra_info={
            "difficulty": row.get("difficulty", ""),
            "source": row.get("source", ""),
            "name": row.get("name", ""),
            **test_info,
        },
    )


def convert_scienceqa_row(wrapped: dict[str, Any], split: str, *, include_images: bool) -> dict[str, Any]:
    row = wrapped["row"]
    has_image = bool(row.get("image"))
    if has_image and not include_images:
        raise ValueError("image_sample")

    question = str(row.get("question", "")).strip()
    if not question:
        raise ValueError("missing_question")

    choices = row.get("choices")
    answer_idx = int(row.get("answer"))
    if not isinstance(choices, list) or not 0 <= answer_idx < len(choices):
        raise ValueError("bad_answer_or_choices")

    source_id = f"scienceqa-{wrapped.get('row_idx')}"
    return make_verl_row(
        data_source="scienceqa",
        prompt=make_scienceqa_prompt(row),
        ability="mcq",
        ground_truth=option_letter(answer_idx),
        source_dataset="scienceqa",
        source_id=source_id,
        split=split,
        extra_info={
            "subject": row.get("subject", ""),
            "topic": row.get("topic", ""),
            "category": row.get("category", ""),
            "grade": row.get("grade", ""),
            "num_choices": len(choices),
            "has_hint": int(bool(str(row.get("hint", "") or "").strip())),
            "has_image": int(has_image),
        },
    )


def convert_row(dataset_name: str, wrapped: dict[str, Any], split: str, args: argparse.Namespace) -> dict[str, Any]:
    if dataset_name == "openr1_math":
        return convert_openr1_math_row(wrapped, split)
    if dataset_name == "taco":
        return convert_taco_row(wrapped, split, max_code_tests=args.max_code_tests)
    if dataset_name == "scienceqa":
        return convert_scienceqa_row(wrapped, split, include_images=args.scienceqa_include_images)
    raise ValueError(f"unknown_dataset:{dataset_name}")


# ----------------------------- driver --------------------------------------


def add_prompt_stats(stats: Counter, prompt_lengths: list[int]) -> None:
    if not prompt_lengths:
        return
    lengths = sorted(prompt_lengths)
    stats["lint/prompt_chars_min"] = lengths[0]
    stats["lint/prompt_chars_p50"] = lengths[len(lengths) // 2]
    stats["lint/prompt_chars_p95"] = lengths[int(0.95 * (len(lengths) - 1))]
    stats["lint/prompt_chars_max"] = lengths[-1]


def convert_dataset(dataset_name: str, args: argparse.Namespace) -> None:
    hf_dataset, config, hf_split = DATASET_SPECS[dataset_name]
    out_dir = args.base_dir / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / "train.jsonl"
    test_path = out_dir / "test.jsonl"
    meta_path = out_dir / "conversion_meta.json"

    stats: Counter = Counter()
    prompt_lengths: list[int] = []
    seen_ids: set[str] = set()

    with train_path.open("w", encoding="utf-8") as train_f, test_path.open("w", encoding="utf-8") as test_f:
        for wrapped in iter_dataset_rows(
            dataset_name=dataset_name,
            dataset=hf_dataset,
            config=config,
            split=hf_split,
            page_size=args.page_size,
            max_samples=args.max_samples,
            request_sleep=args.request_sleep,
            backend=args.backend,
        ):
            source_hint = f"{dataset_name}-{wrapped.get('row_idx')}"
            split = "test" if stable_is_val(source_hint, args.val_ratio, args.seed) else "train"

            try:
                out = convert_row(dataset_name, wrapped, split, args)
                source_id = out["extra_info"]["source_id"]
                if source_id in seen_ids:
                    raise ValueError("duplicate_source_id")
                seen_ids.add(source_id)
            except Exception as exc:
                reason = str(exc).split(": ", 1)[-1].replace(" ", "_")
                stats[f"skip/{reason}"] += 1
                continue

            prompt_lengths.append(len(out["prompt"][0]["content"]))
            write_jsonl_row(test_f if split == "test" else train_f, out)
            stats[f"keep/{split}"] += 1
            stats["keep/rows"] += 1
            if stats["keep/rows"] % 10000 == 0:
                print(f"{dataset_name}: converted {stats['keep/rows']} rows")

    add_prompt_stats(stats, prompt_lengths)
    meta = {
        "dataset": dataset_name,
        "hf_dataset": hf_dataset,
        "hf_config": config,
        "hf_split": hf_split,
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "max_samples": args.max_samples,
        "backend": args.backend,
        "max_code_tests": args.max_code_tests if dataset_name == "taco" else None,
        "scienceqa_include_images": args.scienceqa_include_images if dataset_name == "scienceqa" else None,
        "stats": dict(sorted(stats.items())),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"{dataset_name}: wrote {train_path} and {test_path}")
    for key in sorted(stats):
        print(f"  {key}={stats[key]}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and convert RLVR datasets into verl JSONL schema.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["all"],
        choices=["all", "openr1_math", "taco", "scienceqa"],
        help="Datasets to download and convert.",
    )
    parser.add_argument("--base-dir", type=Path, default=Path("datasets"), help="Output dataset directory.")
    parser.add_argument("--val-ratio", type=float, default=0.02, help="Hash-based validation split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for hash-based train/test split.")
    parser.add_argument("--page-size", type=int, default=100, help="Rows API page size.")
    parser.add_argument("--request-sleep", type=float, default=1.0, help="Sleep seconds between Rows API requests.")
    parser.add_argument(
        "--backend",
        choices=["auto", "hf", "rows"],
        default="auto",
        help="auto: try datasets then Rows API; hf: force datasets; rows: force Rows API.",
    )
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap for debugging.")
    parser.add_argument(
        "--max-code-tests",
        type=int,
        default=DEFAULT_CODE_TEST_LIMIT,
        help="Maximum TACO tests retained per example. Use <=0 to keep all tests.",
    )
    parser.add_argument(
        "--scienceqa-include-images",
        action="store_true",
        help="Keep ScienceQA rows with images. Default skips them for text-only training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 <= args.val_ratio < 1:
        raise ValueError(f"val_ratio must be in [0, 1), got {args.val_ratio}")

    selected = set(DATASET_SPECS) if "all" in args.datasets else set(args.datasets)
    for dataset_name in ["openr1_math", "taco", "scienceqa"]:
        if dataset_name in selected:
            convert_dataset(dataset_name, args)


if __name__ == "__main__":
    main()
