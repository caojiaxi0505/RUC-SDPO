#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import numbers
import os
import sys
import time
import types
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
FEEDBACK_DIR = REPO_ROOT / "verl" / "utils" / "reward_score" / "feedback"

for package_name, package_path in [
    ("verl", REPO_ROOT / "verl"),
    ("verl.utils", REPO_ROOT / "verl" / "utils"),
    ("verl.utils.reward_score", REPO_ROOT / "verl" / "utils" / "reward_score"),
    ("verl.utils.reward_score.feedback", FEEDBACK_DIR),
]:
    package = types.ModuleType(package_name)
    package.__path__ = [str(package_path)]  # type: ignore[attr-defined]
    sys.modules.setdefault(package_name, package)


JsonDict = dict[str, Any]


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


math_reward = load_module("verl.utils.reward_score.feedback.math", FEEDBACK_DIR / "math.py")
code_reward = load_module("verl.utils.reward_score.feedback.code", FEEDBACK_DIR / "code.py")
mcq_reward = load_module("verl.utils.reward_score.feedback.mcq", FEEDBACK_DIR / "mcq.py")
tooluse_reward = load_module("verl.utils.reward_score.feedback.tooluse", FEEDBACK_DIR / "tooluse.py")

CODE_SOURCES = {"code", "livecodebench", "humanevalplus", "taco"}
MATH_SOURCES = {"math", "math500", "dapo_math", "gsm8k", "openr1_math"}
MCQ_SOURCES = {"sciknoweval", "scienceqa", "medmcqa"}
BEST_OF_K_METRICS = {"score", "acc"}


def compute_reward_score(data_source: str, solution_str: str, ground_truth: str, extra_info: dict | None = None) -> dict:
    source = str(data_source).lower()
    if source in CODE_SOURCES:
        return code_reward.compute_score(solution_str, ground_truth, extra_info, sparse_rewards=True, max_test_cases=None)
    if source in MATH_SOURCES:
        return math_reward.compute_score(solution_str, ground_truth, extra_info or {})
    if source in MCQ_SOURCES:
        return mcq_reward.compute_score(solution_str, ground_truth)
    if "tooluse" in source:
        return tooluse_reward.compute_score(solution_str, ground_truth)
    raise ValueError(f"Reward style {data_source} not found.")


def json_dumps(obj: Any) -> str:
    return json.dumps(to_jsonable(obj), ensure_ascii=False, separators=(",", ":"))


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [to_jsonable(v) for v in value]
    if isinstance(value, numbers.Integral) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        return float(value)
    return value


def is_numeric(value: Any) -> bool:
    return (
        isinstance(value, numbers.Real)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def short_error(exc: BaseException, limit: int = 500) -> str:
    return f"{type(exc).__name__}: {exc}"[:limit]


def load_jsonl(path: Path, limit: int | None = None) -> list[JsonDict]:
    rows: list[JsonDict] = []
    with path.open(encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            if limit is not None and len(rows) >= limit:
                break
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row["_line_idx"] = line_idx
            rows.append(row)
    return rows


def row_uid(row: JsonDict, dataset: str) -> str:
    info = row.get("extra_info") or {}
    return str(info.get("source_id") or info.get("index") or f"{dataset}-{row['_line_idx']}")


def prompt_messages(row: JsonDict) -> list[JsonDict]:
    prompt = row.get("prompt")
    if isinstance(prompt, list):
        return [{"role": str(msg["role"]), "content": str(msg["content"])} for msg in prompt]
    if isinstance(prompt, str):
        return [{"role": "user", "content": prompt}]
    raise ValueError("row prompt must be a list of chat messages or a string")


def load_generation_cache(path: Path) -> dict[str, JsonDict]:
    cached: dict[str, JsonDict] = {}
    if not path.exists():
        return cached
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            uid = record.get("uid")
            if uid:
                cached[str(uid)] = record
    return cached


def request_chat_completions(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: list[JsonDict],
    num_samples: int,
    temperature: float,
    top_p: float,
    top_k: int,
    max_tokens: int,
    chat_template_kwargs: dict[str, Any],
    seed: int | None,
    timeout: float,
    retries: int,
    retry_sleep: float,
) -> list[JsonDict]:
    url = base_url.rstrip("/") + "/chat/completions"
    payload: JsonDict = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "max_tokens": max_tokens,
        "n": num_samples,
    }
    if chat_template_kwargs:
        payload["chat_template_kwargs"] = chat_template_kwargs
    if seed is not None:
        payload["seed"] = seed

    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    last_error: BaseException | None = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, data=body, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            choices = sorted(data.get("choices", []), key=lambda x: x.get("index", 0))
            out: list[JsonDict] = []
            for choice in choices:
                message = choice.get("message") or {}
                out.append(
                    {
                        "index": int(choice.get("index", len(out))),
                        "text": str(message.get("content") or choice.get("text") or ""),
                        "finish_reason": choice.get("finish_reason"),
                    }
                )
            return out
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError) as exc:
            last_error = exc
            if attempt >= retries:
                break
            time.sleep(retry_sleep * (2**attempt))

    assert last_error is not None
    raise last_error


def make_generation_record(row: JsonDict, args: argparse.Namespace, uid: str, row_idx: int) -> JsonDict:
    info = row.get("extra_info") or {}
    ground_truth = (row.get("reward_model") or {}).get("ground_truth")
    if ground_truth is None:
        raise ValueError(f"{uid}: missing reward_model.ground_truth")

    seed = None if args.seed is None else args.seed + row_idx
    try:
        generations = request_chat_completions(
            base_url=args.base_url,
            api_key=os.environ.get(args.api_key_env, "EMPTY"),
            model=args.model,
            messages=prompt_messages(row),
            num_samples=args.num_samples,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_tokens=args.max_tokens,
            chat_template_kwargs=args.chat_template_kwargs,
            seed=seed,
            timeout=args.request_timeout,
            retries=args.retries,
            retry_sleep=args.retry_sleep,
        )
        error = ""
    except BaseException as exc:
        generations = []
        error = short_error(exc)

    return {
        "uid": uid,
        "dataset": args.dataset,
        "data_source": row.get("data_source", args.dataset),
        "prompt": prompt_messages(row),
        "ground_truth": ground_truth,
        "extra_info": info,
        "generations": generations,
        "error": error,
    }


def generate_missing(rows: list[JsonDict], args: argparse.Namespace, gen_path: Path) -> dict[str, JsonDict]:
    cached: dict[str, JsonDict] = {}
    if not args.overwrite:
        cached = load_generation_cache(gen_path)
    elif gen_path.exists():
        gen_path.unlink()

    missing: list[tuple[int, str, JsonDict]] = []
    seen: set[str] = set()
    for row_idx, row in enumerate(rows):
        uid = row_uid(row, args.dataset)
        if uid in seen:
            raise ValueError(f"duplicate uid in dataset: {uid}")
        seen.add(uid)
        record = cached.get(uid)
        if record and not record.get("error") and len(record.get("generations") or []) >= args.num_samples:
            continue
        missing.append((row_idx, uid, row))

    if not missing:
        print(f"[eval] generation cache complete: {gen_path}")
        return cached

    print(f"[eval] generating {len(missing)} / {len(rows)} prompts -> {gen_path}")
    gen_path.parent.mkdir(parents=True, exist_ok=True)
    completed = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool, gen_path.open("a", encoding="utf-8") as out_f:
        futures = {
            pool.submit(make_generation_record, row, args, uid, row_idx): uid
            for row_idx, uid, row in missing
        }
        for future in as_completed(futures):
            uid = futures[future]
            record = future.result()
            cached[uid] = record
            out_f.write(json_dumps(record) + "\n")
            out_f.flush()
            completed += 1
            if completed % args.log_every == 0 or completed == len(missing):
                print(f"[eval] generated {completed}/{len(missing)}; last_uid={uid}")
    return cached


def zero_score(error: str) -> JsonDict:
    return {
        "score": 0.0,
        "acc": 0.0,
        "pred": "",
        "incorrect_format": 1,
        "feedback": "",
        "eval_error": error,
    }


def score_record(record: JsonDict, num_samples: int) -> JsonDict:
    data_source = str(record["data_source"])
    ground_truth = record["ground_truth"]
    base_extra = dict(record.get("extra_info") or {})
    base_extra["split"] = "test"
    generations = list(record.get("generations") or [])

    scored: list[JsonDict] = []
    for sample_idx in range(num_samples):
        if sample_idx >= len(generations):
            scored.append({"sample_idx": sample_idx, **zero_score("missing_generation")})
            continue

        generation = generations[sample_idx]
        solution = str(generation.get("text", ""))
        extra_info = dict(base_extra)
        extra_info["truncated"] = generation.get("finish_reason") == "length"
        try:
            result = compute_reward_score(
                data_source=data_source,
                solution_str=solution,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )
            if not isinstance(result, dict):
                result = {"score": float(result), "acc": float(result), "pred": ""}
        except BaseException as exc:
            result = zero_score(short_error(exc))

        scored.append(
            {
                "sample_idx": sample_idx,
                "finish_reason": generation.get("finish_reason"),
                **to_jsonable(result),
            }
        )

    return {
        "uid": record["uid"],
        "dataset": record.get("dataset"),
        "data_source": data_source,
        "generation_error": record.get("error", ""),
        "scores": scored,
    }


def score_all(
    rows: list[JsonDict],
    generations: dict[str, JsonDict],
    args: argparse.Namespace,
    scored_path: Path,
) -> list[JsonDict]:
    scored_rows: list[JsonDict] = []
    with scored_path.open("w", encoding="utf-8") as out_f:
        for row in rows:
            uid = row_uid(row, args.dataset)
            record = generations.get(uid)
            if record is None:
                record = {
                    "uid": uid,
                    "dataset": args.dataset,
                    "data_source": row.get("data_source", args.dataset),
                    "ground_truth": (row.get("reward_model") or {}).get("ground_truth", ""),
                    "extra_info": row.get("extra_info") or {},
                    "generations": [],
                    "error": "missing_generation_record",
                }
            scored = score_record(record, args.num_samples)
            scored_rows.append(scored)
            out_f.write(json_dumps(scored) + "\n")
    return scored_rows


def default_metric_ks(num_samples: int) -> list[int]:
    ks: list[int] = []
    k = 1
    while k < num_samples:
        ks.append(k)
        k *= 2
    if num_samples not in ks:
        ks.append(num_samples)
    return ks


def parse_metric_ks(text: str | None, num_samples: int) -> list[int]:
    if not text:
        return default_metric_ks(num_samples)
    ks = sorted({int(part.strip()) for part in text.split(",") if part.strip()})
    if not ks:
        return default_metric_ks(num_samples)
    bad = [k for k in ks if k <= 0 or k > num_samples]
    if bad:
        raise ValueError(f"metric ks must be in [1, num_samples={num_samples}], got {bad}")
    return ks


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def numeric_metric_names(scored_rows: list[JsonDict]) -> list[str]:
    numeric_keys: set[str] = set()
    for row in scored_rows:
        for score in row["scores"]:
            for key, value in score.items():
                if is_numeric(value):
                    numeric_keys.add(key)
    return sorted(numeric_keys)


def score_metric_value(score: JsonDict, metric_name: str) -> float:
    value = score.get(metric_name)
    return float(value) if is_numeric(value) else 0.0


def metric_values_by_prompt(scored_rows: list[JsonDict], metric_name: str) -> list[list[float]]:
    return [
        [score_metric_value(score, metric_name) for score in row["scores"]]
        for row in scored_rows
    ]


def aggregate_metric(values_by_prompt: list[list[float]], metric_name: str, ks: list[int]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for k in ks:
        metrics[f"mean@{k}"] = mean([mean(values[:k]) for values in values_by_prompt])
        if metric_name in BEST_OF_K_METRICS:
            metrics[f"best@{k}"] = mean([max(values[:k]) for values in values_by_prompt])
    return metrics


def count_eval_errors(scored_rows: list[JsonDict]) -> int:
    return sum(
        1
        for row in scored_rows
        for score in row["scores"]
        if score.get("eval_error")
    )


def aggregate_group(scored_rows: list[JsonDict], ks: list[int]) -> JsonDict:
    metrics = {
        metric_name: aggregate_metric(metric_values_by_prompt(scored_rows, metric_name), metric_name, ks)
        for metric_name in numeric_metric_names(scored_rows)
    }

    return {
        "num_prompts": len(scored_rows),
        "generation_error_prompts": sum(1 for row in scored_rows if row.get("generation_error")),
        "eval_error_samples": count_eval_errors(scored_rows),
        "metrics": metrics,
    }


def build_summary(scored_rows: list[JsonDict], args: argparse.Namespace, ks: list[int]) -> JsonDict:
    by_source: dict[str, list[JsonDict]] = defaultdict(list)
    for row in scored_rows:
        by_source[str(row["data_source"])].append(row)

    return {
        "dataset": args.dataset,
        "data_path": str(args.data_path),
        "model": args.model,
        "base_url": args.base_url,
        "num_samples": args.num_samples,
        "metric_ks": ks,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_tokens": args.max_tokens,
        "chat_template_kwargs": args.chat_template_kwargs,
        "overall": aggregate_group(scored_rows, ks),
        "by_data_source": {source: aggregate_group(rows, ks) for source, rows in sorted(by_source.items())},
    }


def write_summary_text(summary: JsonDict, path: Path) -> None:
    lines = [
        f"dataset: {summary['dataset']}",
        f"model: {summary['model']}",
        f"data_path: {summary['data_path']}",
        f"num_prompts: {summary['overall']['num_prompts']}",
        f"num_samples: {summary['num_samples']}",
        f"metric_ks: {','.join(str(k) for k in summary['metric_ks'])}",
        f"temperature: {summary['temperature']}",
        f"top_p: {summary['top_p']}",
        f"top_k: {summary['top_k']}",
        f"max_tokens: {summary['max_tokens']}",
        f"chat_template_kwargs: {json_dumps(summary.get('chat_template_kwargs', {}))}",
        "",
    ]
    for metric_name in ["score", "acc"]:
        metric = summary["overall"]["metrics"].get(metric_name)
        if not metric:
            continue
        lines.append(f"[{metric_name}]")
        for key in sorted(metric, key=lambda x: (x.split("@")[0], int(x.split("@")[1].split("/")[0]))):
            lines.append(f"{key}: {metric[key]:.6f}")
        lines.append("")

    lines.extend(
        [
            f"generation_error_prompts: {summary['overall']['generation_error_prompts']}",
            f"eval_error_samples: {summary['overall']['eval_error_samples']}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate mean@k and best@k on converted verl JSONL datasets.")
    parser.add_argument("--dataset", required=True, help="Dataset name for logging.")
    parser.add_argument("--data-path", type=Path, required=True, help="Converted test JSONL path.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for generations, scores, and summary.")
    parser.add_argument("--model", required=True, help="OpenAI/vLLM served model name.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1", help="OpenAI-compatible /v1 base URL.")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY", help="Environment variable containing API key.")
    parser.add_argument("--num-samples", type=int, default=16, help="Number of completions per prompt.")
    parser.add_argument("--metric-ks", default="", help="Comma-separated k values. Default: 1,2,4,...,num_samples.")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument(
        "--chat-template-kwargs-json",
        default="",
        help="JSON object forwarded to vLLM chat_template_kwargs, e.g. '{\"enable_thinking\": false}'.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Base sampling seed. Use -1 to omit seed.")
    parser.add_argument("--concurrency", type=int, default=8, help="Concurrent API requests.")
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=5.0)
    parser.add_argument("--limit", type=int, default=None, help="Optional prompt cap for smoke tests.")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate even if generations.jsonl exists.")
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args()

    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive")
    if args.concurrency <= 0:
        raise ValueError("--concurrency must be positive")
    if args.seed == -1:
        args.seed = None
    if args.chat_template_kwargs_json:
        parsed_kwargs = json.loads(args.chat_template_kwargs_json)
        if not isinstance(parsed_kwargs, dict):
            raise ValueError("--chat-template-kwargs-json must decode to a JSON object")
        args.chat_template_kwargs = parsed_kwargs
    else:
        args.chat_template_kwargs = {}
    return args


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_jsonl(args.data_path, limit=args.limit)
    if not rows:
        raise ValueError(f"no rows loaded from {args.data_path}")

    ks = parse_metric_ks(args.metric_ks, args.num_samples)
    gen_path = args.output_dir / "generations.jsonl"
    scored_path = args.output_dir / "scored.jsonl"
    summary_json_path = args.output_dir / "summary.json"
    summary_txt_path = args.output_dir / "summary.txt"

    print(f"[eval] dataset={args.dataset} rows={len(rows)} num_samples={args.num_samples} ks={ks}")
    generations = generate_missing(rows, args, gen_path)
    print(f"[eval] scoring -> {scored_path}")
    scored_rows = score_all(rows, generations, args, scored_path)
    summary = build_summary(scored_rows, args, ks)

    summary_json_path.write_text(json.dumps(to_jsonable(summary), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_summary_text(summary, summary_txt_path)
    print(f"[eval] summary -> {summary_txt_path}")
    print(summary_txt_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
