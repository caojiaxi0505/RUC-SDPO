#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import time
import urllib.error
import urllib.request
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any


JsonDict = dict[str, Any]


def load_jsonl(path: Path) -> list[JsonDict]:
    rows: list[JsonDict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"no rows loaded from {path}")
    return rows


def prompt_messages(row: JsonDict) -> list[JsonDict]:
    prompt = row.get("prompt")
    if isinstance(prompt, list):
        return [{"role": str(msg["role"]), "content": str(msg["content"])} for msg in prompt]
    if isinstance(prompt, str):
        return [{"role": "user", "content": prompt}]
    raise ValueError("row prompt must be a list of chat messages or a string")


def request_chat_completions(args: argparse.Namespace, row: JsonDict, request_idx: int) -> int:
    payload: JsonDict = {
        "model": args.model,
        "messages": prompt_messages(row),
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_tokens": args.max_tokens,
        "n": args.num_samples,
    }
    if args.seed >= 0:
        payload["seed"] = args.seed + request_idx

    req = urllib.request.Request(
        args.base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ.get(args.api_key_env, 'EMPTY')}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=args.request_timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    return len(data.get("choices") or [])


def row_stream(rows: list[JsonDict], seed: int):
    rng = random.Random(seed)
    indices = list(range(len(rows)))
    cycle = 0

    while True:
        rng.shuffle(indices)
        for idx in indices:
            yield cycle, idx, rows[idx]
        cycle += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Continuously send eval-like requests without writing artifacts.")
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument("--seed", type=int, default=42, help="Use -1 to omit per-request seeds.")
    parser.add_argument("--log-every", type=int, default=20)
    args = parser.parse_args()

    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive")
    if args.concurrency <= 0:
        raise ValueError("--concurrency must be positive")
    if args.log_every <= 0:
        raise ValueError("--log-every must be positive")
    return args


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.data_path)
    stream = row_stream(rows, seed=max(args.seed, 0))

    print(
        "[busy] loop started: "
        f"rows={len(rows)} concurrency={args.concurrency} "
        f"n={args.num_samples} max_tokens={args.max_tokens}"
    )

    submitted = 0
    completed = 0
    errors = 0
    choices = 0
    started_at = time.time()

    def submit_one(pool: ThreadPoolExecutor):
        nonlocal submitted
        _, _, row = next(stream)
        submitted += 1
        return pool.submit(request_chat_completions, args, row, submitted)

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {submit_one(pool) for _ in range(args.concurrency)}

        while True:
            done, futures = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                completed += 1
                try:
                    choices += future.result()
                except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError, ValueError) as exc:
                    errors += 1
                    print(f"[busy] request error: {type(exc).__name__}: {exc}", flush=True)

                futures.add(submit_one(pool))

                if completed % args.log_every == 0:
                    elapsed = max(time.time() - started_at, 1e-6)
                    print(
                        "[busy] "
                        f"completed={completed} errors={errors} choices={choices} "
                        f"req/s={completed / elapsed:.3f} choices/s={choices / elapsed:.3f}",
                        flush=True,
                    )


if __name__ == "__main__":
    main()
