"""Long-running batch verifier service for TACO rewards.

The service intentionally reuses ``feedback.code.compute_score`` as the single
source of truth for TACO scoring. It only adds an HTTP batch interface and a
bounded worker pool so training can score many rollouts in parallel without
changing reward semantics.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import numpy as np

from verl.utils.reward_score.feedback import code


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 18080
DEFAULT_WORKERS = 2

_EXECUTOR: ThreadPoolExecutor | None = None
_DEFAULT_MAX_TEST_CASES: int | None = None


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and (value != value):
        return None
    return value


def _failure_score(message: str) -> dict[str, Any]:
    return {
        "score": 0.0,
        "acc": 0.0,
        "pred": "",
        "incorrect_format": 0,
        "error_in_test_cases": 1,
        "timed_out": 0,
        "truncated": 0,
        "truncated_and_missing_answer": 0,
        "feedback": message,
        "verifier_error": 1,
    }


def _score_one(item: dict[str, Any], sparse_rewards: bool, max_test_cases: int | None) -> dict[str, Any]:
    item_id = item.get("id")
    try:
        result = code.compute_score(
            solution=item.get("solution", ""),
            ground_truth=item.get("ground_truth", ""),
            extra_info=item.get("extra_info") or {},
            sparse_rewards=sparse_rewards,
            max_test_cases=max_test_cases,
        )
        result["verifier_error"] = 0
        return {"id": item_id, "ok": True, "score": _json_safe(result)}
    except Exception as exc:
        return {
            "id": item_id,
            "ok": False,
            "score": _failure_score(f"TACO verifier failed: {type(exc).__name__}: {exc}"),
        }


def verify_batch(payload: dict[str, Any]) -> dict[str, Any]:
    if _EXECUTOR is None:
        raise RuntimeError("verifier executor is not initialized")

    items = payload.get("items")
    if not isinstance(items, list):
        raise ValueError("request field 'items' must be a list")

    sparse_rewards = bool(payload.get("sparse_rewards", True))
    max_test_cases = payload.get("max_test_cases", _DEFAULT_MAX_TEST_CASES)
    if max_test_cases is not None:
        max_test_cases = int(max_test_cases)
        if max_test_cases <= 0:
            max_test_cases = None

    started = time.time()
    results: list[dict[str, Any] | None] = [None] * len(items)
    futures = {
        _EXECUTOR.submit(_score_one, item, sparse_rewards, max_test_cases): idx
        for idx, item in enumerate(items)
    }
    for future in as_completed(futures):
        idx = futures[future]
        results[idx] = future.result()

    return {
        "ok": True,
        "count": len(items),
        "elapsed_sec": time.time() - started,
        "results": results,
    }


def _parse_optional_positive_int(value: str | None) -> int | None:
    if value in (None, ""):
        return None
    parsed = int(value)
    if parsed <= 0:
        return None
    return parsed


def _default_workers() -> int:
    cpu_count = os.cpu_count() or 8
    return min(DEFAULT_WORKERS, max(1, cpu_count))


class TacoBatchVerifierHandler(BaseHTTPRequestHandler):
    server_version = "TacoBatchVerifier/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        if os.environ.get("TACO_VERIFIER_ACCESS_LOG", "0") == "1":
            super().log_message(fmt, *args)

    def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(_json_safe(payload), ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path != "/health":
            self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not found"})
            return
        self._send_json(HTTPStatus.OK, {"ok": True})

    def do_POST(self) -> None:
        if self.path not in {"/verify", "/verify_batch"}:
            self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not found"})
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(content_length) or b"{}")
            self._send_json(HTTPStatus.OK, verify_batch(payload))
        except Exception as exc:
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {"ok": False, "error": f"{type(exc).__name__}: {exc}"},
            )


def parse_args() -> argparse.Namespace:
    env_max_test_cases = _parse_optional_positive_int(os.environ.get("TACO_VERIFIER_MAX_TEST_CASES"))
    env_workers = _parse_optional_positive_int(os.environ.get("TACO_VERIFIER_WORKERS"))
    parser = argparse.ArgumentParser(description="Start a TACO batch verifier HTTP service.")
    parser.add_argument("--host", default=os.environ.get("TACO_VERIFIER_HOST", DEFAULT_HOST))
    parser.add_argument("--port", type=int, default=int(os.environ.get("TACO_VERIFIER_PORT", DEFAULT_PORT)))
    parser.add_argument(
        "--workers",
        type=int,
        default=env_workers,
        help="Maximum number of rollouts scored concurrently.",
    )
    parser.add_argument(
        "--max-test-cases",
        type=int,
        default=env_max_test_cases,
        help="Optional default cap for train split test cases. Omit to preserve original reward semantics.",
    )
    args = parser.parse_args()
    if args.workers is None:
        args.workers = _default_workers()
    if args.workers <= 0:
        raise ValueError("--workers must be a positive integer")
    if args.max_test_cases is not None and args.max_test_cases <= 0:
        args.max_test_cases = None
    return args


def main() -> None:
    global _EXECUTOR, _DEFAULT_MAX_TEST_CASES

    args = parse_args()
    _DEFAULT_MAX_TEST_CASES = args.max_test_cases
    _EXECUTOR = ThreadPoolExecutor(max_workers=max(1, args.workers), thread_name_prefix="taco-verifier")

    server = ThreadingHTTPServer((args.host, args.port), TacoBatchVerifierHandler)
    print(
        "[taco-verifier] listening on "
        f"http://{args.host}:{args.port} workers={args.workers} "
        f"max_test_cases={_DEFAULT_MAX_TEST_CASES}",
        flush=True,
    )
    if _DEFAULT_MAX_TEST_CASES is not None:
        print(
            "[taco-verifier] WARNING: max_test_cases only caps train split test cases "
            "and changes train reward semantics; validation/test still run all test cases.",
            flush=True,
        )
    try:
        server.serve_forever()
    finally:
        server.server_close()
        _EXECUTOR.shutdown(wait=False, cancel_futures=True)


if __name__ == "__main__":
    main()
