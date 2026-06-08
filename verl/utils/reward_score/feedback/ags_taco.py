from __future__ import annotations

import asyncio
import base64
import json
import math
import os
import shlex
import threading
import time
import uuid
from typing import Any

from verl.utils.reward_score.feedback import code

try:
    import e2b.api
    from e2b import AsyncSandbox
    from e2b.sandbox.commands.command_handle import CommandExitException

    _HAS_E2B = True
except ImportError:  # pragma: no cover - optional AGS/E2B dependency
    AsyncSandbox = None  # type: ignore[assignment]
    CommandExitException = Exception  # type: ignore[assignment]
    _HAS_E2B = False


_RECORD_MARKER = "__TACO_AGS_RECORD__"


def _skip_e2b_api_key_validation(api_key: str) -> None:
    return None


if _HAS_E2B:
    # AGS uses an ark-style key; the E2B SDK local validator rejects it before
    # the request reaches the AGS data plane unless validation is disabled.
    setattr(e2b.api, "validate_api_key", _skip_e2b_api_key_validation)


def _b64_json(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False).encode("utf-8")
    return base64.b64encode(raw).decode("ascii")


def _b64_text(value: str) -> str:
    return base64.b64encode(value.encode("utf-8")).decode("ascii")


def _short_text(value: Any, limit: int = 1000) -> str:
    text = "" if value is None else str(value)
    return text if len(text) <= limit else text[:limit] + "..."


def _build_case_script(
    *,
    completion: str,
    test_type: str,
    fn_name: str,
    context: str,
    test_input: Any,
    test_output: Any,
    test_idx: int,
) -> str:
    payload_b64 = _b64_json(
        {
            "test_type": test_type,
            "fn_name": fn_name,
            "input": test_input,
            "output": test_output,
            "test_idx": test_idx,
        }
    )
    completion_b64 = _b64_text(completion)
    context_b64 = _b64_text(context or "")

    return f"""
import __future__
import ast
import base64
import io
import json
import sys
import time
import traceback
from collections.abc import MutableMapping

MARKER = {_RECORD_MARKER!r}
FILENAME = "Solution.py"
TESTS_FILENAME = "Tests.py"
CONTEXT_FILENAME = "Context.py"
ERROR_PREFIX = "Error: "
PROTECTED_GLOBAL_NAMES = {{"debug_print", "__debug_buffer__"}}


def _decode_text(value):
    return base64.b64decode(value.encode("ascii")).decode("utf-8")


payload = json.loads(_decode_text({payload_b64!r}))
completion = _decode_text({completion_b64!r})
context = _decode_text({context_b64!r})


def _short_trace(exc, limit=3):
    frames = traceback.extract_tb(exc.__traceback__)
    solution_frames = [
        frame for frame in frames
        if isinstance(getattr(frame, "filename", None), str) and frame.filename == FILENAME
    ]
    tail = solution_frames[-limit:] if solution_frames else []
    lines = [f"{{type(exc).__name__}}: {{exc}}"]
    for frame in tail:
        if frame.line:
            lines.append(f"  {{frame.line}}")
        lines.append(f"Line {{frame.lineno}} in {{frame.name}} ({{FILENAME}})")
    return "\\n".join(lines)


def _to_safe_jsonable(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_to_safe_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {{str(key): _to_safe_jsonable(item) for key, item in value.items()}}
    raise TypeError(f"Non-serializable result type: {{type(value).__name__}}")


def _functional_call_args(test_input):
    if isinstance(test_input, dict):
        return (), test_input
    if isinstance(test_input, (list, tuple)):
        return list(test_input), None
    if isinstance(test_input, str):
        stripped = test_input.strip()
        if not stripped:
            return [], None
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, dict):
                return (), parsed
            if isinstance(parsed, (list, tuple)):
                return list(parsed), None
            return [parsed], None
        except Exception:
            return [json.loads(part) for part in stripped.split()], None
    return [test_input], None


def _expected_functional_output(test_output):
    if isinstance(test_output, str):
        try:
            test_output = json.loads(test_output)
        except Exception:
            return test_output
    if isinstance(test_output, (list, tuple)) and len(test_output) == 1:
        return test_output[0]
    return test_output


def _stdio_text(value):
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return "\\n".join(str(item) for item in value) + "\\n"
    return str(value)


def _exec_with_isolated_locals(code_obj, globals_ns):
    class _GuardedLocals(MutableMapping):
        def __init__(self, backing):
            self._backing = backing
        def __getitem__(self, key):
            return self._backing[key]
        def __setitem__(self, key, value):
            if key in PROTECTED_GLOBAL_NAMES:
                return
            self._backing[key] = value
        def __delitem__(self, key):
            if key in PROTECTED_GLOBAL_NAMES:
                return
            del self._backing[key]
        def __iter__(self):
            return iter(self._backing)
        def __len__(self):
            return len(self._backing)

    exec(code_obj, globals_ns, _GuardedLocals(globals_ns))


def _new_namespace():
    ns = {{"__name__": "__not_main__"}}
    debug_buffer = io.StringIO()

    def debug_print(*args, **kwargs):
        sep = kwargs.get("sep", " ")
        end = kwargs.get("end", "\\n")
        debug_buffer.write(sep.join(str(arg) for arg in args) + end)

    ns["debug_print"] = debug_print
    ns["__debug_buffer__"] = debug_buffer

    try:
        import typing as _typing
        _common_typing_names = (
            "Any",
            "Optional",
            "Union",
            "List",
            "Dict",
            "Set",
            "Tuple",
            "Callable",
            "Iterable",
            "Iterator",
            "Sequence",
            "Mapping",
            "MutableMapping",
            "MutableSequence",
            "MutableSet",
            "DefaultDict",
            "Deque",
            "FrozenSet",
            "Type",
            "TypeVar",
            "Generic",
            "Literal",
            "TypedDict",
            "NoReturn",
            "overload",
        )
        for _name in _common_typing_names:
            if hasattr(_typing, _name):
                ns[_name] = getattr(_typing, _name)
    except Exception:
        pass

    return ns


def _infer_func_name(src, explicit_name, namespace):
    try:
        tree = ast.parse(src)
        if explicit_name:
            for node in tree.body:
                if isinstance(node, ast.FunctionDef) and node.name == explicit_name:
                    return explicit_name
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                return node.name
    except Exception:
        pass
    if explicit_name and explicit_name in namespace and callable(namespace[explicit_name]):
        return explicit_name
    return None


def _emit(record):
    print(MARKER + json.dumps(record, ensure_ascii=False, default=str), flush=True)


def _run_functional(namespace, test_input, test_output, fn_name):
    code_obj = compile(
        completion,
        FILENAME,
        "exec",
        flags=__future__.annotations.compiler_flag,
        dont_inherit=True,
    )
    _exec_with_isolated_locals(code_obj, namespace)
    func_name = fn_name if fn_name and fn_name in namespace and callable(namespace[fn_name]) else None
    if func_name is None:
        func_name = _infer_func_name(completion, fn_name, namespace)
    if func_name not in namespace or not callable(namespace.get(func_name, None)):
        try:
            func_name = completion.split("(")[0].split()[-1]
        except Exception:
            pass
    if func_name not in namespace or not callable(namespace.get(func_name, None)):
        raise NameError(f"Function {{fn_name or '<unknown>'}} not found")

    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        args, kwargs = _functional_call_args(test_input)
        result = namespace[func_name](**kwargs) if kwargs is not None else namespace[func_name](*args)
    finally:
        sys.stdout = old_stdout

    lhs = _to_safe_jsonable(result)
    rhs = _to_safe_jsonable(_expected_functional_output(test_output))
    passed = json.dumps(lhs, sort_keys=True, separators=(",", ":")) == json.dumps(
        rhs,
        sort_keys=True,
        separators=(",", ":"),
    )
    return passed, result


def _run_stdin(namespace, test_input, test_output):
    if isinstance(test_output, str):
        test_output = test_output.strip()
        if test_output.endswith("-"):
            test_output = test_output[: test_output.rfind("-")].rstrip()

    output = io.StringIO()
    old_stdout, old_stdin = sys.stdout, sys.stdin
    try:
        sys.stdout = output
        sys.stdin = io.StringIO(_stdio_text(test_input))
        namespace["__name__"] = "__main__"
        _exec_with_isolated_locals(compile('__name__ = "__main__"\\n' + completion, FILENAME, "exec"), namespace)
    finally:
        sys.stdout = old_stdout
        sys.stdin = old_stdin

    actual = output.getvalue().strip()
    normalized_actual = actual.replace("\\n", " ").replace("\\r", "")
    normalized_expected = _stdio_text(test_output).strip().replace("\\n", " ").replace("\\r", "")
    return normalized_actual == normalized_expected, actual


def _run_code(namespace, test_input):
    namespace["__name__"] = "__main__"
    _exec_with_isolated_locals(compile(completion, FILENAME, "exec"), namespace)
    _exec_with_isolated_locals(compile(test_input, TESTS_FILENAME, "exec"), namespace)
    return True, "All tests pass"


def main():
    namespace = _new_namespace()
    test_idx = payload["test_idx"]
    test_input = payload["input"]
    test_output = payload["output"]
    start = time.time()
    old_stderr = sys.stderr
    sys.stderr = namespace["__debug_buffer__"]
    try:
        if context.strip():
            _exec_with_isolated_locals(compile(context, CONTEXT_FILENAME, "exec"), namespace)
        if payload["test_type"] == "functional":
            passed, actual = _run_functional(namespace, test_input, test_output, payload.get("fn_name") or "")
        elif payload["test_type"] == "stdin":
            passed, actual = _run_stdin(namespace, test_input, test_output)
        elif payload["test_type"] == "code":
            passed, actual = _run_code(namespace, test_input)
        else:
            raise ValueError(f"Invalid test type: {{payload['test_type']}}")
    except BaseException as exc:
        passed = False
        actual = ERROR_PREFIX + _short_trace(exc)
    finally:
        sys.stderr = old_stderr

    debug_buffer = namespace.get("__debug_buffer__")
    _emit({{
        "test_idx": test_idx,
        "input": test_input,
        "expected": test_output,
        "actual": actual,
        "passed": bool(passed),
        "debug": debug_buffer.getvalue() if debug_buffer is not None else "",
        "time": time.time() - start,
    }})


if __name__ == "__main__":
    main()
"""


def _failure_record(test_cases: dict[str, Any], test_idx: int, actual: str) -> dict[str, Any]:
    return {
        "test_idx": test_idx,
        "input": test_cases["inputs"][test_idx],
        "expected": test_cases["outputs"][test_idx],
        "actual": actual,
        "passed": False,
        "debug": "",
        "time": float("inf"),
    }


def _timeout_record(test_cases: dict[str, Any], test_idx: int, elapsed: float) -> dict[str, Any]:
    return {
        "test_idx": test_idx,
        "input": test_cases["inputs"][test_idx],
        "expected": test_cases["outputs"][test_idx],
        "actual": code.TIMEOUT,
        "passed": False,
        "debug": "",
        "time": elapsed,
    }


def _is_timeout_exception(exc: BaseException) -> bool:
    exc_name = type(exc).__name__.lower()
    exc_text = str(exc).lower()
    return (
        isinstance(exc, (asyncio.TimeoutError, TimeoutError))
        or "timeout" in exc_name
        or "timed out" in exc_text
        or "time limit" in exc_text
    )


def _parse_record(stdout: str, test_cases: dict[str, Any], test_idx: int) -> dict[str, Any]:
    marker_pos = stdout.rfind(_RECORD_MARKER)
    if marker_pos < 0:
        return _failure_record(
            test_cases,
            test_idx,
            code.ERROR_PREFIX + "AGS sandbox did not return a verifier record.",
        )
    record_line = stdout[marker_pos + len(_RECORD_MARKER):].splitlines()[0]
    try:
        record = json.loads(record_line)
    except Exception as exc:
        return _failure_record(
            test_cases,
            test_idx,
            code.ERROR_PREFIX + f"Failed to parse AGS verifier record: {type(exc).__name__}: {exc}",
        )
    return record


class AGSTacoVerifier:
    def __init__(
        self,
        *,
        template: str,
        sandbox_count: int,
        sandbox_timeout: int,
        command_timeout_buffer: float,
        python_bin: str = "python3",
        image: str | None = None,
        image_registry_type: str = "enterprise",
        cpus: int | None = None,
        memory_mb: int | None = None,
        allow_internet_access: bool = False,
        custom_config: dict[str, Any] | None = None,
    ) -> None:
        if not _HAS_E2B:
            raise RuntimeError("AGS backend requires the optional 'e2b' Python package to be installed.")
        missing = [name for name in ("E2B_API_KEY", "E2B_DOMAIN") if not os.environ.get(name)]
        if missing:
            raise RuntimeError("AGS backend requires " + ", ".join(missing) + " to be set.")
        if not template:
            raise RuntimeError("AGS backend requires TACO_AGS_TEMPLATE or AGS_TEMPLATE.")
        if sandbox_count <= 0:
            raise ValueError("sandbox_count must be positive")

        self.template = template
        self.sandbox_count = int(sandbox_count)
        self.sandbox_timeout = int(sandbox_timeout)
        self.command_timeout_buffer = float(command_timeout_buffer)
        self.python_bin = python_bin
        self.image = image
        self.image_registry_type = image_registry_type
        self.cpus = cpus
        self.memory_mb = memory_mb
        self.allow_internet_access = allow_internet_access
        self.custom_config = custom_config or {}
        self._sandboxes: list[Any] = []
        self._sandbox_queue: asyncio.Queue[Any] | None = None
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, name="taco-ags-loop", daemon=True)
        self._thread.start()
        self._run_sync(self._start_pool())

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _run_sync(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    def _metadata(self) -> dict[str, str]:
        metadata = {
            "environment_name": "ruc-sdpo-taco-verifier",
            "session_id": f"taco-verifier-{uuid.uuid4().hex[:8]}",
        }
        custom_config = dict(self.custom_config)
        if self.image:
            custom_config.setdefault("image", self.image)
            custom_config.setdefault("imageRegistryType", self.image_registry_type)
        resources: dict[str, str] = {}
        if self.cpus is not None:
            resources["cpu"] = str(self.cpus)
        if self.memory_mb is not None:
            resources["memory"] = f"{self.memory_mb // 1024}Gi" if self.memory_mb % 1024 == 0 else f"{self.memory_mb}Mi"
        if resources:
            custom_config["resources"] = resources
        if custom_config:
            metadata["x-custom-config"] = json.dumps(custom_config, ensure_ascii=False)
        return metadata

    async def _start_pool(self) -> None:
        self._sandbox_queue = asyncio.Queue()
        create_tasks = [
            AsyncSandbox.create(
                template=self.template,
                metadata=self._metadata(),
                timeout=self.sandbox_timeout,
                allow_internet_access=self.allow_internet_access,
            )
            for _ in range(self.sandbox_count)
        ]
        try:
            self._sandboxes = list(await asyncio.gather(*create_tasks))
        except BaseException:
            await self._close_pool()
            raise
        for sandbox in self._sandboxes:
            await self._sandbox_queue.put(sandbox)

    async def _close_pool(self) -> None:
        sandboxes = list(self._sandboxes)
        self._sandboxes = []
        for sandbox in sandboxes:
            try:
                await sandbox.kill()
            except Exception:
                pass

    def close(self) -> None:
        try:
            self._run_sync(self._close_pool())
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=5)

    async def _run_case(
        self,
        *,
        completion: str,
        test_cases: dict[str, Any],
        test_idx: int,
        timeout_sec: float,
    ) -> dict[str, Any]:
        assert self._sandbox_queue is not None
        sandbox = await self._sandbox_queue.get()
        remote_path = f"/tmp/taco_case_{uuid.uuid4().hex}.py"
        try:
            script = _build_case_script(
                completion=completion,
                test_type=test_cases["testtype"],
                fn_name=test_cases.get("fn_name") or "",
                context=test_cases.get("context") or "",
                test_input=test_cases["inputs"][test_idx],
                test_output=test_cases["outputs"][test_idx],
                test_idx=test_idx,
            )
            await sandbox.files.write(remote_path, script.encode("utf-8"), user="root")
            command_timeout = max(1, int(math.ceil(timeout_sec + self.command_timeout_buffer)))
            started = time.time()
            handle = await sandbox.commands.run(
                cmd=f"{shlex.quote(self.python_bin)} {shlex.quote(remote_path)}",
                background=True,
                timeout=command_timeout,
                user="root",
            )
            try:
                # AGS/E2B command timeout behavior differs across SDK/data-plane
                # versions. Add a client-side guard so an infinite loop cannot
                # occupy one sandbox until sandbox_timeout expires.
                wait_timeout = command_timeout + max(1.0, self.command_timeout_buffer)
                result = await asyncio.wait_for(handle.wait(), timeout=wait_timeout)
            except CommandExitException as exc:
                if _is_timeout_exception(exc):
                    return _timeout_record(test_cases, test_idx, time.time() - started)
                result = exc
            except Exception as exc:
                if _is_timeout_exception(exc):
                    return _timeout_record(test_cases, test_idx, time.time() - started)
                raise

            stdout = result.stdout or ""
            stderr = result.stderr or ""
            record = _parse_record(stdout, test_cases, test_idx)
            record["time"] = time.time() - started
            if not record["passed"] and record["actual"] == code.ERROR_PREFIX + "AGS sandbox did not return a verifier record.":
                detail = _short_text(stderr or stdout)
                if detail:
                    record["actual"] = code.ERROR_PREFIX + detail
            return record
        except Exception as exc:
            return _failure_record(
                test_cases,
                test_idx,
                code.ERROR_PREFIX + f"AGS sandbox failed: {type(exc).__name__}: {exc}",
            )
        finally:
            async def cleanup_and_return():
                try:
                    # Best effort: if the client-side wait timed out while the
                    # remote process kept running, kill that unique testcase script.
                    kill_cmd = f"pkill -f {shlex.quote(remote_path)} 2>/dev/null || true"
                    kill_handle = await sandbox.commands.run(cmd=kill_cmd, background=True, timeout=5, user="root")
                    await asyncio.wait_for(kill_handle.wait(), timeout=6)
                except Exception:
                    pass
                try:
                    cleanup = await sandbox.commands.run(
                        cmd=f"rm -f {shlex.quote(remote_path)}",
                        background=True,
                        timeout=5,
                        user="root",
                    )
                    await asyncio.wait_for(cleanup.wait(), timeout=6)
                except Exception:
                    pass
                await self._sandbox_queue.put(sandbox)

            await asyncio.shield(cleanup_and_return())

    async def _run_tests_async(
        self,
        *,
        test_cases: dict[str, Any],
        solution: str,
        max_test_cases: int | None,
    ) -> list[dict[str, Any]]:
        completion = code.extract_code(solution)
        if completion is None:
            return [
                {
                    "test_idx": 0,
                    "input": None,
                    "expected": None,
                    "actual": code.INCORRECT_FORMAT,
                    "passed": False,
                    "debug": "",
                    "time": float("inf"),
                }
            ]

        num_test_cases = min(max_test_cases, len(test_cases["inputs"])) if max_test_cases else len(test_cases["inputs"])
        timeout_per_test_case = (
            float(test_cases["time_limit"]) if test_cases.get("time_limit") is not None else code.DEFAULT_TIMEOUT
        )
        tasks = [
            asyncio.create_task(
                self._run_case(
                    completion=completion,
                    test_cases=test_cases,
                    test_idx=test_idx,
                    timeout_sec=timeout_per_test_case * code.TIMEOUT_SCALER,
                )
            )
            for test_idx in range(num_test_cases)
        ]
        if not tasks:
            return []
        return list(await asyncio.gather(*tasks))

    def compute_score(
        self,
        *,
        solution: str,
        ground_truth: str,
        extra_info: dict[str, Any],
        sparse_rewards: bool,
        max_test_cases: int | None,
    ) -> dict[str, Any]:
        split = extra_info["split"]
        was_truncated = extra_info.get("truncated", False)
        if split == "test":
            sparse_rewards = True
            max_test_cases = None

        try:
            test_cases = json.loads(ground_truth)
        except Exception:
            return {
                "score": 0.0,
                "acc": 0.0,
                "pred": "",
                "incorrect_format": 0,
                "error_in_test_cases": 1,
                "timed_out": 0,
                "truncated": 1 if was_truncated else 0,
                "truncated_and_missing_answer": 1 if was_truncated else 0,
                "feedback": "Failed to parse ground truth test cases.",
            }

        records = self._run_sync(
            self._run_tests_async(
                test_cases=test_cases,
                solution=solution,
                max_test_cases=max_test_cases,
            )
        )
        score = code.score_from_records(
            records,
            split=split,
            was_truncated=was_truncated,
            sparse_rewards=sparse_rewards,
        )
        score["verifier_error"] = int(
            any(
                isinstance(record.get("actual"), str)
                and (
                    "AGS sandbox failed:" in record["actual"]
                    or "AGS sandbox did not return a verifier record" in record["actual"]
                    or "Failed to parse AGS verifier record" in record["actual"]
                )
                for record in records
            )
        )
        return score


def parse_custom_config(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("AGS custom config must be a JSON object")
    return parsed
