
from __future__ import annotations

import argparse
import ast
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

Row = dict[str, Any]
Rows = list[Row]

APIGEN_FILE = "xlam_function_calling_60k.json"
TOOLACE_FILE = "data.json"


def json_dumps(obj: Any, *, sort_keys: bool = False) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=sort_keys, separators=(",", ":"))


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json_dumps(row) + "\n")


def safe_reason(exc: Exception) -> str:
    text = str(exc).split(": ", 1)[-1]
    text = re.sub(r"[^0-9a-zA-Z_]+", "_", text).strip("_")
    return text or "error"


def add_stat(stats: Counter[str], name: str) -> None:
    stats[name] += 1


def percentile(values: list[int], q: float) -> int:
    if not values:
        return 0
    values = sorted(values)
    return values[int(q * (len(values) - 1))]


def make_prompt(query: str, tools: list[Row]) -> str:
    tool_blocks = []
    for tool in tools:
        tool_blocks.append(
            "\n".join(
                [
                    f"Name: {tool['name']}",
                    f"Description: {tool.get('description', '')}",
                    f"Parameters: {json_dumps(tool.get('parameters', {}), sort_keys=True)}",
                    "Output: Successful response.",
                    " - Format: application/json",
                    " - Structure: Object",
                ]
            )
        )

    return "\n".join(
        [
            "Your task is to answer the user's question using available tools.",
            "You have access to the following tools:",
            "\n\n".join(tool_blocks),
            "",
            "Use exactly this format:",
            "Thought: think about what to do",
            "Action: one of the tool names above",
            "Action Input: a JSON object grounded in the user request or conversation context",
            "",
            "Begin!",
            f"Question: {query.strip()}",
        ]
    )


def calls_to_ground_truth(calls: list[Row]) -> str:
    return json_dumps(
        [
            {
                "Action": call["name"],
                "Action_Input": json_dumps(call.get("arguments", {}), sort_keys=True),
            }
            for call in calls
        ],
        sort_keys=True,
    )


def make_verl_row(
    *,
    data_source: str,
    prompt: str,
    ground_truth: str,
    source_dataset: str,
    source_id: str,
    source_group: str,
    split: str,
    num_tools: int,
    num_calls: int,
) -> Row:
    return {
        "data_source": data_source,
        "prompt": [{"role": "user", "content": prompt}],
        "ability": "tooluse",
        "reward_model": {"style": "rule", "ground_truth": ground_truth},
        "extra_info": {
            "split": split,
            "index": source_id,
            "source_dataset": source_dataset,
            "source_id": source_id,
            "source_group": source_group,
            "num_tools": num_tools,
            "num_calls": num_calls,
        },
    }


def load_json_value(value: Any, field: str, source_id: str) -> Any:
    if value is None:
        raise ValueError(f"{source_id}: missing {field}")
    if isinstance(value, str):
        return json.loads(value)
    return value


def validate_tools(tools: Any, source_id: str) -> list[Row]:
    if not isinstance(tools, list) or not tools:
        raise ValueError(f"{source_id}: tools must be a non-empty list")

    out = []
    seen = set()
    for idx, tool in enumerate(tools):
        if not isinstance(tool, dict):
            raise ValueError(f"{source_id}: tool {idx} must be an object")
        name = str(tool.get("name", "")).strip()
        if not name:
            raise ValueError(f"{source_id}: tool {idx} missing name")
        if name in seen:
            raise ValueError(f"{source_id}: duplicate tool name {name}")
        seen.add(name)
        out.append(
            {
                "name": name,
                "description": str(tool.get("description", "")).strip(),
                "parameters": tool.get("parameters", {}) or {},
            }
        )
    return out


def validate_calls(calls: Any, source_id: str, tool_names: set[str]) -> list[Row]:
    if not isinstance(calls, list) or not calls:
        raise ValueError(f"{source_id}: answers must be a non-empty list")

    out = []
    for idx, call in enumerate(calls):
        if not isinstance(call, dict):
            raise ValueError(f"{source_id}: answer {idx} must be an object")
        name = str(call.get("name", "")).strip()
        if not name:
            raise ValueError(f"{source_id}: answer {idx} missing name")
        if name not in tool_names:
            raise ValueError(f"{source_id}: action_not_in_tools")
        arguments = call.get("arguments", {})
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            raise ValueError(f"{source_id}: answer {idx} arguments must be an object")
        out.append({"name": name, "arguments": arguments})
    return out


def split_rows(
    rows: Rows,
    *,
    val_ratio: float,
    seed: int,
    group_key: str | None = None,
) -> tuple[Rows, Rows]:
    if not 0 <= val_ratio < 1:
        raise ValueError(f"val_ratio must be in [0, 1), got {val_ratio}")

    if group_key:
        units = sorted({row["extra_info"][group_key] for row in rows})
    else:
        units = list(range(len(rows)))

    random.Random(seed).shuffle(units)
    val_size = int(round(len(units) * val_ratio))
    if val_ratio > 0 and len(units) > 1:
        val_size = max(1, val_size)
    val_units = set(units[:val_size])

    train, test = [], []
    for idx, row in enumerate(rows):
        key = row["extra_info"][group_key] if group_key else idx
        split = "test" if key in val_units else "train"

        out = dict(row)
        out["extra_info"] = dict(row["extra_info"])
        out["extra_info"]["split"] = split
        (test if split == "test" else train).append(out)

    return train, test


def lint_rows(rows: Rows, dataset_name: str) -> Counter[str]:
    stats: Counter[str] = Counter()
    lengths, num_calls = [], []
    seen_ids = set()

    for row in rows:
        info = row.get("extra_info", {})
        source_id = info.get("source_id")
        if source_id in seen_ids:
            add_stat(stats, "lint/duplicate_source_id")
        seen_ids.add(source_id)

        prompt = row["prompt"][0]["content"]
        lengths.append(len(prompt))

        try:
            gt_calls = json.loads(row["reward_model"]["ground_truth"])
        except Exception:
            add_stat(stats, "lint/bad_ground_truth_json")
            continue

        if not isinstance(gt_calls, list) or not gt_calls:
            add_stat(stats, "lint/empty_ground_truth")
            continue

        num_calls.append(len(gt_calls))
        for call in gt_calls:
            action = call.get("Action")
            action_input = call.get("Action_Input")
            if f"Name: {action}" not in prompt:
                add_stat(stats, "lint/action_not_in_prompt")
            try:
                parsed = json.loads(action_input) if isinstance(action_input, str) else action_input
            except Exception:
                add_stat(stats, "lint/bad_action_input_json")
                continue
            if not isinstance(parsed, dict):
                add_stat(stats, "lint/action_input_not_dict")

    if lengths:
        stats["lint/prompt_chars_min"] = min(lengths)
        stats["lint/prompt_chars_p50"] = percentile(lengths, 0.50)
        stats["lint/prompt_chars_p95"] = percentile(lengths, 0.95)
        stats["lint/prompt_chars_max"] = max(lengths)
    if num_calls:
        stats["lint/num_calls_min"] = min(num_calls)
        stats["lint/num_calls_p50"] = percentile(num_calls, 0.50)
        stats["lint/num_calls_max"] = max(num_calls)

    print_counter(f"{dataset_name} lint", stats)
    return stats


def print_counter(title: str, stats: Counter[str]) -> None:
    print(f"{title}:")
    for key in sorted(stats):
        print(f"  {key}={stats[key]}")


def finish_dataset(
    root: Path,
    dataset_name: str,
    rows: Rows,
    stats: Counter[str],
    val_ratio: float,
    seed: int,
    group_split: bool,
) -> None:
    train, test = split_rows(
        rows,
        val_ratio=val_ratio,
        seed=seed,
        group_key="source_group" if group_split else None,
    )

    if group_split:
        train_groups = {row["extra_info"]["source_group"] for row in train}
        test_groups = {row["extra_info"]["source_group"] for row in test}
        overlap = train_groups & test_groups
        if overlap:
            raise RuntimeError(f"{dataset_name}: group split leaked {len(overlap)} groups")

    lint_stats = lint_rows(rows, dataset_name)
    write_jsonl(root / "train.jsonl", train)
    write_jsonl(root / "test.jsonl", test)
    write_json(
        root / "conversion_meta.json",
        {
            "dataset": dataset_name,
            "num_rows": len(rows),
            "num_train": len(train),
            "num_test": len(test),
            "val_ratio": val_ratio,
            "seed": seed,
            "group_split": group_split,
            "stats": dict(sorted(stats.items())),
            "lint": dict(sorted(lint_stats.items())),
        },
    )

    skipped = sum(
        v
        for k, v in stats.items()
        if k.startswith("skip/") or k.startswith("skip_item/") or k.startswith("skip_turn/")
    )
    print(f"{dataset_name}: converted={len(rows)} train={len(train)} test={len(test)} skipped={skipped}")
    print_counter(f"{dataset_name} stats", stats)


def convert_apigen(root: Path, val_ratio: float, seed: int) -> None:
    raw_rows = read_json(root / APIGEN_FILE)
    if not isinstance(raw_rows, list):
        raise ValueError(f"{root / APIGEN_FILE} must contain a JSON list")

    rows: Rows = []
    stats: Counter[str] = Counter()

    for idx, raw in enumerate(raw_rows):
        source_id = f"apigen-{raw.get('id', idx)}"
        try:
            query = str(raw.get("query", "")).strip()
            if not query:
                raise ValueError(f"{source_id}: missing query")

            tools = validate_tools(load_json_value(raw.get("tools"), "tools", source_id), source_id)
            calls = validate_calls(
                load_json_value(raw.get("answers"), "answers", source_id),
                source_id,
                {tool["name"] for tool in tools},
            )

            rows.append(
                make_verl_row(
                    data_source="tooluse_apigen",
                    prompt=make_prompt(query, tools),
                    ground_truth=calls_to_ground_truth(calls),
                    source_dataset="apigen",
                    source_id=source_id,
                    source_group=source_id,
                    split="train",
                    num_tools=len(tools),
                    num_calls=len(calls),
                )
            )
            add_stat(stats, "keep/rows")
        except json.JSONDecodeError:
            add_stat(stats, "skip/json_parse_error")
        except Exception as exc:
            add_stat(stats, f"skip/{safe_reason(exc)}")

    finish_dataset(root, "apigen", rows, stats, val_ratio, seed, group_split=False)


def find_matching(text: str, start: int, open_ch: str, close_ch: str) -> int | None:
    depth = 0
    quote = ""
    escape = False

    for idx in range(start, len(text)):
        ch = text[idx]

        if quote:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == quote:
                quote = ""
            continue

        if ch in {"'", '"'}:
            quote = ch
        elif ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return idx

    return None


def extract_first_json_array(text: str, start_at: int = 0) -> list[Any] | None:
    start = text.find("[", start_at)
    if start < 0:
        return None
    end = find_matching(text, start, "[", "]")
    if end is None:
        return None
    value = json.loads(text[start : end + 1])
    return value if isinstance(value, list) else None


def extract_toolace_tools(system: str) -> list[Row] | None:
    marker = "Here is a list of functions in JSON format that you can invoke:"
    marker_idx = system.find(marker)
    start_at = marker_idx + len(marker) if marker_idx >= 0 else 0
    tools = extract_first_json_array(system, start_at=start_at)
    return tools if isinstance(tools, list) else None


def split_top_level(text: str, delimiter: str = ",") -> list[str]:
    parts, start = [], 0
    paren = bracket = brace = 0
    quote = ""
    escape = False

    for idx, ch in enumerate(text):
        if quote:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == quote:
                quote = ""
            continue

        if ch in {"'", '"'}:
            quote = ch
        elif ch == "(":
            paren += 1
        elif ch == ")":
            paren -= 1
        elif ch == "[":
            bracket += 1
        elif ch == "]":
            bracket -= 1
        elif ch == "{":
            brace += 1
        elif ch == "}":
            brace -= 1
        elif ch == delimiter and paren == bracket == brace == 0:
            part = text[start:idx].strip()
            if part:
                parts.append(part)
            start = idx + 1

    tail = text[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def parse_value(text: str) -> Any:
    value = text.strip()
    if value in {"true", "false", "null"}:
        return {"true": True, "false": False, "null": None}[value]
    try:
        return ast.literal_eval(value)
    except Exception:
        try:
            return json.loads(value)
        except Exception:
            return value


def parse_toolace_calls(text: str) -> list[Row] | None:
    text = text.strip()
    if not text.startswith("["):
        return None

    end = find_matching(text, 0, "[", "]")
    if end is None or text[end + 1 :].strip():
        return None

    inner = text[1:end].strip()
    if not inner:
        return None

    calls = []
    for call_text in split_top_level(inner):
        open_idx = call_text.find("(")
        if open_idx < 1:
            return None
        close_idx = find_matching(call_text, open_idx, "(", ")")
        if close_idx is None or call_text[close_idx + 1 :].strip():
            return None

        name = call_text[:open_idx].strip()
        args_text = call_text[open_idx + 1 : close_idx].strip()
        arguments: Row = {}

        if args_text:
            for arg_idx, arg in enumerate(split_top_level(args_text)):
                if "=" in arg:
                    key, value = arg.split("=", 1)
                    key = key.strip()
                else:
                    key, value = f"arg{arg_idx}", arg
                if not key:
                    return None
                arguments[key] = parse_value(value)

        calls.append({"name": name, "arguments": arguments})

    return calls


def render_toolace_context(conversations: list[Row]) -> str:
    if len(conversations) == 1 and conversations[0].get("from") == "user":
        return str(conversations[0].get("value", "")).strip()

    role_names = {"user": "User", "assistant": "Assistant", "tool": "Tool"}
    lines = ["Conversation so far:"]
    for turn in conversations:
        role = role_names.get(turn.get("from"), str(turn.get("from", "unknown")).title())
        lines.append(f"{role}: {turn.get('value', '')}")
    lines.append("Respond with the next tool call for the latest user request.")
    return "\n".join(lines)


def convert_toolace(root: Path, val_ratio: float, seed: int) -> None:
    raw_rows = read_json(root / TOOLACE_FILE)
    if not isinstance(raw_rows, list):
        raise ValueError(f"{root / TOOLACE_FILE} must contain a JSON list")

    rows: Rows = []
    stats: Counter[str] = Counter()

    for item_idx, raw in enumerate(raw_rows):
        source_group = f"toolace-{item_idx}"

        try:
            tools = validate_tools(extract_toolace_tools(str(raw.get("system", ""))), source_group)
            tool_names = {tool["name"] for tool in tools}
        except Exception:
            add_stat(stats, "skip_item/tool_parse_error")
            continue

        conversations = raw.get("conversations", [])
        if not isinstance(conversations, list):
            add_stat(stats, "skip_item/bad_conversations")
            continue

        kept = 0
        for turn_idx, turn in enumerate(conversations):
            if not isinstance(turn, dict) or turn.get("from") != "assistant":
                continue
            add_stat(stats, "turn/assistant_total")

            calls = parse_toolace_calls(str(turn.get("value", "")))
            if not calls:
                if str(turn.get("value", "")).strip().startswith("["):
                    add_stat(stats, "turn/parser_failed_toollike")
                continue
            add_stat(stats, "turn/assistant_call_total")

            context = conversations[:turn_idx]
            # ToolACE 训练目标是“根据最新 user 请求预测下一次 tool call”。
            # 只保留 assistant tool-call 前一轮正好是 user 的样本，和旧 SDPO 目录的数据构造保持一致。
            if not context or context[-1].get("from") != "user":
                prev_role = "none" if not context else str(context[-1].get("from", "unknown"))
                add_stat(stats, f"skip_turn/prev_{prev_role}")
                continue

            source_id = f"{source_group}-turn{turn_idx}"
            try:
                calls = validate_calls(calls, source_id, tool_names)
            except Exception as exc:
                add_stat(stats, f"skip_turn/{safe_reason(exc)}")
                continue

            rows.append(
                make_verl_row(
                    data_source="tooluse_toolace",
                    prompt=make_prompt(render_toolace_context(context), tools),
                    ground_truth=calls_to_ground_truth(calls),
                    source_dataset="toolace",
                    source_id=source_id,
                    source_group=source_group,
                    split="train",
                    num_tools=len(tools),
                    num_calls=len(calls),
                )
            )
            kept += 1
            add_stat(stats, "keep/rows")

        if kept == 0:
            add_stat(stats, "skip_item/no_kept_turns")

    finish_dataset(root, "toolace", rows, stats, val_ratio, seed, group_split=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert APIGen and ToolACE into verl JSONL schema.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["all"],
        choices=["all", "apigen", "toolace"],
        help="Datasets to convert.",
    )
    parser.add_argument("--base-dir", type=Path, default=Path("datasets"), help="Directory containing dataset folders.")
    parser.add_argument("--val-ratio", type=float, default=0.02, help="Validation split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic splitting.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = {"apigen", "toolace"} if "all" in args.datasets else set(args.datasets)

    if "apigen" in selected:
        convert_apigen(args.base_dir / "apigen", args.val_ratio, args.seed)
    if "toolace" in selected:
        convert_toolace(args.base_dir / "toolace", args.val_ratio, args.seed)


if __name__ == "__main__":
    main()
