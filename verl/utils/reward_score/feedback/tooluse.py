from __future__ import annotations

import json
import re
from collections import Counter


_ACTION_RE = re.compile(r"(^|\n)Action:[^\S\r\n]*([^\r\n]+)")
_ACTION_INPUT_RE = re.compile(r"(^|\n)Action Input:[^\S\r\n]*", re.MULTILINE)


def extract_actions(text: str) -> list[str]:
    """Extract all action names after 'Action:' occurrences."""
    return [match.group(2).strip() for match in _ACTION_RE.finditer(text) if match.group(2).strip()]


def extract_action_inputs(text: str) -> dict:
    """Extract and merge all JSON blocks following 'Action Input:'."""
    calls, valid, _ = parse_tooluse_tool_calls(text)
    if valid:
        return merge_action_inputs([call["Action_Input"] for call in calls])

    json_blocks = re.findall(r'Action Input:\s*({.*?})', text, re.DOTALL)

    combined_dict = {}
    for block in json_blocks:
        try:
            parsed = json.loads(block)
            combined_dict.update(parsed)
        except json.JSONDecodeError:
            pass

    return combined_dict


def merge_action_inputs(action_inputs_list: list[dict]) -> dict:
    """Merge a list of action input dicts into a single dict."""
    combined = {}
    for d in action_inputs_list:
        if d:
            combined.update(d)
    return combined


def _call_counter(calls: list[dict]) -> Counter:
    serialized_calls = []
    for call in calls:
        serialized_calls.append(
            json.dumps(
                {
                    "Action": call["Action"],
                    "Action_Input": call["Action_Input"],
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    return Counter(serialized_calls)


def is_correct_format(text: str) -> bool:
    """Check if the text contains the expected Action/Action Input format."""
    pattern = re.compile(r"Action:.*?\nAction Input:.*?", re.DOTALL)
    return pattern.search(text) is not None


def _find_json_object_end(text: str, start: int) -> int | None:
    """Return the exclusive end offset of a JSON object starting at start."""
    if start >= len(text) or text[start] != "{":
        return None
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return idx + 1
    return None


def parse_tooluse_tool_calls(text: str) -> tuple[list[dict], bool, list[tuple[int, int]]]:
    """Parse executable ToolUse Action/Action Input pairs.

    The parser keeps the full action line, so tool names such as
    "Market Trends API" are not truncated to the first word. It also returns
    character spans used by optional action-span masking code.
    """
    action_matches = list(_ACTION_RE.finditer(text))
    if not action_matches:
        return [], False, []

    calls: list[dict] = []
    spans: list[tuple[int, int]] = []
    for idx, action_match in enumerate(action_matches):
        action = action_match.group(2).strip()
        if not action:
            return [], False, []

        segment_end = action_matches[idx + 1].start() if idx + 1 < len(action_matches) else len(text)
        input_match = _ACTION_INPUT_RE.search(text, action_match.end(), segment_end)
        if input_match is None:
            return [], False, []

        json_start = input_match.end()
        while json_start < segment_end and text[json_start].isspace():
            json_start += 1
        json_end = _find_json_object_end(text, json_start)
        if json_end is None or json_end > segment_end:
            return [], False, []

        try:
            action_input = json.loads(text[json_start:json_end])
        except json.JSONDecodeError:
            return [], False, []
        if not isinstance(action_input, dict):
            return [], False, []

        calls.append({"Action": action, "Action_Input": action_input})
        spans.append((action_match.start(0) + len(action_match.group(1)), json_end))

    return calls, True, spans


def canonicalize_tooluse_solution(solution: str) -> str | None:
    """Return a canonical ToolUse solution or None if parsing fails."""
    calls, valid, _ = parse_tooluse_tool_calls(solution)
    if not valid or not calls:
        return None
    chunks = []
    for call in calls:
        chunks.append(f"Action: {call['Action']}")
        chunks.append(
            "Action Input: "
            + json.dumps(call["Action_Input"], ensure_ascii=False, sort_keys=True)
        )
    return "\n".join(chunks)


def get_tooluse_action_spans(solution: str) -> list[tuple[int, int]]:
    """Return valid executable ToolUse spans; empty if parsing fails."""
    _, valid, spans = parse_tooluse_tool_calls(solution)
    return spans if valid else []


def compute_score(solution: str, ground_truth: str) -> dict:
    """
    Compute score for tooluse task.
    
    Args:
        solution: The model's response text
        ground_truth: JSON string containing list of dicts with 'Action' and 'Action_Input' keys
                      e.g., '[{"Action": "search", "Action_Input": "{\"query\": \"test\"}"}]'
    
    Returns:
        dict with score, acc, pred, incorrect_format, feedback
    """
    # Parse ground truth
    try:
        gt_list = json.loads(ground_truth)
    except json.JSONDecodeError:
        # If ground_truth is already a list (passed directly), handle that case
        if isinstance(ground_truth, list):
            gt_list = ground_truth
        else:
            return {
                "score": 0.0,
                "acc": 0.0,
                "pred": "",
                "incorrect_format": 1,
                "feedback": "Failed to parse ground truth JSON",
            }
    # Extract ground truth calls.
    gt_calls = []
    for item in gt_list:
        try:
            parsed_input = json.loads(item['Action_Input']) if isinstance(item['Action_Input'], str) else item['Action_Input']
            if not isinstance(parsed_input, dict):
                parsed_input = {}
            gt_calls.append({"Action": item["Action"], "Action_Input": parsed_input})
        except (json.JSONDecodeError, KeyError):
            gt_calls.append({"Action": item.get("Action", ""), "Action_Input": {}})
    gt_actions = [item["Action"] for item in gt_calls]
    gt_action_inputs_list = [item["Action_Input"] for item in gt_calls]
    gt_action_inputs = merge_action_inputs(gt_action_inputs_list)

    # Extract predicted actions and action inputs from solution
    pred_calls, pred_valid, _ = parse_tooluse_tool_calls(solution)
    if pred_valid:
        pred_actions = [item["Action"] for item in pred_calls]
        pred_action_inputs = merge_action_inputs([item["Action_Input"] for item in pred_calls])
    else:
        pred_actions = extract_actions(solution)
        pred_action_inputs = extract_action_inputs(solution)

    # Check correctness
    actions_correct = Counter(pred_actions) == Counter(gt_actions)
    action_inputs_correct = pred_action_inputs == gt_action_inputs
    calls_correct = pred_valid and _call_counter(pred_calls) == _call_counter(gt_calls)

    # Both must be correct for full score
    is_correct = calls_correct
    reward = 1.0 if is_correct else 0.0

    # Check format
    correct_format = pred_valid or is_correct_format(solution)

    # Build prediction string for logging
    prediction = f"Actions: {pred_actions}, Inputs: {pred_action_inputs}"

    # Build feedback
    feedback_parts = []
    if not actions_correct:
        feedback_parts.append(f"Actions mismatch: predicted {pred_actions}, expected {gt_actions}")
    if not action_inputs_correct:
        feedback_parts.append(f"Action inputs mismatch: predicted {pred_action_inputs}, expected {gt_action_inputs}")
    if actions_correct and action_inputs_correct and not calls_correct:
        feedback_parts.append(f"Tool calls mismatch: predicted {pred_calls}, expected {gt_calls}")

    if len(feedback_parts) == 0:
        feedback = "" # no feedback means correct
    else:
        feedback = "; ".join(feedback_parts)

    return {
        "score": reward,
        "acc": reward,
        "pred": prediction,
        "incorrect_format": 0 if correct_format else 1,
        "feedback": feedback,
    }
