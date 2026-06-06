from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections import defaultdict
from typing import Any

import torch

from verl import DataProto
from verl.utils.reward_score.feedback import code as feedback_code
from verl.utils.reward_score.feedback import compute_score as feedback_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager, RawRewardFn


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    return {}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


@register("taco_batch")
class TacoBatchRewardManager(AbstractRewardManager):
    """Batch reward manager that delegates TACO scoring to a long-running verifier service."""

    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score: RawRewardFn | None = None,
        reward_fn_key="data_source",
        server_url: str = "http://127.0.0.1:18080",
        request_timeout_sec: float = 3600.0,
        max_retries: int = 0,
        retry_sleep_sec: float = 2.0,
        max_test_cases: int | None = None,
        local_fallback: bool = False,
        **reward_kwargs,
    ):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or feedback_compute_score
        self.reward_fn_key = reward_fn_key
        self.server_url = server_url.rstrip("/")
        self.request_timeout_sec = float(request_timeout_sec)
        self.max_retries = int(max_retries)
        self.retry_sleep_sec = float(retry_sleep_sec)
        self.max_test_cases = None if max_test_cases in (None, "", "null") else int(max_test_cases)
        self.local_fallback = bool(local_fallback)
        self.reward_kwargs = reward_kwargs

    def _post_batch(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        payload: dict[str, Any] = {"items": items, "sparse_rewards": True}
        if self.max_test_cases is not None:
            payload["max_test_cases"] = self.max_test_cases

        body = json.dumps(_json_safe(payload)).encode("utf-8")
        request = urllib.request.Request(
            f"{self.server_url}/verify",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.request_timeout_sec) as response:
                    data = json.loads(response.read().decode("utf-8"))
                if not data.get("ok", False):
                    raise RuntimeError(data.get("error", "TACO verifier returned ok=false"))
                results = data.get("results")
                if not isinstance(results, list) or len(results) != len(items):
                    raise RuntimeError("TACO verifier returned malformed results")
                return [result["score"] for result in results]
            except (urllib.error.URLError, TimeoutError, RuntimeError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(self.retry_sleep_sec)

        assert last_error is not None
        raise RuntimeError(
            f"TACO batch verifier request failed after {self.max_retries + 1} attempt(s): {last_error}. "
            "Start the service, or set reward_model.reward_kwargs.local_fallback=True "
            "to fall back to the local scorer."
        )

    def _score_local(self, item: dict[str, Any]) -> dict[str, Any]:
        if str(item["data_source"]).lower() == "taco":
            result = feedback_code.compute_score(
                solution=item["solution"],
                ground_truth=item["ground_truth"],
                extra_info=item["extra_info"],
                sparse_rewards=True,
                max_test_cases=self.max_test_cases,
            )
            result.setdefault("verifier_error", 0)
            return result
        return self.compute_score(
            data_source=item["data_source"],
            solution_str=item["solution"],
            ground_truth=item["ground_truth"],
            extra_info=item["extra_info"],
        )

    def _build_items(self, data: DataProto) -> tuple[list[dict[str, Any]], list[int], list[int], list[str]]:
        prompt_ids = data.batch["prompts"]
        response_ids = data.batch["responses"]
        attention_mask = data.batch["attention_mask"]
        prompt_len = prompt_ids.shape[-1]
        valid_response_lengths = attention_mask[:, prompt_len:].sum(dim=-1)
        data_sources = data.non_tensor_batch[self.reward_fn_key]
        extras = data.non_tensor_batch.get("extra_info", [{} for _ in range(len(data))])
        rollout_reward_scores = data.non_tensor_batch.get("reward_scores", [{} for _ in range(len(data))])
        num_turns = data.non_tensor_batch.get("__num_turns__", [None for _ in range(len(data))])

        items: list[dict[str, Any]] = []
        taco_indices: list[int] = []
        response_lengths: list[int] = []
        response_texts: list[str] = []
        for i in range(len(data)):
            valid_len = int(valid_response_lengths[i].item())
            valid_response_ids = response_ids[i][:valid_len]
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            response_texts.append(response_str)
            response_lengths.append(valid_len)

            data_source = str(data_sources[i])
            extra_info = _as_dict(extras[i])
            extra_info["num_turns"] = num_turns[i] if i < len(num_turns) else None
            extra_info["rollout_reward_scores"] = rollout_reward_scores[i] if i < len(rollout_reward_scores) else {}
            extra_info["truncated"] = not (valid_response_ids == self.tokenizer.eos_token_id).any().item()

            item = {
                "id": i,
                "data_source": data_source,
                "solution": response_str,
                "ground_truth": data[i].non_tensor_batch["reward_model"]["ground_truth"],
                "extra_info": extra_info,
            }
            items.append(item)
            if data_source.lower() == "taco":
                taco_indices.append(i)

        return items, taco_indices, response_lengths, response_texts

    def __call__(self, data: DataProto, return_dict: bool = True) -> torch.Tensor | dict[str, Any]:
        reward_from_rm_scores = self._extract_reward_from_rm_scores(data, return_dict)
        if reward_from_rm_scores is not None:
            return reward_from_rm_scores

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        prompt_ids = data.batch["prompts"]
        data_sources = data.non_tensor_batch[self.reward_fn_key]
        items, taco_indices, response_lengths, response_texts = self._build_items(data)

        scores: list[Any] = [None] * len(data)
        if taco_indices:
            taco_items = [items[i] for i in taco_indices]
            try:
                taco_scores = self._post_batch(taco_items)
            except Exception:
                if not self.local_fallback:
                    raise
                taco_scores = [self._score_local(item) for item in taco_items]
            for idx, score in zip(taco_indices, taco_scores):
                scores[idx] = score

        for i, item in enumerate(items):
            if scores[i] is None:
                scores[i] = self._score_local(item)

        extra_info_keys: set[str] = set()
        for i, score in enumerate(scores):
            data_source = str(data_sources[i])
            if isinstance(score, dict) and data_source.lower() == "taco":
                score.setdefault("verifier_error", 0)
            if isinstance(score, dict):
                extra_info_keys.update(score.keys())

        rewards = []
        already_printed: dict[str, int] = {}
        for i, score in enumerate(scores):
            data_source = str(data_sources[i])
            if isinstance(score, dict):
                reward = score["score"]
                for key in extra_info_keys:
                    reward_extra_info[key].append(score.get(key))
            else:
                reward = score
                for key in extra_info_keys:
                    reward_extra_info[key].append(None)

            rewards.append(float(reward))
            reward_idx = max(response_lengths[i] - 1, 0)
            reward_tensor[i, reward_idx] = float(reward)

            if already_printed.get(data_source, 0) < self.num_examine:
                prompt_str = self.tokenizer.decode(data.batch["prompts"][i], skip_special_tokens=True)
                print("[prompt]", prompt_str)
                print("[response]", response_texts[i])
                print("[ground_truth]", data[i].non_tensor_batch["reward_model"].get("ground_truth", None))
                print("[score]", score)
                already_printed[data_source] = already_printed.get(data_source, 0) + 1

        data.batch["acc"] = torch.tensor(rewards, dtype=torch.float32, device=prompt_ids.device)

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
        return reward_tensor
