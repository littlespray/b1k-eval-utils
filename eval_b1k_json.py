#!/usr/bin/env python3
"""
统计 Hugging Face dataset 仓库中所有 exp 的逐轮成功率。

目录结构约定：task/exp/metrics/*.json
示例文件：
  turning-on-radio/turning_on_radio_xxx_20260216085234/metrics/turning_on_radio_109_0.json

规则：
1. 对同一个 episode id，收集所有 exp 中对应 json；
2. 按 exp 目录末尾时间戳升序排序；
3. 第 i 轮平均值 = 所有 episode 各自第 i 个结果的 q_score.final 平均；
4. 当任一 episode 不足第 i 个结果时停止；
5. 输出每轮平均值，以及这些平均值的平均值。
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from statistics import mean
from typing import DefaultDict, Dict, List, Tuple

from huggingface_hub import HfApi, hf_hub_download


EXP_TIMESTAMP_RE = re.compile(r"_(\d{14})$")
EPISODE_ID_RE = re.compile(r"_(\d+)_\d+\.json$")
METRICS_JSON_RE = re.compile(r"^[^/]+/[^/]+/metrics/[^/]+\.json$")


@dataclass(frozen=True)
class EpisodeEval:
    task: str
    episode_id: str
    timestamp: str
    q_score: float
    wall_time: float | None
    path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="计算 HF dataset 仓库中逐轮 episode 成功率均值。"
    )
    parser.add_argument(
        "repo_id",
        help="HF dataset 仓库名，例如 shangkuns/pt12-3w-comet2-20260204022222",
    )
    parser.add_argument(
        "--repo-type",
        default="dataset",
        choices=["dataset", "model", "space"],
        help="HF 仓库类型，默认 dataset。",
    )
    return parser.parse_args()


def extract_timestamp_from_exp(exp_name: str) -> str | None:
    match = EXP_TIMESTAMP_RE.search(exp_name)
    return match.group(1) if match else None


def extract_episode_id_from_filename(filename: str) -> str | None:
    match = EPISODE_ID_RE.search(filename)
    return match.group(1) if match else None


def collect_episode_evals(
    repo_id: str, repo_type: str
) -> Dict[Tuple[str, str], List[EpisodeEval]]:
    api = HfApi()
    repo_files = api.list_repo_files(repo_id=repo_id, repo_type=repo_type)

    episode_to_evals: DefaultDict[Tuple[str, str], List[EpisodeEval]] = defaultdict(list)

    for path in repo_files:
        if not METRICS_JSON_RE.match(path):
            continue

        parts = path.split("/")
        # task / exp / metrics / xxx.json
        if len(parts) != 4:
            continue

        task_name = parts[0]
        exp_name = parts[1]
        file_name = parts[3]

        timestamp = extract_timestamp_from_exp(exp_name)
        if timestamp is None:
            continue

        episode_id = extract_episode_id_from_filename(file_name)
        if episode_id is None:
            continue

        local_path = hf_hub_download(repo_id=repo_id, repo_type=repo_type, filename=path)
        with open(local_path, "r", encoding="utf-8") as f:
            content = json.load(f)

        q_score = float(content["q_score"]["final"])
        wall_time = None
        if isinstance(content.get("time"), dict) and "wall_time" in content["time"]:
            wall_time = float(content["time"]["wall_time"])

        episode_to_evals[(task_name, episode_id)].append(
            EpisodeEval(
                task=task_name,
                episode_id=episode_id,
                timestamp=timestamp,
                q_score=q_score,
                wall_time=wall_time,
                path=path,
            )
        )

    return dict(episode_to_evals)


def compute_round_means(
    episode_to_evals: Dict[Tuple[str, str], List[EpisodeEval]]
) -> Tuple[List[float], List[EpisodeEval]]:
    if not episode_to_evals:
        return [], []

    for evals in episode_to_evals.values():
        evals.sort(key=lambda x: x.timestamp)

    max_rounds = min(len(evals) for evals in episode_to_evals.values())
    if max_rounds == 0:
        return [], []

    round_means: List[float] = []
    used_evals: List[EpisodeEval] = []
    episode_keys = sorted(episode_to_evals.keys(), key=lambda x: (x[0], int(x[1])))
    for round_idx in range(max_rounds):
        this_round = [episode_to_evals[key][round_idx] for key in episode_keys]
        values = [item.q_score for item in this_round]
        used_evals.extend(this_round)
        round_means.append(mean(values))
    return round_means, used_evals


def compute_wall_time_stats(used_evals: List[EpisodeEval]) -> Tuple[Dict[str, float], float | None]:
    task_to_times: DefaultDict[str, List[float]] = defaultdict(list)
    for item in used_evals:
        if item.wall_time is not None:
            task_to_times[item.task].append(item.wall_time)

    task_means = {task: mean(times) for task, times in task_to_times.items() if times}
    if not task_means:
        return task_means, None
    overall_task_mean = mean(task_means.values())
    return task_means, overall_task_mean


def main() -> None:
    args = parse_args()

    episode_to_evals = collect_episode_evals(args.repo_id, args.repo_type)
    if not episode_to_evals:
        print("未找到符合 task/exp/metrics/*.json 结构且可解析 episode/timestamp 的文件。")
        return

    round_means, used_evals = compute_round_means(episode_to_evals)
    if not round_means:
        print("没有可计算的轮次（可能某些 episode 无有效结果）。")
        return

    print(f"repo: {args.repo_id}")
    print(f"episode 数量: {len(episode_to_evals)}")
    print(f"可计算轮次: {len(round_means)}")
    for idx, value in enumerate(round_means, start=1):
        print(f"第{idx}轮平均成功率: {value:.6f}")

    overall = mean(round_means)
    print(f"各轮平均值的平均值: {overall:.6f}")

    task_means, overall_task_mean = compute_wall_time_stats(used_evals)
    if not task_means:
        print("参与 q-score 统计的样本中未找到 time.wall_time，跳过用时统计。")
        return

    print("各 task 平均 episode 用时:")
    for task in sorted(task_means.keys()):
        print(f"- {task}: {task_means[task]:.6f}")
    print(f"所有 task 平均用时: {overall_task_mean:.6f}")


if __name__ == "__main__":
    main()
