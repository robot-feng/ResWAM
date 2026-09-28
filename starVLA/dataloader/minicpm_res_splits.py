# Copyright 2026 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""Frozen, source-demonstration-grouped splits for MiniCPM residual studies."""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


SPLIT_NAMES = ("train", "validation", "test")
DATASET_META_FILES = (
    "info.json",
    "episodes.jsonl",
    "tasks.jsonl",
    "modality.json",
    "downsample.json",
    "stats_gr00t.json",
)
LEGACY_TRAIN_EPISODE_IDS = ("0", "1", "3", "6", "7")
LEGACY_VALIDATION_EPISODE_IDS = ("2", "4", "10", "21", "422")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: dict[str, Any]) -> str:
    unsigned = dict(value)
    unsigned.pop("manifest_sha256", None)
    encoded = json.dumps(
        unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_annotation_rows(path: str | Path) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    records: dict[str, dict[str, Any]] = {}
    excluded: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid terminal annotation JSON at line {line_no}") from exc
            if "episode_id" not in row:
                raise ValueError(f"terminal annotation line {line_no} is missing episode_id")
            episode_id = str(row["episode_id"])
            if episode_id in records or any(item["episode_id"] == episode_id for item in excluded):
                raise ValueError(f"duplicate terminal annotation for episode {episode_id}")
            if row.get("is_success") is not True:
                excluded.append(
                    {
                        "episode_id": episode_id,
                        "task": row.get("task"),
                        "reason": row.get("reason", "not explicitly labeled successful"),
                    }
                )
                continue
            for key in ("task", "terminal_step", "provenance"):
                if key not in row:
                    raise ValueError(f"successful episode {episode_id} is missing {key}")
            provenance = row["provenance"]
            for key in ("source_file", "source_demo"):
                if key not in provenance:
                    raise ValueError(
                        f"successful episode {episode_id} provenance is missing {key}"
                    )
            task = str(row["task"])
            source_file = str(provenance["source_file"])
            source_demo = str(provenance["source_demo"])
            group_id = json.dumps(
                [task, source_file, source_demo],
                ensure_ascii=False,
                separators=(",", ":"),
            )
            records[episode_id] = {
                "episode_id": episode_id,
                "task": task,
                "group_id": group_id,
                "source_file": source_file,
                "source_demo": source_demo,
                "terminal_step": int(row["terminal_step"]),
            }
    if not records:
        raise ValueError(f"no explicitly successful terminal records in {path}")
    return records, excluded


def _dataset_metadata(dataset_path: str | Path) -> dict[str, Any]:
    dataset_path = Path(dataset_path)
    meta_path = dataset_path / "meta"
    hashes = {
        name: sha256_file(meta_path / name)
        for name in DATASET_META_FILES
        if (meta_path / name).is_file()
    }
    if "episodes.jsonl" not in hashes:
        raise FileNotFoundError(f"dataset metadata is missing {meta_path / 'episodes.jsonl'}")
    return {
        "dataset_name": dataset_path.name,
        "metadata_file_sha256": hashes,
        "metadata_manifest_sha256": _canonical_sha256(hashes),
    }


def _dataset_episode_records(dataset_path: str | Path) -> dict[str, dict[str, Any]]:
    path = Path(dataset_path) / "meta" / "episodes.jsonl"
    episodes: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            episode_id = str(row["episode_index"])
            if episode_id in episodes:
                raise ValueError(f"duplicate dataset episode_index {episode_id}")
            tasks = row.get("tasks") or []
            episodes[episode_id] = {
                "task": str(tasks[0]) if tasks else None,
                "length": int(row["length"]),
            }
    return episodes


def _episode_sort_key(record: dict[str, Any]) -> tuple[str, int | str]:
    episode_id = record["episode_id"]
    try:
        return (record["task"], int(episode_id))
    except ValueError:
        return (record["task"], episode_id)


def build_residual_split_manifest(
    *,
    success_terminal_manifest: str | Path,
    dataset_path: str | Path,
    success_manifest_repo_path: str,
    seed: int = 20260929,
    legacy_train_episode_ids: tuple[str, ...] = LEGACY_TRAIN_EPISODE_IDS,
    legacy_validation_episode_ids: tuple[str, ...] = LEGACY_VALIDATION_EPISODE_IDS,
) -> dict[str, Any]:
    """Create a deterministic task-stratified group split, preserving pilot roles.

    Previously used pilot training episodes remain in train. The five episodes
    used to choose the 36-to-72 update continuation remain in validation and
    are excluded from the untouched test partition.
    """
    success_terminal_manifest = Path(success_terminal_manifest)
    dataset_path = Path(dataset_path)
    records, excluded = _load_annotation_rows(success_terminal_manifest)
    dataset_episodes = _dataset_episode_records(dataset_path)
    annotation_ids = set(records) | {row["episode_id"] for row in excluded}
    if annotation_ids != set(dataset_episodes):
        raise ValueError(
            "terminal annotation episode IDs must exactly match dataset episodes; "
            f"missing annotations={sorted(set(dataset_episodes) - annotation_ids)[:8]}, "
            f"unknown annotations={sorted(annotation_ids - set(dataset_episodes))[:8]}"
        )
    for episode_id, record in records.items():
        metadata = dataset_episodes[episode_id]
        if metadata["task"] != record["task"]:
            raise ValueError(
                f"task mismatch for episode {episode_id}: "
                f"annotation={record['task']!r}, dataset={metadata['task']!r}"
            )
        if record["terminal_step"] < 0 or record["terminal_step"] >= metadata["length"]:
            raise ValueError(f"terminal_step is outside dataset episode {episode_id}")

    legacy_train = {str(item) for item in legacy_train_episode_ids}
    legacy_validation = {str(item) for item in legacy_validation_episode_ids}
    if legacy_train & legacy_validation:
        raise ValueError("legacy train and validation episode IDs overlap")
    unknown_legacy = (legacy_train | legacy_validation) - set(records)
    if unknown_legacy:
        raise ValueError(
            "legacy pilot episodes must have explicit success labels: "
            f"{sorted(unknown_legacy)}"
        )

    groups_by_task: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    group_for_episode: dict[str, str] = {}
    for record in records.values():
        groups_by_task[record["task"]][record["group_id"]].append(record)
        group_for_episode[record["episode_id"]] = record["group_id"]
    pinned_train_groups = {group_for_episode[item] for item in legacy_train}
    pinned_validation_groups = {
        group_for_episode[item] for item in legacy_validation
    }
    if pinned_train_groups & pinned_validation_groups:
        raise ValueError("a source-demo group appears in both legacy train and validation")

    split_groups: dict[str, set[str]] = {name: set() for name in SPLIT_NAMES}
    split_groups["train"].update(pinned_train_groups)
    split_groups["validation"].update(pinned_validation_groups)
    for task in sorted(groups_by_task):
        task_groups = groups_by_task[task]
        task_group_ids = set(task_groups)
        fixed_train = task_group_ids & pinned_train_groups
        fixed_validation = task_group_ids & pinned_validation_groups
        remaining = sorted(task_group_ids - fixed_train - fixed_validation)
        task_seed = int.from_bytes(
            hashlib.sha256(f"{seed}\0{task}".encode("utf-8")).digest()[:8],
            "big",
        )
        random.Random(task_seed).shuffle(remaining)
        target_validation = max(len(fixed_validation), round(0.15 * len(task_groups)))
        target_test = max(1, round(0.15 * len(task_groups)))
        additional_validation = target_validation - len(fixed_validation)
        if additional_validation + target_test > len(remaining):
            raise ValueError(f"not enough source-demo groups to split task {task!r}")
        split_groups["validation"].update(remaining[:additional_validation])
        test_start = additional_validation
        split_groups["test"].update(remaining[test_start : test_start + target_test])
        train_start = test_start + target_test
        split_groups["train"].update(remaining[train_start:])

    all_groups = set().union(*split_groups.values())
    if all_groups != set(group_for_episode.values()):
        raise ValueError("split generation did not assign every successful source-demo group")
    if any(
        split_groups[left] & split_groups[right]
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
    ):
        raise ValueError("a source-demo group leaked across splits")

    episode_splits = {
        split: sorted(
            (
                record
                for record in records.values()
                if record["group_id"] in group_ids
            ),
            key=_episode_sort_key,
        )
        for split, group_ids in split_groups.items()
    }
    task_counts = {
        split: {
            task: sum(record["task"] == task for record in episode_splits[split])
            for task in sorted(groups_by_task)
        }
        for split in SPLIT_NAMES
    }
    dataset_info = _dataset_metadata(dataset_path)
    manifest = {
        "schema_version": 1,
        "manifest_id": "libero_goal_terminal_residual_group_split_v1",
        "seed": int(seed),
        "dataset": dataset_info,
        "success_terminal_source": {
            "path": success_manifest_repo_path,
            "sha256": sha256_file(success_terminal_manifest),
        },
        "group_key": ["task", "provenance.source_file", "provenance.source_demo"],
        "allocation": {
            "method": "deterministic task-stratified source-demo group assignment",
            "target_fractions": {"train": 0.70, "validation": 0.15, "test": 0.15},
            "legacy_pilot_train_episode_ids_pinned": sorted(legacy_train, key=int),
            "legacy_pilot_validation_episode_ids_pinned": sorted(legacy_validation, key=int),
            "legacy_validation_note": (
                "These five episodes informed the earlier continuation decision; "
                "they remain validation and are not part of the untouched test split."
            ),
        },
        "counts": {
            "dataset_episodes": len(dataset_episodes),
            "successful_episodes": len(records),
            "excluded_unlabeled_episodes": len(excluded),
            "source_demo_groups": len(set(group_for_episode.values())),
            "episodes_by_split": {
                split: len(episode_splits[split]) for split in SPLIT_NAMES
            },
            "groups_by_split": {
                split: len(split_groups[split]) for split in SPLIT_NAMES
            },
            "episodes_by_task_and_split": task_counts,
        },
        "splits": episode_splits,
        "excluded_unlabeled_episodes": sorted(excluded, key=lambda row: int(row["episode_id"])),
    }
    manifest["manifest_sha256"] = _canonical_sha256(manifest)
    return manifest


def validate_residual_split_manifest(
    manifest: dict[str, Any],
    *,
    success_terminal_manifest: str | Path | None = None,
    dataset_path: str | Path | None = None,
) -> None:
    """Verify frozen split integrity, source hashes, and group disjointness."""
    if manifest.get("schema_version") != 1:
        raise ValueError(f"unsupported residual split schema {manifest.get('schema_version')!r}")
    expected_hash = manifest.get("manifest_sha256")
    if not expected_hash or _canonical_sha256(manifest) != expected_hash:
        raise ValueError("residual split manifest SHA-256 verification failed")
    if success_terminal_manifest is not None:
        annotation_path = Path(success_terminal_manifest)
        current_hash = sha256_file(annotation_path)
        if current_hash != manifest["success_terminal_source"]["sha256"]:
            raise ValueError("success-terminal annotation hash does not match frozen split source")
        current_records, current_excluded = _load_annotation_rows(annotation_path)
        listed_success = {
            str(row["episode_id"])
            for split in SPLIT_NAMES
            for row in manifest["splits"].get(split, [])
        }
        if listed_success != set(current_records):
            raise ValueError("frozen split episode IDs no longer match success annotations")
        listed_excluded = {
            str(row["episode_id"]) for row in manifest.get("excluded_unlabeled_episodes", [])
        }
        if listed_excluded != {row["episode_id"] for row in current_excluded}:
            raise ValueError("frozen split exclusions no longer match success annotations")
        for split in SPLIT_NAMES:
            for entry in manifest["splits"].get(split, []):
                current = current_records.get(str(entry["episode_id"]))
                if current is None or any(
                    current[key] != entry.get(key)
                    for key in ("task", "group_id", "source_file", "source_demo", "terminal_step")
                ):
                    raise ValueError(
                        f"split entry for episode {entry.get('episode_id')} differs from annotation"
                    )
    if dataset_path is not None:
        if _dataset_metadata(dataset_path) != manifest["dataset"]:
            raise ValueError("dataset metadata hashes do not match frozen residual split")

    split_ids: dict[str, set[str]] = {}
    split_groups: dict[str, set[str]] = {}
    for split in SPLIT_NAMES:
        rows = manifest.get("splits", {}).get(split)
        if not isinstance(rows, list):
            raise ValueError(f"split {split!r} must contain a list of episode records")
        ids = [str(row["episode_id"]) for row in rows]
        groups = [str(row["group_id"]) for row in rows]
        if len(ids) != len(set(ids)):
            raise ValueError(f"split {split!r} contains duplicate episode IDs")
        split_ids[split] = set(ids)
        split_groups[split] = set(groups)
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if split_ids[left] & split_ids[right]:
            raise ValueError(f"episode IDs leak between {left} and {right}")
        if split_groups[left] & split_groups[right]:
            raise ValueError(f"source-demo groups leak between {left} and {right}")


def load_residual_episode_split(
    manifest_path: str | Path,
    *,
    split: str,
    success_terminal_manifest: str | Path | None = None,
    dataset_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load one named split and verify it against the current data sources."""
    if split not in SPLIT_NAMES:
        raise ValueError(f"split must be one of {SPLIT_NAMES}, got {split!r}")
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_residual_split_manifest(
        manifest,
        success_terminal_manifest=success_terminal_manifest,
        dataset_path=dataset_path,
    )
    return {
        "split": split,
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": manifest["manifest_sha256"],
        "episode_ids": [str(row["episode_id"]) for row in manifest["splits"][split]],
        "entries": manifest["splits"][split],
    }
