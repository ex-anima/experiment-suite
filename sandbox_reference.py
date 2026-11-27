"""Fit separate nearest-neighbour references and evaluate them in the Sandbox."""

import argparse
import csv
import hashlib
import json
import math
import statistics
import subprocess
from collections import Counter
from pathlib import Path

from sandbox_decisions import FEATURES, INTERVAL, SCHEMA, STYLES, observations, write_json


AXES = tuple((x, y) for x in (-1, 0, 1) for y in (-1, 0, 1) if (x, y) != (0, 0))
COOLDOWNS = {"melee": "melee_cd", "ranged": "ranged_cd", "parry": "parry_cd", "dodge": "dodge_cd"}


def predict(training, observation):
    closest = min(training, key=lambda row: math.dist(row["observation"], observation))
    return closest["decision"]


def read_samples(directory, manifest, style, split):
    name = f"{style}-{split}.jsonl"
    raw = (directory / name).read_bytes()
    expected = manifest["files"][name]
    if hashlib.sha256(raw).hexdigest() != expected["sha256"]:
        raise ValueError(f"Dataset hash mismatch: {name}")
    rows = [json.loads(line) for line in raw.splitlines()]
    if (not rows or len(rows) != expected["count"]
            or any(row["seed"] not in manifest[f"{split}_seeds"] or row["style"] != style
                   for row in rows)):
        raise ValueError(f"Dataset split mismatch: {name}")
    return rows


def offline(training, testing, predict_label):
    labels = Counter(row["decision"] for row in training)
    majority = max(sorted(labels), key=labels.get)
    guesses = [predict_label(row["observation"]) for row in testing]
    actual = [row["decision"] for row in testing]
    guessed_abilities = [label.split("+")[1] for label in guesses]
    actual_abilities = [label.split("+")[1] for label in actual]
    return {
        "samples": len(testing),
        "exact_matches": sum(g == a for g, a in zip(guesses, actual)),
        "majority_exact_matches": actual.count(majority),
        "movement_matches": sum(g.split("+")[0] == a.split("+")[0] for g, a in zip(guesses, actual)),
        "applied_ability_labels": sum(a != "none" for a in actual_abilities),
        "predicted_abilities": sum(g != "none" for g in guessed_abilities),
        "matched_non_none_abilities": sum(g == a and a != "none"
                                          for g, a in zip(guessed_abilities, actual_abilities)),
    }


def decode_movement(row, movement):
    if movement == "hold":
        return 0, 0
    dx = float(row["enemy_x"]) - float(row["player_x"])
    dy = float(row["enemy_y"]) - float(row["player_y"])
    if dx == dy == 0:
        return 0, 0
    dx, dy = {
        "approach": (dx, dy), "retreat": (-dx, -dy),
        "strafe_clockwise": (-dy, dx), "strafe_counterclockwise": (dy, -dx),
    }[movement]
    return max(AXES, key=lambda axis: (axis[0] * dx + axis[1] * dy) / math.hypot(*axis))


def ready_for_ability(row, pending):
    return (pending != "none" and int(row[COOLDOWNS[pending]]) == 0
            and int(row["parry_ticks"]) == int(row["dodge_ticks"]) == 0
            and (pending != "ranged" or int(row["projectile_ticks"]) == 0))


def run_policy(sandbox, predict_label, seed, ticks, directory, wait_for_melee_range):
    cmd = [str(sandbox), "--policy-stdio", "--seed", str(seed), "--ticks", str(ticks),
           "--record", str(directory)]
    movement = "hold"
    pending = "none"
    decisions = Counter()
    end = None
    with subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                          text=True, encoding="ascii") as simulator:
        try:
            for line in simulator.stdout:
                kind, *fields = line.rstrip("\n").split("\t")
                if kind == "end" and len(fields) == 4:
                    tick, player_hp, enemy_hp = map(int, fields[:3])
                    end = {"seed": seed, "ticks": tick, "player_hp": player_hp,
                           "enemy_hp": enemy_hp, "outcome": fields[3]}
                    break
                if kind != "state" or len(fields) != len(FEATURES) + 2 or fields[-1] not in {"0", "1"}:
                    raise ValueError(f"Unexpected Sandbox protocol: {line!r}")
                tick = int(fields[0])
                row = dict(zip(FEATURES, fields[1:]))
                row["tick"] = tick
                if tick % INTERVAL == 0:
                    label = predict_label(observations(row))
                    decisions[label] += 1
                    movement, pending = label.split("+")
                x, y = decode_movement(row, movement)
                ability = "none"
                if ready_for_ability(row, pending) and (
                    not wait_for_melee_range or pending != "melee" or fields[-1] == "1"
                ):
                    ability, pending = pending, "none"
                simulator.stdin.write(f"{x}\t{y}\t{ability}\n")
                simulator.stdin.flush()
            simulator.stdin.close()
            if simulator.wait() != 0 or end is None:
                raise RuntimeError("Sandbox closed without a valid episode result")
        except BaseException:
            if simulator.poll() is None:
                simulator.kill()
            raise
    recording = directory / "episode-1.tsv"
    subprocess.run([str(sandbox), "--replay", str(recording)], capture_output=True, check=True)
    with recording.open(encoding="ascii", newline="") as episode:
        next(episode)
        frames = list(csv.DictReader((line for line in episode if not line.startswith("#")), delimiter="\t"))
    attacks = [row for row in frames if row["applied"] in {"melee", "ranged"}]
    attack_distances = [
        math.hypot(float(r["enemy_x"]) - float(r["player_x"]), float(r["enemy_y"]) - float(r["player_y"]))
        for r in attacks
    ]
    recovery_approach_ticks = 0
    for row in frames:
        if int(row["enemy_recovery"]) > 0:
            dx = float(row["enemy_x"]) - float(row["player_x"])
            dy = float(row["enemy_y"]) - float(row["player_y"])
            recovery_approach_ticks += int(row["move_x"]) * dx + int(row["move_y"]) * dy > 0
    return {
        **end, "decisions": dict(decisions),
        "applied_attacks": dict(Counter(row["applied"] for row in attacks)),
        "melee_during_recovery": sum(row["applied"] == "melee" and int(row["enemy_recovery"]) > 0
                                     for row in attacks),
        "melee_hits": sum(row["applied"] == "melee" and int(row["damage_to_enemy"]) > 0
                          for row in attacks),
        "recovery_approach_input_ticks": recovery_approach_ticks,
        "damage_received": sum(int(row["damage_to_player"]) for row in frames),
        "median_attack_distance": statistics.median(attack_distances) if attack_distances else None,
        "record_sha256": hashlib.sha256(recording.read_bytes()).hexdigest(),
        "replay_verified": True,
    }


def evaluate(directory, sandbox, output, ticks, wait_for_melee_range, rollout_seeds):
    raw_manifest = (directory / "dataset.json").read_bytes()
    manifest = json.loads(raw_manifest)
    if (manifest["schema"] != SCHEMA or manifest["decision_interval_ticks"] != INTERVAL
            or set(manifest["train_seeds"]) & set(manifest["test_seeds"])):
        raise ValueError("Unexpected schema or overlapping seed split")
    rollout_seeds = list(dict.fromkeys(rollout_seeds or manifest["test_seeds"]))
    if (set(rollout_seeds) & set(manifest["train_seeds"])
            or any(not 0 <= seed <= 2**32 - 1 for seed in rollout_seeds)):
        raise ValueError("Rollout seeds must be uint32 values outside the training set")
    output.mkdir()
    styles = {}
    for style in STYLES:
        training = read_samples(directory, manifest, style, "train")
        testing = read_samples(directory, manifest, style, "test")
        def predict_label(observation):
            return predict(training, observation)

        write_json(output / f"{style}-model.json", {
            "schema": "sandbox-nearest-neighbour-v1", "style": style, "train_seeds": manifest["train_seeds"],
            "feature_order": manifest["feature_order"],
            "examples": [{"observation": row["observation"], "decision": row["decision"]} for row in training],
        })
        styles[style] = {
            "offline": offline(training, testing, predict_label),
            "closed_loop": [run_policy(sandbox, predict_label, seed, ticks, output / f"{style}-{seed}",
                                       wait_for_melee_range)
                            for seed in rollout_seeds],
        }
    write_json(output / "summary.json", {
        "dataset_manifest_sha256": hashlib.sha256(raw_manifest).hexdigest(),
        "simulation_binary_sha256": hashlib.sha256(sandbox.read_bytes()).hexdigest(),
        "train_seeds": manifest["train_seeds"], "test_seeds": manifest["test_seeds"],
        "rollout_seeds": rollout_seeds,
        "tick_limit": ticks, "interval_ticks": INTERVAL,
        "wait_for_melee_range": wait_for_melee_range,
        "model": "1-nearest neighbour, Euclidean distance, no random seed",
        "styles": styles,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("sandbox", type=Path)
    parser.add_argument("output", type=Path, help="New output directory")
    parser.add_argument("--ticks", type=int, default=3600)
    parser.add_argument("--wait-for-melee-range", action="store_true")
    parser.add_argument("--rollout-seed", type=int, action="append",
                        help="Override the default Sandbox rollout seeds; repeat to specify more than one")
    args = parser.parse_args()
    if args.ticks <= 0:
        parser.error("--ticks must be positive")
    evaluate(args.dataset, args.sandbox.resolve(), args.output.resolve(), args.ticks,
             args.wait_for_melee_range, args.rollout_seed)
