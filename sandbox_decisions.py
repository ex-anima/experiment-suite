"""Convert matched Sandbox demonstrations to one categorical decision per interval."""

import argparse
import csv
import hashlib
import io
import json
import math
import statistics
from collections import Counter
from pathlib import Path, PureWindowsPath


SCHEMA = "sandbox-decisions-v1"
TICK_HZ = 60
INTERVAL = 6
STYLES = ("cautious", "aggressive")
ABILITIES = {"none", "melee", "ranged", "parry", "dodge"}
FEATURES = {
    "player_x": 800, "player_y": 520, "enemy_x": 800, "enemy_y": 520,
    "player_hp": 100, "enemy_hp": 100,
    "melee_cd": 28, "ranged_cd": 42, "parry_cd": 45, "dodge_cd": 48,
    "enemy_windup": 36, "enemy_recovery": 90, "projectile_ticks": 39,
    "parry_ticks": 18, "dodge_ticks": 12,
}


def observations(row):
    values = {key: float(row[key]) / scale for key, scale in FEATURES.items()}
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values.values()):
        raise ValueError(f"Invalid pre-decision observation at tick {row['tick']}")
    dx = float(row["enemy_x"]) - float(row["player_x"])
    dy = float(row["enemy_y"]) - float(row["player_y"])
    return [*values.values(), dx / 800, dy / 520, math.hypot(dx, dy) / math.hypot(800, 520)]


def classify_movement(window):
    x = y = 0.0
    for row in window:
        dx, dy = int(row["move_x"]), int(row["move_y"])
        if dx not in {-1, 0, 1} or dy not in {-1, 0, 1}:
            raise ValueError(f"Invalid movement at tick {row['tick']}")
        length = math.hypot(dx, dy)
        if length:
            x += dx / length
            y += dy / length

    if math.hypot(x, y) / len(window) < 0.25:
        return "hold"
    first = window[0]
    dx = float(first["enemy_x"]) - float(first["player_x"])
    dy = float(first["enemy_y"]) - float(first["player_y"])
    toward = dx * x + dy * y
    clockwise = dx * y - dy * x
    if not (dx or dy):
        return "hold"
    if abs(toward) >= abs(clockwise):
        return "approach" if toward > 0 else "retreat"
    return "strafe_clockwise" if clockwise > 0 else "strafe_counterclockwise"


def episode_rows(pilot, record):
    relative = PureWindowsPath(record["path"])
    if relative.anchor or ".." in relative.parts:
        raise ValueError("Recording path must be inside the pilot directory")
    path = pilot.joinpath(*relative.parts).resolve()
    if not path.is_relative_to(pilot):
        raise ValueError("Recording path resolves outside the pilot directory")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != record["sha256"]:
        raise ValueError(f"Recording hash differs: {relative}")
    with io.StringIO(raw.decode("ascii"), newline="") as source:
        seed = record["seed"]
        if source.readline().rstrip("\n") != f"# sandbox-episode-v2\tseed={seed}\tcontroller=human":
            raise ValueError(f"Unsupported recording or seed: {relative}")
        reader = csv.DictReader(source, delimiter="\t")
        frames = []
        for row in reader:
            if row["tick"] == "# end":
                if row["move_x"] not in {"won", "lost"} or row["move_x"] != record["outcome"]:
                    raise ValueError(f"Unexpected end of episode: {relative}")
                if not frames or frames[-1]["outcome"] != row["move_x"] or next(reader, None) is not None:
                    raise ValueError(f"Unexpected terminal state: {relative}")
                return frames
            if int(row["tick"]) != len(frames) or row["applied"] not in ABILITIES:
                raise ValueError(f"Unexpected frame: {relative}")
            frames.append(row)
    raise ValueError(f"Episode ending missing: {relative}")


def decisions(frames, style, seed):
    rows = []
    raw_abilities = Counter()
    kept_abilities = Counter()
    moving_attacks = covered_moving_attacks = collisions = 0
    for start in range(0, len(frames), INTERVAL):
        window = frames[start:start + INTERVAL]
        movement = classify_movement(window)
        events = [row for row in window if row["applied"] != "none"]
        raw_abilities.update(row["applied"] for row in events)
        moving_attacks += sum(row["applied"] in {"melee", "ranged"}
                              and (int(row["move_x"]) or int(row["move_y"])) != 0 for row in events)
        collisions += len(events) > 1
        action = events[0] if events else None
        ability = action["applied"] if action else "none"
        if action:
            kept_abilities[ability] += 1
            covered_moving_attacks += (
                ability in {"melee", "ranged"} and movement != "hold"
                and (int(action["move_x"]) != 0 or int(action["move_y"]) != 0)
            )
        rows.append({
            "seed": seed, "style": style, "tick": start, "duration_ticks": len(window),
            "observation": observations(window[0]), "decision": f"{movement}+{ability}",
            "applied_event_tick": int(action["tick"]) if action else None,
        })
    attack_distances = [
        math.hypot(float(row["enemy_x"]) - float(row["player_x"]),
                   float(row["enemy_y"]) - float(row["player_y"]))
        for row in frames if row["applied"] in {"ranged", "melee"}
    ]
    metrics = {
        "ticks": len(frames), "decisions": len(rows),
        "raw_applied": dict(raw_abilities), "kept_applied": dict(kept_abilities),
        "multiple_abilities_windows": collisions,
        "moving_attacks": moving_attacks,
        "moving_attacks_kept_as_movement_and_attack": covered_moving_attacks,
        "median_attack_distance": statistics.median(attack_distances) if attack_distances else None,
    }
    return rows, metrics


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def convert(pilot, output, heldout):
    pilot = pilot.resolve()
    manifest = json.loads((pilot / "pilot.json").read_text(encoding="utf-8"))
    if manifest["record_format"] != "sandbox-episode-v2" or manifest["sample_hz"] != TICK_HZ:
        raise ValueError("Expected a 60 Hz human pilot using Sandbox episode v2")
    keys = [(row["seed"], row["style"]) for row in manifest["completed"]]
    seeds = sorted({seed for seed, _ in keys})
    if (len(keys) != len(seeds) * len(STYLES)
            or set(keys) != {(seed, style) for seed in seeds for style in STYLES}):
        raise ValueError("Expected one completed episode per style and seed")
    if heldout not in seeds or len(seeds) < 2:
        raise ValueError("The held-out seed needs a paired episode and at least one training seed")

    groups = {(style, split): [] for style in STYLES for split in ("train", "test")}
    episodes = []
    for record in manifest["completed"]:
        seed, style = record["seed"], record["style"]
        split = "test" if seed == heldout else "train"
        frames = episode_rows(pilot, record)
        if len(frames) != record["ticks"]:
            raise ValueError("Recorded tick count differs from the manifest")
        samples, metrics = decisions(frames, style, seed)
        groups[style, split].extend(samples)
        episodes.append({"seed": seed, "style": style, "split": split,
                         "record_sha256": record["sha256"], **metrics})
    train_labels = {style: {row["decision"] for row in groups[style, "train"]} for style in STYLES}
    files = {}
    output.mkdir()
    for (style, split), rows in groups.items():
        name = f"{style}-{split}.jsonl"
        text = "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows)
        path = output / name
        path.write_bytes(text.encode("utf-8"))
        files[name] = {
            "count": len(rows), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "labels": dict(sorted(Counter(row["decision"] for row in rows).items())),
            "unseen_in_style_training": sum(row["decision"] not in train_labels[style] for row in rows),
        }
    write_json(output / "dataset.json", {
        "schema": SCHEMA,
        "sandbox_commit": manifest["commit"],
        "sandbox_binary_sha256": manifest["binary_sha256"],
        "tick_hz": TICK_HZ, "decision_interval_ticks": INTERVAL,
        "feature_order": [*FEATURES, "relative_x", "relative_y", "enemy_distance"],
        "train_seeds": [seed for seed in seeds if seed != heldout],
        "test_seeds": [heldout],
        "files": files, "episodes": episodes,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pilot", type=Path)
    parser.add_argument("output", type=Path, help="New output directory")
    parser.add_argument("--test-seed", type=int, required=True)
    args = parser.parse_args()
    convert(args.pilot, args.output, args.test_seed)
