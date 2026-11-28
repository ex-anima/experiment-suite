"""Run both saved release-candidate policies and verify their Sandbox replays."""

import argparse
import json
from pathlib import Path

from core_style import inference, sha256
from sandbox_decisions import FEATURES, INTERVAL, STYLES, write_json
from sandbox_reference import COOLDOWNS, decode_movement, run_policy


SCREEN_AXES = {
    "hold": (0, 0), "north": (0, -1), "northeast": (1, -1), "east": (1, 0),
    "southeast": (1, 1), "south": (0, 1), "southwest": (-1, 1), "west": (-1, 0), "northwest": (-1, -1),
}
RELATIVE_DIRECTIONS = {"hold", "approach", "retreat", "strafe_clockwise", "strafe_counterclockwise"}
MODEL_FILES = ("model.json", "weights.bin")


def screen_movement(_state, movement):
    return SCREEN_AXES[movement]


def read_bundle(bundle):
    manifest = json.loads((bundle / "candidate.json").read_bytes())
    if (manifest["format"] != "exanima-sandbox-candidate-v1" or manifest["decision_interval_ticks"] != INTERVAL
            or set(manifest["styles"]) != set(STYLES) or manifest["tick_limit"] <= 0):
        raise ValueError("Unexpected Sandbox candidate manifest")
    for style in STYLES:
        entry = manifest["styles"][style]
        model = bundle / style
        if {name: sha256(model / name) for name in MODEL_FILES} != entry["sha256"]:
            raise ValueError(f"Model hash mismatch: {style}")
        directions = {"screen": SCREEN_AXES, "relative": RELATIVE_DIRECTIONS}[entry["movement"]]
        metadata = json.loads((model / "model.json").read_bytes())
        allowed = {f"{movement}+{ability}" for movement in directions for ability in (*COOLDOWNS, "none")}
        if (metadata["input_features"] != list(FEATURES) + ["relative_x", "relative_y", "enemy_distance"]
                or not metadata["labels"] or set(metadata["labels"]) - allowed):
            raise ValueError(f"Model observations or labels do not match its movement encoding: {style}")
    return manifest


def evaluate(bundle, infer, sandbox, output, layout, seeds=None):
    manifest = read_bundle(bundle)
    seeds = manifest["seeds"] if seeds is None else seeds
    if not seeds or any(type(seed) is not int or not 0 <= seed <= 2**32 - 1 for seed in seeds):
        raise ValueError("Sandbox seeds must be uint32 values")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Sandbox seeds must be unique")
    output.mkdir()
    settings = {
        "candidate_sha256": sha256(bundle / "candidate.json"),
        "inference_sha256": sha256(infer), "sandbox_sha256": sha256(sandbox),
        "layout": layout, "seeds": seeds, "models": manifest["styles"],
        "tick_limit": manifest["tick_limit"], "decision_interval_ticks": INTERVAL,
    }
    write_json(output / "protocol.json", settings)
    results = []
    for style in STYLES:
        movement = manifest["styles"][style]["movement"]
        decoder = {"screen": screen_movement, "relative": decode_movement}[movement]
        directory = output / style
        directory.mkdir()
        with inference(infer, bundle / style) as predict:
            for seed in seeds:
                episode = directory / str(seed)
                result = run_policy(sandbox, predict, seed, manifest["tick_limit"], episode, True,
                                    layout=layout, movement_decoder=decoder)
                results.append({"style": style, **result, "record": f"{style}/{seed}/episode-1.tsv"})
    write_json(output / "summary.json", {
        "protocol_sha256": sha256(output / "protocol.json"), "runs": results,
        "verified_replays": len(results),
    })
    print(f"{layout}: {len(results)} verified replays")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path, help="Directory holding candidate.json and both saved models")
    parser.add_argument("infer", type=Path, help="Core inference-only executable")
    parser.add_argument("sandbox", type=Path, help="Sandbox executable")
    parser.add_argument("output", type=Path, help="New output directory")
    parser.add_argument("--layout", choices=("standard", "inset"), default="standard")
    parser.add_argument("--seed", type=int, action="append", help="Override the manifest seeds")
    args = parser.parse_args()
    evaluate(args.bundle.resolve(), args.infer.resolve(), args.sandbox.resolve(), args.output.resolve(),
             args.layout, args.seed)
