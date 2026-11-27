"""Train separate Core MLPs and run their saved models in the Sandbox."""

import argparse
import hashlib
import json
import subprocess
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter

from sandbox_decisions import INTERVAL, SCHEMA, STYLES, write_json
from sandbox_reference import offline, read_samples, run_policy


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@contextmanager
def inference(executable, model):
    with subprocess.Popen([str(executable), str(model)], stdin=subprocess.PIPE,
                          stdout=subprocess.PIPE, text=True, encoding="utf-8") as process:
        def predict_label(observation):
            process.stdin.write(json.dumps(observation, separators=(",", ":")) + "\n")
            process.stdin.flush()
            line = process.stdout.readline()
            if not line:
                raise RuntimeError("Core inference ended before returning a prediction")
            return json.loads(line)["label"]

        try:
            yield predict_label
            process.stdin.close()
            if process.wait() != 0:
                raise RuntimeError("Core inference exited with an error")
        except BaseException:
            if process.poll() is None:
                process.kill()
            raise


def evaluate(args):
    meta_path = args.dataset / "dataset.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if (meta["schema"] != SCHEMA or meta["decision_interval_ticks"] != INTERVAL
            or set(meta["train_seeds"]) & set(meta["test_seeds"])):
        raise ValueError("Unexpected dataset schema or overlapping train and test seeds")
    seeds = list(dict.fromkeys(args.rollout_seed or meta["test_seeds"]))
    if any(not 0 <= seed <= 2**32 - 1 for seed in seeds) or set(seeds) & set(meta["train_seeds"]):
        raise ValueError("Rollout seeds must be uint32 values outside the training set")

    output = args.output.resolve()
    output.mkdir()
    results = {}
    for style in STYLES:
        training = read_samples(args.dataset, meta, style, "train")
        testing = read_samples(args.dataset, meta, style, "test")
        config = {
            "input_features": meta["feature_order"],
            "labels": sorted({row["decision"] for row in training}),
            "hidden_units": args.hidden_units, "seed": args.training_seed,
            "epochs": args.epochs, "learning_rate": args.learning_rate,
            "balance_classes": args.balance_classes,
        }
        config_path = output / f"{style}-training.json"
        model = output / f"{style}-model"
        write_json(config_path, config)
        started = perf_counter()
        command = [str(args.train.resolve()), str(config_path),
                   str((args.dataset / f"{style}-train.jsonl").resolve()), str(model)]
        fit = json.loads(subprocess.check_output(command, text=True, encoding="utf-8"))
        training_seconds = perf_counter() - started
        with inference(args.infer.resolve(), model) as predict_label:
            results[style] = {
                "fit": fit, "training_wall_seconds": training_seconds,
                "files": {name: sha256(model / name) for name in ("model.json", "weights.bin")},
                "offline": offline(training, testing, predict_label),
                "closed_loop": [run_policy(args.sandbox.resolve(), predict_label, seed, args.ticks,
                                           output / f"{style}-{seed}", True) for seed in seeds],
            }
    write_json(output / "summary.json", {
        "model": "Core exanima-mlp-v1",
        "dataset_manifest_sha256": sha256(meta_path),
        "training_binary_sha256": sha256(args.train),
        "inference_binary_sha256": sha256(args.infer),
        "sandbox_binary_sha256": sha256(args.sandbox),
        "train_seeds": meta["train_seeds"], "offline_seeds": meta["test_seeds"],
        "rollout_seeds": seeds, "interval_ticks": INTERVAL, "tick_limit": args.ticks,
        "wait_for_melee_range": True, "styles": results,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("train", type=Path, help="Core exanima-train executable")
    parser.add_argument("infer", type=Path, help="Core exanima-infer executable")
    parser.add_argument("sandbox", type=Path)
    parser.add_argument("output", type=Path, help="New output directory")
    parser.add_argument("--hidden-units", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--training-seed", type=int, default=42)
    parser.add_argument("--balance-classes", action="store_true")
    parser.add_argument("--ticks", type=int, default=3600)
    parser.add_argument("--rollout-seed", type=int, action="append")
    args = parser.parse_args()
    if args.ticks <= 0:
        parser.error("--ticks must be positive")
    evaluate(args)
