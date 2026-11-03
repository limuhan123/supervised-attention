"""Launch fixed-protocol two-scale external controls through the existing train.py."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

MODELS = {"E0": "YOLOv8_small_se", "E1": "YOLOv8_small_cbam",
          "A0": "YOLOv8_small", "A1": "YOLOv8_small_self_branch",
          "A2": "YOLOv8_small_self_branch", "A3": "YOLOv8_small_self_branch"}
ROOT = Path(__file__).resolve().parent


def command(code, seed, args):
    return [sys.executable, str(ROOT / "train.py"),
            "--model_name", MODELS[code], "--data_path", str(Path(args.data_path).resolve()),
            "--save_dir", str(Path(args.output).resolve()), "--save_name", f"{code}_seed{seed}",
            "--seed", str(seed), "--device", args.device, "--num_classes", "1",
            "--label_filter_classes", "0", "--attention_context_classes", "4",
            "--image_size", "640", "--batch_size", "32", "--accumulate", "2",
            "--epochs", "300", "--patience", "30", "--num_workers", "0",
            "--lr", "0.005", "--weight_decay", "0.0005", "--use_amp",
            "--val_conf_threshold", "0.001", "--val_nms_iou_threshold", "0.7",
            "--val_pre_nms_topk", "1000", "--val_max_det", "300",
            "--gate_mode", "off" if code == "A2" else "learned", "--gate_init", "0.5",
            "--branch_loss", "0.5" if code in ("A2", "A3") else "0",
            "--attention_loss_mode", "full", "--attention_supervision_mode", "relaxed"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_path", required=True, help="Original fixed-split dataset root")
    p.add_argument("--output", default="weights/external_controls")
    p.add_argument("--device", default="0")
    p.add_argument("--experiments", nargs="+", choices=MODELS, default=["E0", "E1"])
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--execute", action="store_true", help="Execute; otherwise print plan only")
    args = p.parse_args()
    jobs = [(code, seed, command(code, seed, args))
            for code in args.experiments for seed in args.seeds]
    for code, seed, cmd in jobs:
        print(f"{code}_seed{seed}: {subprocess.list2cmdline(cmd)}", flush=True)
    if not args.execute:
        return
    data = Path(args.data_path).resolve()
    inventory = {}
    for split in ("train", "val"):
        for kind in ("images", "labels"):
            folder = data / kind / split
            if not folder.is_dir():
                raise FileNotFoundError(folder)
            files = sorted(f for f in folder.iterdir() if f.is_file())
            if not files:
                raise ValueError(f"Empty dataset directory: {folder}")
            inventory[f"{kind}/{split}"] = {
                f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    output = Path(args.output).resolve()
    # Refuse to overwrite any existing run, including a partially completed run.
    for code, seed, _ in jobs:
        if (output / f"{code}_seed{seed}").exists():
            raise FileExistsError(output / f"{code}_seed{seed}")
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"data": str(data), "inventory": inventory, "commands": [j[2] for j in jobs],
                "source_sha256": {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest()
                                  for f in [ROOT / "train.py", *sorted((ROOT / "network").glob("*.py"))]}}
    manifest_path = output / ("plan_" + "_".join(args.experiments) + "_" +
                              "_".join(map(str, args.seeds)) + ".json")
    with manifest_path.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
    env = dict(os.environ, NO_ALBUMENTATIONS_UPDATE="1", CUBLAS_WORKSPACE_CONFIG=":4096:8")
    for _, _, cmd in jobs:
        subprocess.run(cmd, cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
