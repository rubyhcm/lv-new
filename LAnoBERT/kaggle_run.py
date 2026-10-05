"""Run the LAnoBERT baseline on Kaggle.

Upload this repository as a Kaggle Dataset (or upload a zip containing it),
attach the dataset to a GPU notebook, and run:

    !python /kaggle/input/<dataset-name>/kaggle_run.py \
        --project-input /kaggle/input/<dataset-name> \
        --config configs/bgl.yaml \
        --max-eval-samples 10000

For a full run, omit ``--max-eval-samples``. The raw BGL file must be present
under ``data/BGL/BGL.log`` or supplied with ``--download-data`` when Kaggle
Internet is enabled.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


KAGGLE_WORKING = Path("/kaggle/working")


def run(command: list[str], cwd: Path) -> None:
    print("$", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def copy_project(source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(
        source,
        destination,
        ignore=shutil.ignore_patterns(
            ".git", "__pycache__", ".pytest_cache", "outputs", "data",
        ),
    )


def find_project(source: Path) -> Path:
    if (source / "lanobert").is_dir() and (source / "configs").is_dir():
        return source
    candidates = [
        path for path in source.rglob("lanobert")
        if path.is_dir() and (path.parent / "configs").is_dir()
    ]
    if len(candidates) != 1:
        raise FileNotFoundError(
            "Could not uniquely locate LAnoBERT. Pass --project-input pointing "
            "to the repository directory or a directory containing it."
        )
    return candidates[0].parent


def ensure_bgl_data(project: Path, download_data: bool) -> None:
    raw_log = project / "data/BGL/BGL.log"
    if raw_log.is_file():
        return
    if not download_data:
        raise FileNotFoundError(
            f"Missing {raw_log}. Attach a Kaggle dataset containing BGL.log "
            "or rerun with --download-data and enable Internet."
        )
    run(["bash", "scripts/download_data.sh", "bgl"], project)
    if not raw_log.is_file():
        raise FileNotFoundError(f"Download completed but {raw_log} is missing")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-input", type=Path, required=True)
    parser.add_argument("--config", default="configs/bgl.yaml")
    parser.add_argument("--download-data", action="store_true")
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()

    source = find_project(args.project_input.resolve())
    project = KAGGLE_WORKING / "LAnoBERT"
    copy_project(source, project)
    print(f"Project copied to {project}")

    run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], project)
    ensure_bgl_data(project, args.download_data)

    config = project / args.config
    if not config.is_file():
        raise FileNotFoundError(f"Config not found: {config}")

    runtime_config = config
    if args.max_eval_samples is not None or args.epochs is not None:
        import yaml

        with config.open(encoding="utf-8") as handle:
            values = yaml.safe_load(handle)
        if args.max_eval_samples is not None:
            values.setdefault("inference", {})["max_eval_samples"] = args.max_eval_samples
        if args.epochs is not None:
            values.setdefault("train", {})["num_train_epochs"] = args.epochs
        runtime_config = project / "configs/kaggle_runtime.yaml"
        with runtime_config.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(values, handle, sort_keys=False)

    run(["bash", "scripts/ensure_data.sh", str(runtime_config)], project)
    run([sys.executable, "-m", "lanobert.tokenizer",
         "--config", str(runtime_config)], project)

    if not args.skip_train:
        run([sys.executable, "-m", "lanobert.train",
             "--config", str(runtime_config)], project)

    run([sys.executable, "-m", "lanobert.inference",
         "--config", str(runtime_config)], project)

    archive = KAGGLE_WORKING / "lanobert-results"
    if archive.exists():
        archive.unlink()
    shutil.make_archive(str(archive), "zip", project / "outputs")
    print(f"Results archive: {archive}.zip")


if __name__ == "__main__":
    main()
