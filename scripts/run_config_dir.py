"""
Run all YAML config files in a directory sequentially.

The script to run for each config is taken from the YAML's `script` field
(main_forget.py if omitted).

If a config sets `mask_path` but that file (or directory for mask bundles) does not exist yet,
`generate_mask.py` is run automatically using `mask.yaml` in the same folder as the config
(SalUn / RL layouts that ship a sibling mask.yaml).

Usage:
    python scripts/run_config_dir.py <config_dir> [--run RUN ...] [--gpu ID]

Examples:
    python scripts/run_config_dir.py configs/cifar10/random_50percent/evomoul/...
    python scripts/run_config_dir.py configs/.../GA --run 1 2 3 4 5   # runs 1, 2, 3, 4, 5
    python scripts/run_config_dir.py configs/.../GA --run 1 5        # runs 1 and 5 only
    python scripts/run_config_dir.py configs/.../GA --gpu 1          # use GPU 1
"""

import argparse
import re
import subprocess
import sys
import yaml
from pathlib import Path
from typing import Optional

RUN_PATTERN = re.compile(r"run_(\d+)\.yaml$", re.IGNORECASE)


def get_run_number(path: Path) -> Optional[int]:
    """Extract run number from filename (e.g. run_01.yaml -> 1). Returns None if not matched."""
    m = RUN_PATTERN.search(path.name)
    return int(m.group(1)) if m else None


def _resolve_project_path(path_str: str, project_root: Path) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (project_root / p)


def ensure_mask_if_needed(
    config: dict,
    yaml_path: Path,
    project_root: Path,
    gpu: int,
    dry_run: bool,
) -> bool:
    """
    If mask_path is set but missing, run generate_mask.py with sibling mask.yaml.
    Returns False on failure (caller should record the failed config).
    """
    mask_path_str = config.get("mask_path")
    if not mask_path_str:
        return True

    mask_ref = _resolve_project_path(mask_path_str, project_root)

    if mask_ref.is_dir():
        if any(mask_ref.rglob("*.pt")):
            return True
    elif mask_ref.is_file():
        return True

    mask_yaml = yaml_path.parent / "mask.yaml"
    if not mask_yaml.is_file():
        print(
            f"Error: mask_path '{mask_path_str}' not found at {mask_ref} "
            f"and no mask.yaml beside {yaml_path.name}",
            file=sys.stderr,
        )
        return False

    script_dir = Path(__file__).resolve().parent
    generate_script = script_dir / "generate_mask.py"
    if not generate_script.is_file():
        print(f"Error: generate_mask.py not found at {generate_script}", file=sys.stderr)
        return False

    cmd = [
        sys.executable,
        str(generate_script),
        "--config",
        str(mask_yaml.resolve()),
        "--gpu",
        str(gpu),
    ]
    print(f"  Mask not found at {mask_ref}; running generate_mask.py...")
    if dry_run:
        print(f"  Would run: {' '.join(cmd)}")
        return True

    result = subprocess.run(cmd, cwd=str(project_root))
    if result.returncode != 0:
        print(f"  generate_mask.py exited with code {result.returncode}", file=sys.stderr)
        return False

    if mask_ref.is_dir():
        if not any(mask_ref.rglob("*.pt")):
            print(
                f"  After generate_mask, no .pt files under {mask_ref}",
                file=sys.stderr,
            )
            return False
    elif not mask_ref.is_file():
        print(
            f"  After generate_mask, expected mask still missing: {mask_ref}",
            file=sys.stderr,
        )
        return False

    return True


def get_script_for_config(config: dict) -> str:
    """Determine which main script to run from YAML config."""
    if "script" in config:
        s = config["script"]
        if s in ("main_random", "main_forget", "generate_mask"):
            return s
        # Allow "main_random.py" or "main_random" etc
        if s.endswith(".py"):
            s = s[:-3]
        if s == "main_random":
            return "main_random"
        if s == "main_forget":
            return "main_forget"
        if s == "generate_mask":
            return "generate_mask"
    return "main_forget"


def main():
    parser = argparse.ArgumentParser(
        description="Run all YAML configs in a directory sequentially"
    )
    parser.add_argument(
        "config_dir",
        type=str,
        help="Directory containing YAML config files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing",
    )
    parser.add_argument(
        "--run",
        type=int,
        nargs="*",
        default=None,
        metavar="N",
        help="Run numbers to execute (e.g. --run 1 2 3 4 5 or --run 1 5). "
        "If omitted, all runs in the directory are executed.",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        metavar="ID",
        help="GPU device ID to use (default: 0)",
    )
    args = parser.parse_args()

    config_dir = Path(args.config_dir)
    if not config_dir.is_dir():
        print(f"Error: '{config_dir}' is not a directory", file=sys.stderr)
        sys.exit(1)

    yaml_files = sorted(config_dir.glob("*.yaml"))

    # Filter by --run if specified
    if args.run is not None and len(args.run) > 0:
        run_set = set(args.run)
        yaml_files = [p for p in yaml_files if get_run_number(p) in run_set]
        if not yaml_files:
            print(
                f"Error: No YAML files matching --run {args.run} in '{config_dir}'",
                file=sys.stderr,
            )
            sys.exit(1)

    if not yaml_files:
        print(f"Error: No .yaml files found in '{config_dir}'", file=sys.stderr)
        sys.exit(1)

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    print(f"Found {len(yaml_files)} YAML file(s) in {config_dir}")
    print("-" * 60)

    failed = []
    for i, yaml_path in enumerate(yaml_files, 1):
        with open(yaml_path) as f:
            config = yaml.safe_load(f) or {}
        script_name = get_script_for_config(config)
        main_script = script_dir / f"{script_name}.py"
        if not main_script.exists():
            print(f"  Error: Script not found: {main_script}", file=sys.stderr)
            failed.append(str(yaml_path))
            continue
        print(f"\n[{i}/{len(yaml_files)}] Running {yaml_path.name} with {script_name}.py...")
        if not ensure_mask_if_needed(
            config, yaml_path, project_root, args.gpu, args.dry_run
        ):
            failed.append(str(yaml_path))
            continue
        cmd = [
            sys.executable,
            str(main_script),
            "--config",
            str(yaml_path.resolve()),
            "--gpu",
            str(args.gpu),
        ]
        if args.dry_run:
            print(f"  Would run: {' '.join(cmd)}")
            continue
        result = subprocess.run(cmd, cwd=str(project_root))
        if result.returncode != 0:
            print(f"  Failed with exit code {result.returncode}", file=sys.stderr)
            failed.append(str(yaml_path))

    if failed:
        print("\n" + "=" * 60)
        print(f"Failed configs: {len(failed)}")
        for p in failed:
            print(f"  - {p}")
        sys.exit(1)
    elif not args.dry_run:
        print("\n" + "=" * 60)
        print(f"All {len(yaml_files)} configs completed successfully.")


if __name__ == "__main__":
    main()
