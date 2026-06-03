#!/usr/bin/env python3
"""
Checkpointed reproduction pipeline for HolUE.

The pipeline is intentionally an orchestration layer: it runs existing training,
prediction, evaluation, and table/CSV collection commands from a YAML file.

Main features
-------------
1. Stage-level checkpointing.
2. Resume after crash: completed stages are skipped.
3. Output validation after every stage.
4. Separate logs per stage.
5. Optional in-process PyTorch-Lightning/Hydra training stage with checkpoint resume.
6. Works with shell-command stages for existing repo scripts.

Typical usage
-------------
python pipeline_config.py --config pipeline_config.yaml
python pipeline_config.py --config pipeline_config.yaml --list
python pipeline_config.py --config pipeline_config.yaml --dry-run
python pipeline_config.py --config pipeline_config.yaml --from-stage eval_main_table
python pipeline_config.py --config pipeline_config.yaml --force-stage eval_main_table
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as _dt
import glob
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from string import Template
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union


try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


class PipelineError(RuntimeError):
    pass


# ---------------------------------------------------------------------
# YAML / config helpers
# ---------------------------------------------------------------------


def load_yaml_config(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Config file does not exist: {path}")

    if yaml is not None:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data or {}

    # Fallback: OmegaConf is already a project dependency because Hydra is used.
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(path)
    return OmegaConf.to_container(cfg, resolve=False)  # type: ignore[return-value]


def parse_scalar(value: str) -> Any:
    """Parse CLI override value into bool/int/float/None/list/dict/string."""
    lowered = value.strip().lower()
    if lowered in {"none", "null"}:
        return None
    if lowered in {"true", "false"}:
        return lowered == "true"

    try:
        return json.loads(value)
    except Exception:
        return value


def set_plain_dotted(
    container: Union[Dict[str, Any], List[Any]], dotted: str, value: Any
) -> None:
    """Set key like 'vars.dataset_root=/data' in a plain dict/list config."""
    parts = dotted.split(".")
    cur: Any = container

    for token in parts[:-1]:
        if isinstance(cur, list):
            cur = cur[int(token)]
        else:
            if token not in cur or cur[token] is None:
                cur[token] = {}
            cur = cur[token]

    last = parts[-1]
    if isinstance(cur, list):
        cur[int(last)] = value
    else:
        cur[last] = value


def build_context(raw_cfg: Mapping[str, Any], config_path: Path) -> Dict[str, str]:
    """Build string-substitution context for ${var} placeholders."""
    project_root_raw = raw_cfg.get("project_root", str(config_path.parent))
    project_root = Path(str(project_root_raw)).expanduser()
    if not project_root.is_absolute():
        project_root = (config_path.parent / project_root).resolve()

    ctx: Dict[str, str] = {
        "project_root": str(project_root),
        "python": str(raw_cfg.get("python", sys.executable)),
    }

    # User variables.
    for k, v in dict(raw_cfg.get("vars", {}) or {}).items():
        ctx[k] = str(v)

    # Frequently useful top-level paths.
    for k in ("checkpoint_dir", "log_dir"):
        if k in raw_cfg:
            ctx[k] = str(raw_cfg[k])

    # Iteratively resolve nested ${...}.
    for _ in range(20):
        changed = False
        for k, v in list(ctx.items()):
            new_v = Template(v).safe_substitute(ctx)
            if new_v != v:
                ctx[k] = new_v
                changed = True
        if not changed:
            break

    return ctx


def render_placeholders(obj: Any, ctx: Mapping[str, str]) -> Any:
    if isinstance(obj, str):
        return Template(obj).safe_substitute(ctx)
    if isinstance(obj, list):
        return [render_placeholders(x, ctx) for x in obj]
    if isinstance(obj, dict):
        return {k: render_placeholders(v, ctx) for k, v in obj.items()}
    return obj


def as_path(path_like: Union[str, Path], project_root: Union[str, Path]) -> Path:
    p = Path(str(path_like)).expanduser()
    if not p.is_absolute():
        p = Path(project_root) / p
    return p


def now_iso() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


def json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


# ---------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------


def marker_path(cfg: Mapping[str, Any], stage_name: str) -> Path:
    return as_path(cfg["checkpoint_dir"], cfg["project_root"]) / f"{stage_name}.json"


def read_marker(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def write_marker(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=json_default)
    tmp.replace(path)


def is_glob_pattern(s: str) -> bool:
    return any(ch in s for ch in "*?[")


def expand_output_pattern(pattern: str, project_root: Union[str, Path]) -> List[Path]:
    if is_glob_pattern(pattern):
        p = Path(pattern)
        pat = str(p if p.is_absolute() else Path(project_root) / p)
        return [Path(x) for x in glob.glob(pat, recursive=True)]

    return [as_path(pattern, project_root)]


def missing_paths(paths: Iterable[str], project_root: Union[str, Path]) -> List[str]:
    missing: List[str] = []
    for pattern in paths:
        expanded = expand_output_pattern(str(pattern), project_root)
        if is_glob_pattern(str(pattern)):
            if not expanded:
                missing.append(str(pattern))
        else:
            p = expanded[0]
            if not p.exists():
                missing.append(str(p))
    return missing


def stage_outputs_exist(stage: Mapping[str, Any], cfg: Mapping[str, Any]) -> bool:
    outputs = stage.get("outputs", []) or []
    if isinstance(outputs, str):
        outputs = [outputs]
    return len(missing_paths(outputs, cfg["project_root"])) == 0


def stage_is_complete(stage: Mapping[str, Any], cfg: Mapping[str, Any]) -> bool:
    m = read_marker(marker_path(cfg, stage["name"]))
    if not m or m.get("status") != "complete":
        return False

    if stage.get("check_outputs", True):
        return stage_outputs_exist(stage, cfg)

    return True


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------


class Tee:
    """Write stdout/stderr both to terminal and to a file."""

    def __init__(self, *streams: Any):
        self.streams = streams

    def write(self, data: str) -> None:
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self) -> None:
        for s in self.streams:
            s.flush()


def stage_log_path(cfg: Mapping[str, Any], stage_name: str) -> Path:
    return as_path(cfg["log_dir"], cfg["project_root"]) / f"{stage_name}.log"


# ---------------------------------------------------------------------
# Stage implementations
# ---------------------------------------------------------------------


def merged_env(cfg: Mapping[str, Any], stage: Mapping[str, Any]) -> Dict[str, str]:
    env = os.environ.copy()

    for k, v in dict(cfg.get("global_env", {}) or {}).items():
        env[str(k)] = str(v)

    for k, v in dict(stage.get("env", {}) or {}).items():
        env[str(k)] = str(v)

    return env


def run_command_stage(
    stage: Mapping[str, Any], cfg: Mapping[str, Any], log_path: Path
) -> None:
    command = stage.get("command")
    if command is None:
        raise PipelineError(f"Stage '{stage['name']}' has no command.")

    cwd = as_path(stage.get("cwd", cfg["project_root"]), cfg["project_root"])
    cwd.mkdir(parents=True, exist_ok=True)

    env = merged_env(cfg, stage)

    shell = isinstance(command, str)
    printable_command = command if shell else " ".join(map(str, command))

    print(f"[{stage['name']}] cwd: {cwd}")
    print(f"[{stage['name']}] command: {printable_command}")

    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("a", encoding="utf-8") as log_f:
        log_f.write(f"\n\n# ===== {now_iso()} | stage={stage['name']} =====\n")
        log_f.write(f"# cwd: {cwd}\n")
        log_f.write(f"# command: {printable_command}\n\n")
        log_f.flush()

        if shell:
            proc = subprocess.Popen(
                command,
                cwd=str(cwd),
                env=env,
                shell=True,
                executable="/bin/bash",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        else:
            proc = subprocess.Popen(
                [str(x) for x in command],
                cwd=str(cwd),
                env=env,
                shell=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log_f.write(line)

        ret = proc.wait()
        log_f.write(f"\n# return_code: {ret}\n")
        log_f.flush()

    if ret != 0:
        raise PipelineError(
            f"Command stage '{stage['name']}' failed with return code {ret}."
        )


def run_check_paths_stage(stage: Mapping[str, Any], cfg: Mapping[str, Any]) -> None:
    required = stage.get("paths", []) or []
    if isinstance(required, str):
        required = [required]

    optional = stage.get("optional_paths", []) or []
    if isinstance(optional, str):
        optional = [optional]

    missing_required = missing_paths(required, cfg["project_root"])
    missing_optional = missing_paths(optional, cfg["project_root"])

    if missing_optional:
        print(f"[{stage['name']}] Optional paths missing:")
        for p in missing_optional:
            print(f"  - {p}")

    if missing_required:
        msg = "\n".join(f"  - {p}" for p in missing_required)
        raise FileNotFoundError(
            f"Required paths are missing for stage '{stage['name']}':\n{msg}"
        )

    print(f"[{stage['name']}] All required paths exist.")


def run_copy_stage(stage: Mapping[str, Any], cfg: Mapping[str, Any]) -> None:
    files = stage.get("files", []) or []
    if not files:
        raise PipelineError(f"Copy stage '{stage['name']}' has no files.")

    for item in files:
        src = as_path(item["src"], cfg["project_root"])
        dst = as_path(item["dst"], cfg["project_root"])

        if not src.is_file():
            raise FileNotFoundError(f"Copy source does not exist: {src}")

        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"[{stage['name']}] copied: {src} -> {dst}")


def run_collect_csvs_stage(stage: Mapping[str, Any], cfg: Mapping[str, Any]) -> None:
    patterns = stage.get("patterns", []) or []
    if isinstance(patterns, str):
        patterns = [patterns]

    destination = as_path(stage["destination"], cfg["project_root"])
    destination.mkdir(parents=True, exist_ok=True)

    collected: List[Dict[str, str]] = []
    project_root = Path(cfg["project_root"])

    for pattern in patterns:
        pat = (
            str(as_path(pattern, cfg["project_root"]))
            if not Path(str(pattern)).is_absolute()
            else str(pattern)
        )
        for src_str in glob.glob(pat, recursive=True):
            src = Path(src_str)
            if not src.is_file():
                continue

            try:
                rel = src.relative_to(project_root)
            except ValueError:
                rel = Path(src.name)

            dst = destination / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            collected.append({"src": str(src), "dst": str(dst)})

    manifest = {
        "created_at": now_iso(),
        "num_files": len(collected),
        "files": collected,
    }

    manifest_path = destination / "MANIFEST.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[{stage['name']}] collected {len(collected)} CSV files.")
    print(f"[{stage['name']}] manifest: {manifest_path}")


def set_omegaconf_dotted(cfg: Any, dotted: str, value: Any) -> None:
    """Set OmegaConf key/list-index by dotted path."""
    parts = dotted.split(".")
    cur = cfg

    for token in parts[:-1]:
        if token.isdigit():
            cur = cur[int(token)]
        else:
            if token not in cur or cur[token] is None:
                cur[token] = {}
            cur = cur[token]

    last = parts[-1]
    if last.isdigit():
        cur[int(last)] = value
    else:
        cur[last] = value


def find_lightning_last_checkpoint(
    trainer: Any, cfg_obj: Any, project_root: Union[str, Path]
) -> Optional[Path]:
    candidates: List[Path] = []

    for cb in getattr(trainer, "callbacks", []) or []:
        last_model_path = getattr(cb, "last_model_path", None)
        if last_model_path:
            candidates.append(as_path(last_model_path, project_root))

        dirpath = getattr(cb, "dirpath", None)
        if dirpath:
            candidates.append(as_path(Path(dirpath) / "last.ckpt", project_root))

    default_root_dir = getattr(trainer, "default_root_dir", None)
    if default_root_dir:
        candidates.append(as_path(Path(default_root_dir) / "last.ckpt", project_root))

    try:
        cfg_default_root = cfg_obj.trainer.default_root_dir
        candidates.append(as_path(Path(cfg_default_root) / "last.ckpt", project_root))
    except Exception:
        pass

    for p in candidates:
        if p.is_file():
            return p

    return None


def run_lightning_stage(stage: Mapping[str, Any], cfg: Mapping[str, Any]) -> None:
    """
    Optional in-process Lightning/Hydra stage.

    This is useful for training uncertainty heads because it can resume from
    last.ckpt via trainer.fit(..., ckpt_path=...).
    """
    try:
        import torch
        from hydra.utils import instantiate
        from omegaconf import OmegaConf
        from pytorch_lightning import seed_everything
    except Exception as e:
        raise PipelineError(
            "Lightning stage requires torch, hydra, omegaconf, and pytorch_lightning."
        ) from e

    config_path = as_path(stage["config"], cfg["project_root"])
    if not config_path.is_file():
        raise FileNotFoundError(f"Lightning config not found: {config_path}")

    pl_cfg = OmegaConf.load(config_path)

    for key, value in dict(stage.get("overrides", {}) or {}).items():
        set_omegaconf_dotted(pl_cfg, key, value)

    if "mode" in stage:
        pl_cfg.mode = stage["mode"]

    seed = int(pl_cfg.get("seed_everything", 777))
    seed_everything(seed, workers=True)

    trainer = instantiate(pl_cfg.trainer)
    model = instantiate(pl_cfg.model)

    weights_path = stage.get("weights_path", pl_cfg.get("weights_path", None))
    if weights_path:
        weights_path = as_path(weights_path, cfg["project_root"])
        if not weights_path.is_file():
            raise FileNotFoundError(f"weights_path does not exist: {weights_path}")

        checkpoint = torch.load(
            str(weights_path), map_location="cpu", weights_only=False
        )
        state_dict = checkpoint.get("state_dict", checkpoint)
        strict = bool(stage.get("strict_load", True))
        model.load_state_dict(state_dict, strict=strict)
        print(
            f"[{stage['name']}] Loaded weights from {weights_path} with strict={strict}"
        )

    datamodule = instantiate(pl_cfg.data)
    mode = str(pl_cfg.mode)

    if mode == "train":
        resume_ckpt = stage.get("resume_checkpoint", None)
        if resume_ckpt:
            resume_ckpt = as_path(resume_ckpt, cfg["project_root"])
        else:
            resume_ckpt = find_lightning_last_checkpoint(
                trainer, pl_cfg, cfg["project_root"]
            )

        if bool(stage.get("resume", True)) and resume_ckpt and resume_ckpt.is_file():
            print(f"[{stage['name']}] Resuming Lightning training from {resume_ckpt}")
            trainer.fit(model=model, datamodule=datamodule, ckpt_path=str(resume_ckpt))
        else:
            print(f"[{stage['name']}] Starting Lightning training from scratch.")
            trainer.fit(model=model, datamodule=datamodule)

    elif mode == "predict":
        trainer.predict(model=model, datamodule=datamodule)

    else:
        raise PipelineError(f"Unknown Lightning mode: {mode}")


def run_stage_payload(
    stage: Mapping[str, Any], cfg: Mapping[str, Any], log_path: Path
) -> None:
    stage_type = stage.get("type", "command")

    if stage_type == "command":
        run_command_stage(stage, cfg, log_path)
    elif stage_type == "check_paths":
        run_check_paths_stage(stage, cfg)
    elif stage_type == "copy":
        run_copy_stage(stage, cfg)
    elif stage_type == "collect_csvs":
        run_collect_csvs_stage(stage, cfg)
    elif stage_type == "lightning":
        run_lightning_stage(stage, cfg)
    else:
        raise PipelineError(
            f"Unsupported stage type '{stage_type}' in stage '{stage['name']}'."
        )


# ---------------------------------------------------------------------
# Pipeline control
# ---------------------------------------------------------------------


def enabled_stages(cfg: Mapping[str, Any]) -> List[Dict[str, Any]]:
    stages = cfg.get("stages", []) or []
    result: List[Dict[str, Any]] = []

    for st in stages:
        if st.get("enabled", True):
            result.append(st)

    return result


def select_stages(
    stages: List[Dict[str, Any]],
    only: Optional[Sequence[str]],
    from_stage: Optional[str],
) -> List[Dict[str, Any]]:
    if only:
        wanted = set(only)
        return [s for s in stages if s["name"] in wanted]

    if from_stage:
        names = [s["name"] for s in stages]
        if from_stage not in names:
            raise PipelineError(f"--from-stage '{from_stage}' not found.")
        idx = names.index(from_stage)
        return stages[idx:]

    return stages


def assert_dependencies(stage: Mapping[str, Any], cfg: Mapping[str, Any]) -> None:
    deps = stage.get("depends_on", []) or []
    if isinstance(deps, str):
        deps = [deps]

    missing = []
    for dep in deps:
        dep_stage = {"name": dep, "outputs": [], "check_outputs": False}
        m = read_marker(marker_path(cfg, dep))
        if not m or m.get("status") != "complete":
            missing.append(dep)

    if missing:
        raise PipelineError(
            f"Stage '{stage['name']}' depends on incomplete stages: {', '.join(missing)}"
        )


def run_one_stage(
    stage: Dict[str, Any],
    cfg: Dict[str, Any],
    *,
    force: bool,
    force_stage_names: Sequence[str],
    dry_run: bool,
) -> None:
    name = stage["name"]
    mpath = marker_path(cfg, name)
    log_path = stage_log_path(cfg, name)

    force_this = force or name in set(force_stage_names)

    if not force_this and stage_is_complete(stage, cfg):
        print(f"[SKIP] {name} is already complete.")
        return

    assert_dependencies(stage, cfg)

    print(f"\n========== STAGE: {name} ==========")
    print(f"type: {stage.get('type', 'command')}")
    print(f"log:  {log_path}")
    print(f"ckpt: {mpath}")

    if dry_run:
        print(f"[DRY-RUN] Would run stage '{name}'.")
        return

    started = time.time()
    write_marker(
        mpath,
        {
            "stage": name,
            "status": "running",
            "started_at": now_iso(),
            "log_path": str(log_path),
            "stage_config": stage,
        },
    )

    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # For internal Python stages, tee stdout/stderr into the stage log.
        if stage.get("type", "command") == "command":
            run_stage_payload(stage, cfg, log_path)
        else:
            with log_path.open("a", encoding="utf-8") as log_f:
                log_f.write(f"\n\n# ===== {now_iso()} | stage={name} =====\n")
                log_f.flush()
                with contextlib.redirect_stdout(Tee(sys.stdout, log_f)):
                    with contextlib.redirect_stderr(Tee(sys.stderr, log_f)):
                        run_stage_payload(stage, cfg, log_path)

        if stage.get("check_outputs", True):
            outputs = stage.get("outputs", []) or []
            if isinstance(outputs, str):
                outputs = [outputs]
            missing = missing_paths(outputs, cfg["project_root"])
            if missing:
                msg = "\n".join(f"  - {p}" for p in missing)
                raise PipelineError(
                    f"Stage '{name}' finished but outputs are missing:\n{msg}"
                )

        elapsed = time.time() - started
        write_marker(
            mpath,
            {
                "stage": name,
                "status": "complete",
                "started_at": _dt.datetime.fromtimestamp(started).isoformat(
                    timespec="seconds"
                ),
                "finished_at": now_iso(),
                "duration_sec": round(elapsed, 3),
                "log_path": str(log_path),
                "outputs": stage.get("outputs", []),
                "stage_config": stage,
            },
        )
        print(f"[DONE] {name} in {elapsed:.1f}s")

    except Exception as e:
        elapsed = time.time() - started
        err = traceback.format_exc()
        write_marker(
            mpath,
            {
                "stage": name,
                "status": "failed",
                "started_at": _dt.datetime.fromtimestamp(started).isoformat(
                    timespec="seconds"
                ),
                "failed_at": now_iso(),
                "duration_sec": round(elapsed, 3),
                "log_path": str(log_path),
                "error": repr(e),
                "traceback": err,
                "stage_config": stage,
            },
        )
        print(f"[FAILED] {name}")
        print(err)
        raise


def print_stage_list(
    stages: Sequence[Mapping[str, Any]], cfg: Mapping[str, Any]
) -> None:
    print("Enabled stages:")
    for i, st in enumerate(stages):
        status = "complete" if stage_is_complete(st, cfg) else "pending"
        print(
            f"{i:02d}. {st['name']:<35} type={st.get('type', 'command'):<12} status={status}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="pipeline_config.yaml", help="Pipeline YAML file."
    )
    parser.add_argument(
        "--list", action="store_true", help="List enabled stages and exit."
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print stages without executing."
    )
    parser.add_argument(
        "--force", action="store_true", help="Rerun all selected stages."
    )
    parser.add_argument(
        "--force-stage",
        action="append",
        default=[],
        help="Rerun one named stage. Can be repeated.",
    )
    parser.add_argument(
        "--only",
        default=None,
        help="Comma-separated list of stage names to run.",
    )
    parser.add_argument(
        "--from-stage",
        default=None,
        help="Run from this stage to the end, respecting existing checkpoints.",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override YAML key, e.g. --set vars.dataset_root=/data",
    )
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    raw_cfg = load_yaml_config(config_path)

    for assignment in args.set:
        if "=" not in assignment:
            raise PipelineError(
                f"Bad --set override '{assignment}', expected KEY=VALUE."
            )
        key, value = assignment.split("=", 1)
        set_plain_dotted(raw_cfg, key, parse_scalar(value))

    ctx = build_context(raw_cfg, config_path)
    cfg = render_placeholders(raw_cfg, ctx)
    cfg["project_root"] = ctx["project_root"]
    cfg["checkpoint_dir"] = cfg.get(
        "checkpoint_dir",
        str(Path(ctx["project_root"]) / "outputs/pipeline/checkpoints"),
    )
    cfg["log_dir"] = cfg.get(
        "log_dir", str(Path(ctx["project_root"]) / "outputs/pipeline/logs")
    )

    Path(cfg["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["log_dir"]).mkdir(parents=True, exist_ok=True)

    stages = enabled_stages(cfg)
    only = args.only.split(",") if args.only else None
    selected = select_stages(stages, only=only, from_stage=args.from_stage)

    if args.list:
        print_stage_list(stages, cfg)
        return

    print(f"Project root:    {cfg['project_root']}")
    print(f"Checkpoint dir:  {cfg['checkpoint_dir']}")
    print(f"Log dir:         {cfg['log_dir']}")
    print(f"Selected stages: {[s['name'] for s in selected]}")

    for stage in selected:
        run_one_stage(
            stage,
            cfg,
            force=args.force,
            force_stage_names=args.force_stage,
            dry_run=args.dry_run,
        )

    print("\nPipeline finished successfully.")


if __name__ == "__main__":
    main()
