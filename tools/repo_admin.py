#!/usr/bin/env python3
"""Repository administration CLI for Master-Thesis-ESCP.

This CLI provides a single command surface to orchestrate all core repository
workflows:
- analysis scripts
- mapping workflows
- pipeline execution
- enrichment execution
- Worker API utility actions
- Streamlit UI launch
- Cloud Build invocations
- local diagnostics
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import error as urlerror
from urllib import request as urlrequest


REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_SRC_DIR = REPO_ROOT / "jobs" / "analysis" / "src"
ANALYSIS_OUT_DIR = REPO_ROOT / "jobs" / "analysis" / "output"


@dataclass
class CommandSpec:
    cmd: list[str]
    cwd: Path
    env: dict[str, str] | None = None


def _print_header(message: str) -> None:
    print(f"\n==> {message}", flush=True)


def _run_command(spec: CommandSpec, dry_run: bool = False) -> int:
    cmd_display = " ".join(spec.cmd)
    print(f"$ (cd {spec.cwd} && {cmd_display})", flush=True)
    if dry_run:
        return 0

    env = os.environ.copy()
    if spec.env:
        env.update(spec.env)

    process = subprocess.run(spec.cmd, cwd=spec.cwd, env=env)
    return process.returncode


def _run_specs(specs: list[CommandSpec], dry_run: bool = False) -> int:
    for spec in specs:
        code = _run_command(spec, dry_run=dry_run)
        if code != 0:
            return code
    return 0


def _python_module_spec(
    module: str,
    args: list[str],
    cwd: Path,
    python_exec: str,
    src_dir: Path | None = None,
) -> CommandSpec:
    env: dict[str, str] | None = None
    if src_dir is not None:
        existing = os.getenv("PYTHONPATH", "")
        src = str(src_dir)
        env = {"PYTHONPATH": f"{src}:{existing}" if existing else src}
    return CommandSpec(cmd=[python_exec, "-m", module, *args], cwd=cwd, env=env)


def _script_spec(script_path: Path, python_exec: str) -> CommandSpec:
    return CommandSpec(cmd=[python_exec, str(script_path)], cwd=REPO_ROOT)


def build_analysis_command_specs(step: str, python_exec: str) -> list[CommandSpec]:
    scripts = {
        "eda": ANALYSIS_SRC_DIR / "eda_initial.py",
        "correlation": ANALYSIS_SRC_DIR / "correlation_analysis.py",
        "regression": ANALYSIS_SRC_DIR / "regression_analysis.py",
        "robustness": ANALYSIS_SRC_DIR / "robustness_checks.py",
    }

    if step == "all":
        order = ["eda", "correlation", "regression", "robustness"]
    else:
        order = [step]

    return [_script_spec(scripts[name], python_exec=python_exec) for name in order]


def build_mapping_propose_spec(args: argparse.Namespace) -> CommandSpec:
    module_args = ["--version_id", args.version_id]
    if args.output_prefix:
        module_args.extend(["--output_prefix", args.output_prefix])
    if args.model:
        module_args.extend(["--model", args.model])

    return _python_module_spec(
        module="mapping.propose",
        args=module_args,
        cwd=REPO_ROOT / "jobs" / "mapping",
        python_exec=args.python,
        src_dir=REPO_ROOT / "jobs" / "mapping" / "src",
    )


def build_mapping_freeze_spec(args: argparse.Namespace) -> CommandSpec:
    module_args = ["--output_bundle_id", args.output_bundle_id]

    if args.proposal_prefix:
        module_args.extend(["--proposal_prefix", args.proposal_prefix])
    if args.proposal_path:
        module_args.extend(["--proposal_path", args.proposal_path])
    if args.approved_by:
        module_args.extend(["--approved_by", args.approved_by])
    if args.approved_at:
        module_args.extend(["--approved_at", args.approved_at])
    if args.output_prefix:
        module_args.extend(["--output_prefix", args.output_prefix])

    return _python_module_spec(
        module="mapping.freeze",
        args=module_args,
        cwd=REPO_ROOT / "jobs" / "mapping",
        python_exec=args.python,
        src_dir=REPO_ROOT / "jobs" / "mapping" / "src",
    )


def build_pipeline_spec(args: argparse.Namespace) -> CommandSpec:
    module_args = ["--version_id", args.version_id]
    if args.mapping_bundle_id:
        module_args.extend(["--mapping_bundle_id", args.mapping_bundle_id])
    if args.run_id:
        module_args.extend(["--run_id", args.run_id])

    return _python_module_spec(
        module="pipeline.cli",
        args=module_args,
        cwd=REPO_ROOT / "jobs" / "pipeline",
        python_exec=args.python,
        src_dir=REPO_ROOT / "jobs" / "pipeline" / "src",
    )


def build_enrichment_spec(args: argparse.Namespace) -> CommandSpec:
    module_args = ["--version_id", args.version_id, "--source", args.source]
    return _python_module_spec(
        module="enrichment.cli",
        args=module_args,
        cwd=REPO_ROOT / "jobs" / "enrichment",
        python_exec=args.python,
        src_dir=REPO_ROOT / "jobs" / "enrichment" / "src",
    )


def build_worker_typecheck_specs(args: argparse.Namespace) -> list[CommandSpec]:
    worker_dir = REPO_ROOT / "apps" / "worker-api"
    specs: list[CommandSpec] = []
    if args.install:
        specs.append(CommandSpec(cmd=["npm", "install"], cwd=worker_dir))
    specs.append(CommandSpec(cmd=["npm", "run", "typecheck"], cwd=worker_dir))
    return specs


def _component_requirements() -> dict[str, dict[str, list[str]]]:
    return {
        "analysis": {
            "env": [],
            "bins": ["python3"],
            "python_imports": [
                "numpy",
                "pandas",
                "matplotlib",
                "seaborn",
                "scipy",
                "statsmodels",
            ],
        },
        "mapping": {
            "env": ["R2_ENDPOINT", "R2_BUCKET", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "GCP_PROJECT_ID"],
            "bins": ["python3"],
            "python_imports": ["boto3", "pandas", "google.genai"],
        },
        "pipeline": {
            "env": ["R2_ENDPOINT", "R2_BUCKET", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY"],
            "bins": ["python3"],
            "python_imports": ["boto3", "pandas", "sklearn", "pyarrow"],
        },
        "enrichment": {
            "env": [],
            "bins": ["python3"],
            "python_imports": ["pandas", "pyarrow"],
        },
        "worker": {
            "env": [],
            "bins": ["node", "npm"],
            "python_imports": [],
        },
        "ui": {
            "env": [
                "WORKER_API_BASE",
                "R2_ENDPOINT",
                "R2_BUCKET",
                "R2_ACCESS_KEY_ID",
                "R2_SECRET_ACCESS_KEY",
                "GOOGLE_OAUTH_CLIENT_ID",
                "GOOGLE_OAUTH_CLIENT_SECRET",
                "GOOGLE_OAUTH_REDIRECT_URI",
            ],
            "bins": ["python3"],
            "python_imports": ["streamlit", "boto3", "pandas", "requests", "google.auth"],
        },
    }


def _check_python_imports(imports: list[str], python_exec: str) -> tuple[bool, list[str]]:
    if not imports:
        return True, []

    check_lines = [
        "import importlib.util",
        f"mods = {imports!r}",
        "missing = [m for m in mods if importlib.util.find_spec(m) is None]",
        "print('\\n'.join(missing))",
        "raise SystemExit(1 if missing else 0)",
    ]

    proc = subprocess.run(
        [python_exec, "-c", "\n".join(check_lines)],
        capture_output=True,
        text=True,
    )
    missing = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    return proc.returncode == 0, missing


def _require_worker_api_base() -> str:
    base = os.getenv("WORKER_API_BASE")
    if not base:
        raise RuntimeError("Missing WORKER_API_BASE")
    return base.rstrip("/")


def _worker_headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    token = os.getenv("WORKER_API_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    cf_id = os.getenv("CF_ACCESS_CLIENT_ID")
    cf_secret = os.getenv("CF_ACCESS_CLIENT_SECRET")
    if cf_id and cf_secret:
        headers["CF-Access-Client-Id"] = cf_id
        headers["CF-Access-Client-Secret"] = cf_secret
    return headers


def _api_request(method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    base = _require_worker_api_base()
    url = f"{base}{path}"
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urlrequest.Request(url, data=data, method=method)
    for key, value in _worker_headers().items():
        req.add_header(key, value)

    try:
        with urlrequest.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw) if raw else {}
    except urlerror.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"HTTP {exc.code}: {details}") from exc


def cmd_status(_: argparse.Namespace) -> int:
    _print_header("Repository status")
    specs = [
        CommandSpec(cmd=["git", "status", "--short", "--branch"], cwd=REPO_ROOT),
        CommandSpec(cmd=["git", "log", "--oneline", "-n", "1"], cwd=REPO_ROOT),
    ]
    code = _run_specs(specs)
    if code != 0:
        return code

    output_files = sorted(ANALYSIS_OUT_DIR.glob("*")) if ANALYSIS_OUT_DIR.exists() else []
    print(f"analysis_output_files={len(output_files)}")
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    requirements = _component_requirements()
    components = list(requirements.keys()) if args.component == "all" else [args.component]

    failures = 0
    for component in components:
        req = requirements[component]
        _print_header(f"doctor::{component}")

        for env_name in req["env"]:
            if os.getenv(env_name):
                print(f"[ok] env {env_name}")
            else:
                print(f"[missing] env {env_name}")
                failures += 1

        for binary in req["bins"]:
            if shutil.which(binary):
                print(f"[ok] bin {binary}")
            else:
                print(f"[missing] bin {binary}")
                failures += 1

        ok_imports, missing_imports = _check_python_imports(req["python_imports"], python_exec=args.python)
        if ok_imports:
            if req["python_imports"]:
                print(f"[ok] python imports ({len(req['python_imports'])})")
        else:
            for name in missing_imports:
                print(f"[missing] python import {name}")
            failures += len(missing_imports)

    return 1 if failures else 0


def cmd_analysis_run(args: argparse.Namespace) -> int:
    _print_header(f"analysis run step={args.step}")
    specs = build_analysis_command_specs(step=args.step, python_exec=args.python)
    return _run_specs(specs, dry_run=args.dry_run)


def cmd_analysis_list_outputs(_: argparse.Namespace) -> int:
    _print_header("analysis outputs")
    if not ANALYSIS_OUT_DIR.exists():
        print("No output directory found.")
        return 0

    files = sorted([p for p in ANALYSIS_OUT_DIR.iterdir() if p.is_file()])
    if not files:
        print("No analysis output files found.")
        return 0

    for path in files:
        print(f"{path.relative_to(REPO_ROOT)}\t{path.stat().st_size} bytes")
    return 0


def cmd_analysis_clean_outputs(args: argparse.Namespace) -> int:
    _print_header("analysis clean outputs")
    if not ANALYSIS_OUT_DIR.exists():
        print("Output directory does not exist.")
        return 0

    files = [p for p in ANALYSIS_OUT_DIR.iterdir() if p.is_file()]
    if not files:
        print("No files to remove.")
        return 0

    if not args.yes:
        print("Refusing to delete without --yes.")
        return 1

    for path in files:
        path.unlink()
        print(f"removed {path.relative_to(REPO_ROOT)}")
    return 0


def cmd_mapping_propose(args: argparse.Namespace) -> int:
    _print_header(f"mapping propose version_id={args.version_id}")
    spec = build_mapping_propose_spec(args)
    return _run_command(spec, dry_run=args.dry_run)


def cmd_mapping_freeze(args: argparse.Namespace) -> int:
    if not args.proposal_prefix and not args.proposal_path:
        print("Provide either --proposal-prefix or --proposal-path", file=sys.stderr)
        return 2

    _print_header(f"mapping freeze output_bundle_id={args.output_bundle_id}")
    spec = build_mapping_freeze_spec(args)
    return _run_command(spec, dry_run=args.dry_run)


def cmd_mapping_test(args: argparse.Namespace) -> int:
    _print_header("mapping tests")
    return _run_specs(
        [
            CommandSpec(
                cmd=[args.python, "-m", "pytest", "-v", "--tb=short"],
                cwd=REPO_ROOT / "jobs" / "mapping",
                env={"PYTHONPATH": str(REPO_ROOT / "jobs" / "mapping" / "src")},
            )
        ],
        dry_run=args.dry_run,
    )


def cmd_pipeline_run(args: argparse.Namespace) -> int:
    _print_header(f"pipeline run version_id={args.version_id}")
    spec = build_pipeline_spec(args)
    return _run_command(spec, dry_run=args.dry_run)


def cmd_enrichment_run(args: argparse.Namespace) -> int:
    _print_header(f"enrichment run version_id={args.version_id} source={args.source}")
    spec = build_enrichment_spec(args)
    return _run_command(spec, dry_run=args.dry_run)


def cmd_worker_typecheck(args: argparse.Namespace) -> int:
    _print_header("worker typecheck")
    specs = build_worker_typecheck_specs(args)
    return _run_specs(specs, dry_run=args.dry_run)


def cmd_worker_dev(args: argparse.Namespace) -> int:
    _print_header("worker dev")
    return _run_command(
        CommandSpec(cmd=["npm", "run", "dev"], cwd=REPO_ROOT / "apps" / "worker-api"),
        dry_run=args.dry_run,
    )


def cmd_worker_deploy(args: argparse.Namespace) -> int:
    _print_header("worker deploy")
    return _run_command(
        CommandSpec(cmd=["npm", "run", "deploy"], cwd=REPO_ROOT / "apps" / "worker-api"),
        dry_run=args.dry_run,
    )


def cmd_ui_run(args: argparse.Namespace) -> int:
    _print_header(f"ui run host={args.host} port={args.port}")
    cmd = [
        args.python,
        "-m",
        "streamlit",
        "run",
        "app.py",
        "--server.address",
        args.host,
        "--server.port",
        str(args.port),
    ]
    return _run_command(
        CommandSpec(cmd=cmd, cwd=REPO_ROOT / "apps" / "research-ui"),
        dry_run=args.dry_run,
    )


def cmd_api_list_runs(_: argparse.Namespace) -> int:
    _print_header("api list-runs")
    payload = _api_request("GET", "/runs")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def cmd_api_list_versions(_: argparse.Namespace) -> int:
    _print_header("api list-versions")
    payload = _api_request("GET", "/versions")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def cmd_api_list_mapping_bundles(_: argparse.Namespace) -> int:
    _print_header("api list-mapping-bundles")
    payload = _api_request("GET", "/mapping_bundles")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def cmd_api_trigger_run(args: argparse.Namespace) -> int:
    _print_header(f"api trigger-run version_id={args.version_id}")
    body: dict[str, Any] = {"version_id": args.version_id}
    if args.mapping_bundle_id:
        body["mapping_bundle_id"] = args.mapping_bundle_id
    if args.run_id:
        body["run_id"] = args.run_id
    payload = _api_request("POST", "/runs/trigger", payload=body)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def cmd_cloudbuild_submit(args: argparse.Namespace) -> int:
    _print_header(f"cloudbuild submit target={args.target}")
    config = REPO_ROOT / "cloudbuild" / f"{args.target}.yaml"
    if not config.exists():
        print(f"Missing config: {config}", file=sys.stderr)
        return 2

    cmd = ["gcloud", "builds", "submit", "--config", str(config), "."]
    if args.project:
        cmd.extend(["--project", args.project])
    if args.substitutions:
        cmd.extend(["--substitutions", args.substitutions])

    return _run_command(CommandSpec(cmd=cmd, cwd=REPO_ROOT), dry_run=args.dry_run)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Master-Thesis-ESCP administrative CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    sub = parser.add_subparsers(dest="command", required=True)

    p_status = sub.add_parser("status", help="Show repo and analysis artifact status")
    p_status.set_defaults(func=cmd_status)

    p_doctor = sub.add_parser("doctor", help="Check environment/tooling readiness")
    p_doctor.add_argument(
        "--component",
        choices=["all", "analysis", "mapping", "pipeline", "enrichment", "worker", "ui"],
        default="all",
    )
    p_doctor.add_argument("--python", default=sys.executable, help="Python executable for import checks")
    p_doctor.set_defaults(func=cmd_doctor)

    p_analysis = sub.add_parser("analysis", help="Run thesis analysis workflows")
    analysis_sub = p_analysis.add_subparsers(dest="analysis_cmd", required=True)

    p_analysis_run = analysis_sub.add_parser("run", help="Run analysis scripts")
    p_analysis_run.add_argument("--step", choices=["eda", "correlation", "regression", "robustness", "all"], default="all")
    p_analysis_run.add_argument("--python", default=sys.executable)
    p_analysis_run.add_argument("--dry-run", action="store_true")
    p_analysis_run.set_defaults(func=cmd_analysis_run)

    p_analysis_ls = analysis_sub.add_parser("list-outputs", help="List generated analysis outputs")
    p_analysis_ls.set_defaults(func=cmd_analysis_list_outputs)

    p_analysis_clean = analysis_sub.add_parser("clean-outputs", help="Delete generated analysis outputs")
    p_analysis_clean.add_argument("--yes", action="store_true", help="Confirm deletion")
    p_analysis_clean.set_defaults(func=cmd_analysis_clean_outputs)

    p_mapping = sub.add_parser("mapping", help="Run mapping workflow commands")
    mapping_sub = p_mapping.add_subparsers(dest="mapping_cmd", required=True)

    p_mapping_propose = mapping_sub.add_parser("propose", help="Generate mapping proposals")
    p_mapping_propose.add_argument("--version-id", required=True)
    p_mapping_propose.add_argument("--output-prefix")
    p_mapping_propose.add_argument("--model")
    p_mapping_propose.add_argument("--python", default=sys.executable)
    p_mapping_propose.add_argument("--dry-run", action="store_true")
    p_mapping_propose.set_defaults(func=cmd_mapping_propose)

    p_mapping_freeze = mapping_sub.add_parser("freeze", help="Freeze approved mapping bundle")
    p_mapping_freeze.add_argument("--proposal-prefix")
    p_mapping_freeze.add_argument("--proposal-path")
    p_mapping_freeze.add_argument("--output-bundle-id", required=True)
    p_mapping_freeze.add_argument("--approved-by")
    p_mapping_freeze.add_argument("--approved-at")
    p_mapping_freeze.add_argument("--output-prefix")
    p_mapping_freeze.add_argument("--python", default=sys.executable)
    p_mapping_freeze.add_argument("--dry-run", action="store_true")
    p_mapping_freeze.set_defaults(func=cmd_mapping_freeze)

    p_mapping_test = mapping_sub.add_parser("test", help="Run mapping job tests")
    p_mapping_test.add_argument("--python", default=sys.executable)
    p_mapping_test.add_argument("--dry-run", action="store_true")
    p_mapping_test.set_defaults(func=cmd_mapping_test)

    p_pipeline = sub.add_parser("pipeline", help="Run main pipeline")
    pipeline_sub = p_pipeline.add_subparsers(dest="pipeline_cmd", required=True)

    p_pipeline_run = pipeline_sub.add_parser("run", help="Execute pipeline job")
    p_pipeline_run.add_argument("--version-id", required=True)
    p_pipeline_run.add_argument("--mapping-bundle-id")
    p_pipeline_run.add_argument("--run-id")
    p_pipeline_run.add_argument("--python", default=sys.executable)
    p_pipeline_run.add_argument("--dry-run", action="store_true")
    p_pipeline_run.set_defaults(func=cmd_pipeline_run)

    p_enrichment = sub.add_parser("enrichment", help="Run enrichment workflow")
    enrichment_sub = p_enrichment.add_subparsers(dest="enrichment_cmd", required=True)

    p_enrichment_run = enrichment_sub.add_parser("run", help="Execute enrichment job")
    p_enrichment_run.add_argument("--version-id", required=True)
    p_enrichment_run.add_argument("--source", required=True)
    p_enrichment_run.add_argument("--python", default=sys.executable)
    p_enrichment_run.add_argument("--dry-run", action="store_true")
    p_enrichment_run.set_defaults(func=cmd_enrichment_run)

    p_worker = sub.add_parser("worker", help="Worker API development commands")
    worker_sub = p_worker.add_subparsers(dest="worker_cmd", required=True)

    p_worker_typecheck = worker_sub.add_parser("typecheck", help="Run worker TypeScript checks")
    p_worker_typecheck.add_argument("--install", action="store_true", help="Run npm install before typecheck")
    p_worker_typecheck.add_argument("--dry-run", action="store_true")
    p_worker_typecheck.set_defaults(func=cmd_worker_typecheck)

    p_worker_dev = worker_sub.add_parser("dev", help="Run wrangler dev")
    p_worker_dev.add_argument("--dry-run", action="store_true")
    p_worker_dev.set_defaults(func=cmd_worker_dev)

    p_worker_deploy = worker_sub.add_parser("deploy", help="Deploy worker")
    p_worker_deploy.add_argument("--dry-run", action="store_true")
    p_worker_deploy.set_defaults(func=cmd_worker_deploy)

    p_ui = sub.add_parser("ui", help="Research UI commands")
    ui_sub = p_ui.add_subparsers(dest="ui_cmd", required=True)

    p_ui_run = ui_sub.add_parser("run", help="Run Streamlit UI")
    p_ui_run.add_argument("--host", default="0.0.0.0")
    p_ui_run.add_argument("--port", type=int, default=8501)
    p_ui_run.add_argument("--python", default=sys.executable)
    p_ui_run.add_argument("--dry-run", action="store_true")
    p_ui_run.set_defaults(func=cmd_ui_run)

    p_api = sub.add_parser("api", help="Worker API utility calls")
    api_sub = p_api.add_subparsers(dest="api_cmd", required=True)

    p_api_runs = api_sub.add_parser("list-runs", help="List runs from Worker API")
    p_api_runs.set_defaults(func=cmd_api_list_runs)

    p_api_versions = api_sub.add_parser("list-versions", help="List versions from Worker API")
    p_api_versions.set_defaults(func=cmd_api_list_versions)

    p_api_bundles = api_sub.add_parser("list-mapping-bundles", help="List mapping bundles from Worker API")
    p_api_bundles.set_defaults(func=cmd_api_list_mapping_bundles)

    p_api_trigger = api_sub.add_parser("trigger-run", help="Create a run in Worker API")
    p_api_trigger.add_argument("--version-id", required=True)
    p_api_trigger.add_argument("--mapping-bundle-id")
    p_api_trigger.add_argument("--run-id")
    p_api_trigger.set_defaults(func=cmd_api_trigger_run)

    p_cloudbuild = sub.add_parser("cloudbuild", help="Cloud Build utility")
    cloudbuild_sub = p_cloudbuild.add_subparsers(dest="cloudbuild_cmd", required=True)

    p_cloudbuild_submit = cloudbuild_sub.add_parser("submit", help="Submit a cloudbuild config")
    p_cloudbuild_submit.add_argument("--target", choices=["pipeline", "mapping", "enrichment", "research-ui"], required=True)
    p_cloudbuild_submit.add_argument("--project")
    p_cloudbuild_submit.add_argument("--substitutions")
    p_cloudbuild_submit.add_argument("--dry-run", action="store_true")
    p_cloudbuild_submit.set_defaults(func=cmd_cloudbuild_submit)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        return int(args.func(args))
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
