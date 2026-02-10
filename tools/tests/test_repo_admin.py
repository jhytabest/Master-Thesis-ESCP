from __future__ import annotations

import importlib.util
from argparse import Namespace
from pathlib import Path
import sys
import unittest


MODULE_PATH = Path(__file__).resolve().parents[1] / "repo_admin.py"
SPEC = importlib.util.spec_from_file_location("repo_admin", MODULE_PATH)
assert SPEC and SPEC.loader
repo_admin = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = repo_admin
SPEC.loader.exec_module(repo_admin)


class RepoAdminCliTests(unittest.TestCase):
    def test_analysis_all_builds_ordered_specs(self) -> None:
        specs = repo_admin.build_analysis_command_specs(step="all", python_exec="python3")
        self.assertEqual(len(specs), 4)
        self.assertTrue(specs[0].cmd[-1].endswith("eda_initial.py"))
        self.assertTrue(specs[1].cmd[-1].endswith("correlation_analysis.py"))
        self.assertTrue(specs[2].cmd[-1].endswith("regression_analysis.py"))
        self.assertTrue(specs[3].cmd[-1].endswith("robustness_checks.py"))

    def test_mapping_propose_command_contains_required_flags(self) -> None:
        args = Namespace(
            version_id="v1",
            output_prefix="mappings/proposals/v1",
            model="gemini-3-flash-preview",
            python="python3",
        )
        spec = repo_admin.build_mapping_propose_spec(args)
        self.assertEqual(spec.cmd[:3], ["python3", "-m", "mapping.propose"])
        self.assertIn("--version_id", spec.cmd)
        self.assertIn("v1", spec.cmd)
        self.assertIn("--output_prefix", spec.cmd)
        self.assertIn("mappings/proposals/v1", spec.cmd)
        self.assertIn("--model", spec.cmd)
        self.assertIn("gemini-3-flash-preview", spec.cmd)
        self.assertIn("PYTHONPATH", spec.env)
        self.assertIn("LOCAL_STORAGE_ROOT", spec.env)

    def test_pipeline_command_optional_flags(self) -> None:
        args = Namespace(
            version_id="ver-001",
            mapping_bundle_id="bundle-001",
            run_id="run-001",
            python="python3",
        )
        spec = repo_admin.build_pipeline_spec(args)
        self.assertEqual(spec.cmd[:3], ["python3", "-m", "pipeline.cli"])
        self.assertIn("--version_id", spec.cmd)
        self.assertIn("ver-001", spec.cmd)
        self.assertIn("--mapping_bundle_id", spec.cmd)
        self.assertIn("bundle-001", spec.cmd)
        self.assertIn("--run_id", spec.cmd)
        self.assertIn("run-001", spec.cmd)
        self.assertIn("LOCAL_STORAGE_ROOT", spec.env)

    def test_parser_worker_typecheck_install(self) -> None:
        parser = repo_admin.build_parser()
        args = parser.parse_args(["worker", "typecheck", "--install"])
        self.assertEqual(args.command, "worker")
        self.assertEqual(args.worker_cmd, "typecheck")
        self.assertTrue(args.install)

    def test_parser_analysis_run_step(self) -> None:
        parser = repo_admin.build_parser()
        args = parser.parse_args(["analysis", "run", "--step", "regression", "--dry-run"])
        self.assertEqual(args.command, "analysis")
        self.assertEqual(args.analysis_cmd, "run")
        self.assertEqual(args.step, "regression")
        self.assertTrue(args.dry_run)

    def test_parser_local_register_version(self) -> None:
        parser = repo_admin.build_parser()
        args = parser.parse_args(["local", "register-version", "--version-id", "local-main"])
        self.assertEqual(args.command, "local")
        self.assertEqual(args.local_cmd, "register-version")
        self.assertEqual(args.version_id, "local-main")

    def test_parser_analysis_literature(self) -> None:
        parser = repo_admin.build_parser()
        args = parser.parse_args(
            [
                "analysis",
                "literature",
                "--finding-source",
                "NORTHSTAR.md",
                "--max-findings",
                "3",
                "--works-per-finding",
                "2",
                "--dry-run",
            ]
        )
        self.assertEqual(args.command, "analysis")
        self.assertEqual(args.analysis_cmd, "literature")
        self.assertEqual(args.finding_source, ["NORTHSTAR.md"])
        self.assertEqual(args.max_findings, 3)
        self.assertEqual(args.works_per_finding, 2)
        self.assertTrue(args.dry_run)

    def test_extract_findings_from_markdown(self) -> None:
        content = """
# Report
## Key Findings
- MBA founders raise more funding.
- Female founder effect is borderline.
| Finding | Strength |
|---|---|
| Founder count predicts funding | Moderate |
"""
        findings = repo_admin._extract_findings_from_markdown(content, source_name="sample.md")
        finding_texts = [item["finding"] for item in findings]
        self.assertIn("MBA founders raise more funding.", finding_texts)
        self.assertIn("Female founder effect is borderline.", finding_texts)
        self.assertIn("Founder count predicts funding", finding_texts)

    def test_derive_openalex_query_adds_domain_terms(self) -> None:
        query = repo_admin._derive_openalex_query("MBA → more funding & rounds")
        self.assertNotIn("→", query)
        self.assertIn("startup funding", query)
        self.assertIn("venture capital", query)
        self.assertIn("founder human capital", query)


if __name__ == "__main__":
    unittest.main()
