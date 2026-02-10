from __future__ import annotations

import importlib.util
from argparse import Namespace
from pathlib import Path
import sys
import tempfile
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

    def test_parser_research_ingest_latest(self) -> None:
        parser = repo_admin.build_parser()
        args = parser.parse_args(["research", "ingest-openalex", "--latest"])
        self.assertEqual(args.command, "research")
        self.assertEqual(args.research_cmd, "ingest-openalex")
        self.assertTrue(args.latest)
        self.assertTrue(args.rebuild_sqlite)

    def test_parser_research_add_edge(self) -> None:
        parser = repo_admin.build_parser()
        args = parser.parse_args(
            [
                "research",
                "add-edge",
                "--from-paper-id",
                "W1",
                "--to-paper-id",
                "W2",
                "--relation",
                "extends",
            ]
        )
        self.assertEqual(args.command, "research")
        self.assertEqual(args.research_cmd, "add-edge")
        self.assertEqual(args.from_paper_id, "W1")
        self.assertEqual(args.to_paper_id, "W2")
        self.assertEqual(args.relation, "extends")

    def test_ingest_openalex_payload_builds_links_and_dependencies(self) -> None:
        state = {
            "papers": [],
            "claims": [],
            "claim_paper_links": [],
            "paper_edges": [],
            "dependencies": [],
            "ingestions": [],
        }
        payload = {
            "run_id": "run-1",
            "items": [
                {
                    "source": "NORTHSTAR.md",
                    "finding": "MBA founders raise more funding",
                    "query": "mba startup funding",
                    "works": [
                        {
                            "id": "https://openalex.org/W123",
                            "title": "Paper A",
                            "publication_year": 2020,
                            "cited_by_count": 10,
                            "doi": "https://doi.org/example-a",
                            "source": "Journal A",
                            "authors": ["Author A"],
                            "concepts": ["Venture capital"],
                            "referenced_works": ["https://openalex.org/W999"],
                        },
                        {
                            "id": "https://openalex.org/W124",
                            "title": "Paper B",
                            "publication_year": 2021,
                            "cited_by_count": 7,
                            "doi": "https://doi.org/example-b",
                            "source": "Journal B",
                            "authors": ["Author B"],
                            "concepts": ["Startups"],
                            "referenced_works": [],
                        },
                    ],
                }
            ],
        }
        created = repo_admin._ingest_openalex_report_payload(
            state=state,
            payload=payload,
            report_path="/tmp/openalex_run-1.json",
            run_id="run-1",
            link_relation="supports",
            max_dependencies_per_paper=5,
        )
        self.assertEqual(created["claims"], 1)
        self.assertEqual(created["papers"], 2)
        self.assertEqual(created["links"], 2)
        self.assertGreaterEqual(created["edges"], 2)  # cites + co_supports_claim
        self.assertEqual(created["dependencies"], 1)
        self.assertEqual(len(state["claims"]), 1)
        self.assertEqual(len(state["claim_paper_links"]), 2)
        self.assertTrue(any(item["paper_id"] == "W999" and item["is_placeholder"] for item in state["papers"]))
        self.assertTrue(any(item["paper_id"] == "W123" and item["quality_tier"] == "emerging" for item in state["papers"]))

    def test_research_overview_markdown_contains_sections(self) -> None:
        state = {
            "papers": [{"paper_id": "W1", "title": "Paper One", "cited_by_count": 5, "publication_year": 2020}],
            "claims": [{"claim_id": "claim_1", "claim_text": "A claim"}],
            "claim_paper_links": [{"claim_id": "claim_1", "paper_id": "W1"}],
            "paper_edges": [{"relation": "co_supports_claim", "from_paper_id": "W1", "to_paper_id": "W2"}],
            "dependencies": [{"paper_id": "W1", "depends_on_paper_id": "W2", "reason": "openalex_referenced_work"}],
            "ingestions": [],
        }
        md = repo_admin._build_research_overview_markdown(state, generated_at="2026-02-10T00:00:00Z")
        self.assertIn("# Thesis Academic Foundation Overview", md)
        self.assertIn("## Claim Coverage", md)
        self.assertIn("## Quality Tiers", md)
        self.assertIn("## Interaction Breakdown", md)
        self.assertIn("## Deep Reading Paths (sample)", md)

    def test_resolve_openalex_report_paths_latest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            reports = root / "reports" / "literature"
            reports.mkdir(parents=True)
            first = reports / "openalex_first.json"
            second = reports / "openalex_second.json"
            first.write_text("{}", encoding="utf-8")
            second.write_text("{}", encoding="utf-8")

            resolved = repo_admin._resolve_openalex_report_paths(
                local_root=root,
                report_paths=None,
                latest=True,
            )
            self.assertEqual(len(resolved), 1)
            self.assertEqual(resolved[0], second)


if __name__ == "__main__":
    unittest.main()
