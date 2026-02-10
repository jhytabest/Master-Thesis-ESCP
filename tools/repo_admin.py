#!/usr/bin/env python3
"""Repository administration CLI for Master-Thesis-ESCP.

This CLI provides a single command surface to orchestrate all core repository
workflows:
- analysis scripts
- analysis-to-literature mapping (OpenAlex)
- iterative research foundation database (claims, papers, interactions, dependencies)
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
import csv
import hashlib
import itertools
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import error as urlerror
from urllib import parse as urlparse
from urllib import request as urlrequest


REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_SRC_DIR = REPO_ROOT / "jobs" / "analysis" / "src"
ANALYSIS_OUT_DIR = REPO_ROOT / "jobs" / "analysis" / "output"
NORTHSTAR_PATH = REPO_ROOT / "NORTHSTAR.md"
LOCAL_STORAGE_ROOT_DEFAULT = REPO_ROOT / "local_store"
RESEARCH_DB_NAME = "research.db"


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


def _resolve_local_storage_root(value: str | None = None) -> Path:
    root = Path(value or os.getenv("LOCAL_STORAGE_ROOT") or str(LOCAL_STORAGE_ROOT_DEFAULT)).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _workflow_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = {
        "LOCAL_STORAGE_ROOT": str(_resolve_local_storage_root()),
        "R2_BUCKET": "__local__",
    }
    if extra:
        env.update(extra)
    return env


def _resolve_executable(executable: str) -> str:
    path = Path(executable)
    if path.is_absolute():
        return str(path)
    if "/" in executable:
        candidate = (REPO_ROOT / path).resolve()
        if candidate.exists():
            return str(candidate)
    return executable


def _python_module_spec(
    module: str,
    args: list[str],
    cwd: Path,
    python_exec: str,
    src_dir: Path | None = None,
    extra_env: dict[str, str] | None = None,
) -> CommandSpec:
    python_exec = _resolve_executable(python_exec)
    env: dict[str, str] = {}
    if src_dir is not None:
        existing = os.getenv("PYTHONPATH", "")
        src = str(src_dir)
        env["PYTHONPATH"] = f"{src}{os.pathsep}{existing}" if existing else src
    if extra_env:
        env.update(extra_env)
    return CommandSpec(cmd=[python_exec, "-m", module, *args], cwd=cwd, env=env or None)


def _script_spec(script_path: Path, python_exec: str) -> CommandSpec:
    return CommandSpec(cmd=[_resolve_executable(python_exec), str(script_path)], cwd=REPO_ROOT)


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


def _normalize_text(value: str) -> str:
    return " ".join(value.split())


def _strip_markdown(value: str) -> str:
    cleaned = value.strip()
    cleaned = re.sub(r"`([^`]+)`", r"\1", cleaned)
    cleaned = re.sub(r"\*\*([^*]+)\*\*", r"\1", cleaned)
    cleaned = re.sub(r"\*([^*]+)\*", r"\1", cleaned)
    cleaned = cleaned.strip(" -|")
    return _normalize_text(cleaned)


def _query_safe_text(value: str) -> str:
    return _normalize_text(re.sub(r"[^0-9A-Za-z]+", " ", value))


def _default_literature_sources() -> list[Path]:
    sources: list[Path] = []
    if NORTHSTAR_PATH.exists():
        sources.append(NORTHSTAR_PATH)
    if ANALYSIS_OUT_DIR.exists():
        sources.extend(sorted(ANALYSIS_OUT_DIR.glob("*.md")))
    return sources


def _resolve_source_path(path_str: str) -> Path:
    candidate = Path(path_str).expanduser()
    if not candidate.is_absolute():
        candidate = (REPO_ROOT / candidate).resolve()
    return candidate


def _resolve_literature_sources(raw_sources: list[str] | None) -> list[Path]:
    if not raw_sources:
        return _default_literature_sources()
    return [_resolve_source_path(item) for item in raw_sources]


def _extract_findings_from_markdown(content: str, source_name: str) -> list[dict[str, str]]:
    heading = ""
    finding_headings = ("finding", "takeaway", "priority", "hypoth", "result")
    findings: list[dict[str, str]] = []

    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if line.startswith("#"):
            heading = _strip_markdown(line.lstrip("#"))
            continue

        heading_lc = heading.lower()
        in_finding_section = any(token in heading_lc for token in finding_headings)

        if in_finding_section and line.startswith(("- ", "* ")):
            text = _strip_markdown(line[2:])
            if text:
                findings.append({"source": source_name, "finding": text})
            continue

        if in_finding_section:
            match = re.match(r"^\*\*(.+?)\*\*:\s*(.+)$", line)
            if match:
                text = f"{_strip_markdown(match.group(1))}: {_strip_markdown(match.group(2))}"
                findings.append({"source": source_name, "finding": text})
                continue

            if line.startswith("|") and line.count("|") >= 3:
                cells = [_strip_markdown(cell) for cell in line.strip("|").split("|")]
                if not cells or not cells[0]:
                    continue
                header_cell = cells[0].lower()
                if header_cell in {"finding", "#", "hypothesis"} or set(header_cell) == {"-"}:
                    continue
                findings.append({"source": source_name, "finding": cells[0]})
                continue

        if line.startswith("- **") and "→" in line:
            text = _strip_markdown(line[2:])
            if text:
                findings.append({"source": source_name, "finding": text})

    deduped: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in findings:
        text = item["finding"]
        normalized = _normalize_text(text).lower()
        if len(normalized) < 16:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(item)
    return deduped


def _derive_openalex_query(finding: str) -> str:
    finding_clean = _strip_markdown(finding)
    finding_lc = finding_clean.lower()
    finding_query = _query_safe_text(finding_clean)

    thematic_terms: list[str] = ["startup funding", "venture capital"]
    if "mba" in finding_lc:
        thematic_terms.extend(["founder human capital", "business education"])
    if "female" in finding_lc or "gender" in finding_lc:
        thematic_terms.extend(["female founders", "gender bias"])
    if "scientific" in finding_lc or "ceo profile" in finding_lc:
        thematic_terms.extend(["technical founders", "deeptech"])
    if "founders_number" in finding_lc or "team size" in finding_lc or "founder count" in finding_lc:
        thematic_terms.append("founding team size")
    if "missing" in finding_lc or "selection bias" in finding_lc or "heckman" in finding_lc:
        thematic_terms.extend(["missing data", "selection bias"])
    if "industry" in finding_lc and "fixed" in finding_lc:
        thematic_terms.append("industry fixed effects")

    suffix = " ".join(dict.fromkeys(thematic_terms))
    return f"{finding_query} {suffix}".strip()


def _openalex_search(query: str, per_page: int, mailto: str | None, from_year: int | None) -> list[dict[str, Any]]:
    params: dict[str, str] = {
        "search": query,
        "sort": "relevance_score:desc",
        "per-page": str(per_page),
    }
    if mailto:
        params["mailto"] = mailto
    if from_year:
        params["filter"] = f"from_publication_date:{from_year}-01-01"

    url = f"https://api.openalex.org/works?{urlparse.urlencode(params)}"
    req = urlrequest.Request(url, method="GET", headers={"User-Agent": "master-thesis-admin-cli/1.0"})

    with urlrequest.urlopen(req, timeout=40) as resp:
        payload = json.loads(resp.read().decode("utf-8"))

    works: list[dict[str, Any]] = []
    for work in payload.get("results", []):
        source_info = (work.get("primary_location") or {}).get("source") or {}
        authors = []
        for authorship in work.get("authorships", [])[:5]:
            author_name = (authorship.get("author") or {}).get("display_name")
            if author_name:
                authors.append(author_name)
        concepts = []
        for concept in work.get("concepts", [])[:8]:
            concept_name = concept.get("display_name")
            if concept_name:
                concepts.append(concept_name)

        works.append(
            {
                "id": work.get("id"),
                "title": work.get("display_name"),
                "publication_year": work.get("publication_year"),
                "cited_by_count": work.get("cited_by_count"),
                "doi": work.get("doi"),
                "source": source_info.get("display_name"),
                "authors": authors,
                "concepts": concepts,
                "referenced_works": (work.get("referenced_works") or [])[:40],
                "openalex_url": work.get("id"),
            }
        )
    return works


def _build_literature_markdown(results: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# OpenAlex Literature Mapping")
    lines.append("")
    lines.append(f"- Run ID: {results.get('run_id')}")
    lines.append(f"- Generated at: {results['generated_at_utc']}")
    lines.append(f"- Sources scanned: {len(results['sources'])}")
    lines.append(f"- Findings mapped: {len(results['items'])}")
    lines.append("")

    for idx, item in enumerate(results["items"], start=1):
        lines.append(f"## Finding {idx}")
        lines.append("")
        lines.append(f"- Source: `{item['source']}`")
        lines.append(f"- Finding: {item['finding']}")
        lines.append(f"- OpenAlex query: `{item['query']}`")
        lines.append("")
        if not item["works"]:
            lines.append("- No works returned.")
            lines.append("")
            continue
        for work in item["works"]:
            title = work.get("title") or "(untitled)"
            year = work.get("publication_year") or "n/a"
            source = work.get("source") or "n/a"
            cited = work.get("cited_by_count")
            cited_label = cited if cited is not None else "n/a"
            lines.append(f"- {title} ({year}) | {source} | cited_by={cited_label}")
            if work.get("openalex_url"):
                lines.append(f"  {work['openalex_url']}")
        lines.append("")
    return "\n".join(lines)


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _stable_id(prefix: str, *parts: str) -> str:
    normalized_parts = [_normalize_text(part).lower() for part in parts if part]
    payload = "|".join(normalized_parts)
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def _paper_id_from_openalex_id(openalex_id: str | None) -> str:
    if openalex_id:
        token = openalex_id.rstrip("/").split("/")[-1]
        if token:
            return token
    return ""


def _paper_quality_tier(cited_by_count: int) -> str:
    if cited_by_count >= 500:
        return "foundational"
    if cited_by_count >= 150:
        return "high"
    if cited_by_count >= 40:
        return "medium"
    return "emerging"


def _research_root(local_root: Path) -> Path:
    return local_root / "research"


def _research_paths(local_root: Path) -> dict[str, Path]:
    root = _research_root(local_root)
    return {
        "root": root,
        "runs_dir": root / "runs",
        "papers": root / "papers.jsonl",
        "claims": root / "claims.jsonl",
        "claim_paper_links": root / "claim_paper_links.jsonl",
        "paper_edges": root / "paper_edges.jsonl",
        "dependencies": root / "dependencies.jsonl",
        "ingestions": root / "ingestions.jsonl",
        "overview_md": root / "foundation_overview.md",
        "db": root / RESEARCH_DB_NAME,
    }


def _ensure_jsonl_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("", encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def _count_jsonl_records(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def _load_research_state(local_root: Path) -> dict[str, list[dict[str, Any]]]:
    paths = _research_paths(local_root)
    for key in ("papers", "claims", "claim_paper_links", "paper_edges", "dependencies", "ingestions"):
        _ensure_jsonl_file(paths[key])
    paths["runs_dir"].mkdir(parents=True, exist_ok=True)
    return {
        "papers": _read_jsonl(paths["papers"]),
        "claims": _read_jsonl(paths["claims"]),
        "claim_paper_links": _read_jsonl(paths["claim_paper_links"]),
        "paper_edges": _read_jsonl(paths["paper_edges"]),
        "dependencies": _read_jsonl(paths["dependencies"]),
        "ingestions": _read_jsonl(paths["ingestions"]),
    }


def _save_research_state(local_root: Path, state: dict[str, list[dict[str, Any]]]) -> None:
    paths = _research_paths(local_root)
    _write_jsonl(paths["papers"], state["papers"])
    _write_jsonl(paths["claims"], state["claims"])
    _write_jsonl(paths["claim_paper_links"], state["claim_paper_links"])
    _write_jsonl(paths["paper_edges"], state["paper_edges"])
    _write_jsonl(paths["dependencies"], state["dependencies"])
    _write_jsonl(paths["ingestions"], state["ingestions"])


def _resolve_openalex_report_paths(local_root: Path, report_paths: list[str] | None, latest: bool) -> list[Path]:
    if report_paths:
        return [_resolve_source_path(path_str) for path_str in report_paths]

    literature_dir = local_root / "reports" / "literature"
    if not literature_dir.exists():
        return []

    all_reports = sorted(literature_dir.glob("openalex_*.json"), key=lambda path: path.stat().st_mtime)
    if not all_reports:
        return []
    if latest:
        return [all_reports[-1]]
    return all_reports


def _extract_run_id_from_report_path(path: Path) -> str:
    stem = path.stem
    if stem.startswith("openalex_"):
        return stem[len("openalex_") :]
    return _stable_id("run", stem)


def _upsert_paper(
    papers_by_id: dict[str, dict[str, Any]],
    work: dict[str, Any],
    now: str,
) -> tuple[str, bool]:
    openalex_id = work.get("id") or work.get("openalex_url")
    paper_id = _paper_id_from_openalex_id(openalex_id)
    if not paper_id:
        fallback = f"{work.get('title') or ''}|{work.get('publication_year') or ''}|{work.get('doi') or ''}"
        paper_id = _stable_id("paper", fallback)

    incoming_authors = work.get("authors") if isinstance(work.get("authors"), list) else []
    incoming_concepts = work.get("concepts") if isinstance(work.get("concepts"), list) else []
    incoming_refs = work.get("referenced_works") if isinstance(work.get("referenced_works"), list) else []
    incoming_cited_by = int(work.get("cited_by_count") or 0)

    record = papers_by_id.get(paper_id)
    created = record is None
    if record is None:
        record = {
            "paper_id": paper_id,
            "openalex_id": openalex_id,
            "title": work.get("title"),
            "publication_year": work.get("publication_year"),
            "doi": work.get("doi"),
            "source": work.get("source"),
            "authors": incoming_authors,
            "concepts": incoming_concepts,
            "cited_by_count": incoming_cited_by,
            "quality_tier": _paper_quality_tier(incoming_cited_by),
            "referenced_works": incoming_refs,
            "is_placeholder": False,
            "first_seen_at": now,
            "last_seen_at": now,
        }
    else:
        record["openalex_id"] = record.get("openalex_id") or openalex_id
        record["title"] = record.get("title") or work.get("title")
        record["publication_year"] = record.get("publication_year") or work.get("publication_year")
        record["doi"] = record.get("doi") or work.get("doi")
        record["source"] = record.get("source") or work.get("source")
        if incoming_authors:
            merged_authors = list(dict.fromkeys([*(record.get("authors") or []), *incoming_authors]))
            record["authors"] = merged_authors
        if incoming_concepts:
            merged_concepts = list(dict.fromkeys([*(record.get("concepts") or []), *incoming_concepts]))
            record["concepts"] = merged_concepts
        if incoming_refs:
            merged_refs = list(dict.fromkeys([*(record.get("referenced_works") or []), *incoming_refs]))
            record["referenced_works"] = merged_refs
        record["cited_by_count"] = max(int(record.get("cited_by_count") or 0), incoming_cited_by)
        record["quality_tier"] = _paper_quality_tier(int(record.get("cited_by_count") or 0))
        record["is_placeholder"] = bool(record.get("is_placeholder")) and not bool(work.get("title"))
        record["last_seen_at"] = now

    papers_by_id[paper_id] = record
    return paper_id, created


def _ensure_placeholder_paper(papers_by_id: dict[str, dict[str, Any]], openalex_id: str, now: str) -> tuple[str, bool]:
    paper_id = _paper_id_from_openalex_id(openalex_id)
    if not paper_id:
        paper_id = _stable_id("paper", openalex_id)

    existing = papers_by_id.get(paper_id)
    created = existing is None
    if existing is None:
        papers_by_id[paper_id] = {
            "paper_id": paper_id,
            "openalex_id": openalex_id,
            "title": None,
            "publication_year": None,
            "doi": None,
            "source": None,
            "authors": [],
            "concepts": [],
            "cited_by_count": 0,
            "quality_tier": "emerging",
            "referenced_works": [],
            "is_placeholder": True,
            "first_seen_at": now,
            "last_seen_at": now,
        }
    else:
        existing["last_seen_at"] = now
    return paper_id, created


def _upsert_claim(claims_by_id: dict[str, dict[str, Any]], claim_text: str, source: str, now: str) -> tuple[str, bool]:
    claim_id = _stable_id("claim", claim_text)
    record = claims_by_id.get(claim_id)
    created = record is None
    if record is None:
        claims_by_id[claim_id] = {
            "claim_id": claim_id,
            "claim_text": claim_text,
            "source_hints": [source],
            "origin": "analysis_literature",
            "first_seen_at": now,
            "last_seen_at": now,
        }
    else:
        record["source_hints"] = list(dict.fromkeys([*(record.get("source_hints") or []), source]))
        record["last_seen_at"] = now
    return claim_id, created


def _upsert_claim_link(
    links_by_key: dict[tuple[str, str, str], dict[str, Any]],
    claim_id: str,
    paper_id: str,
    relation: str,
    run_id: str,
    evidence_source: str,
    query: str,
    now: str,
) -> bool:
    key = (claim_id, paper_id, relation)
    existing = links_by_key.get(key)
    if existing is None:
        links_by_key[key] = {
            "link_id": _stable_id("link", claim_id, paper_id, relation),
            "claim_id": claim_id,
            "paper_id": paper_id,
            "relation": relation,
            "query": query,
            "evidence_source": evidence_source,
            "first_seen_at": now,
            "last_seen_at": now,
            "run_ids": [run_id],
        }
        return True
    existing["last_seen_at"] = now
    existing["query"] = existing.get("query") or query
    existing["evidence_source"] = existing.get("evidence_source") or evidence_source
    existing["run_ids"] = list(dict.fromkeys([*(existing.get("run_ids") or []), run_id]))
    return False


def _upsert_paper_edge(
    edges_by_key: dict[tuple[str, str, str, str], dict[str, Any]],
    from_paper_id: str,
    to_paper_id: str,
    relation: str,
    claim_id: str,
    provenance: str,
    run_id: str,
    note: str,
    now: str,
) -> bool:
    key = (from_paper_id, to_paper_id, relation, claim_id)
    existing = edges_by_key.get(key)
    if existing is None:
        edges_by_key[key] = {
            "edge_id": _stable_id("edge", from_paper_id, to_paper_id, relation, claim_id),
            "from_paper_id": from_paper_id,
            "to_paper_id": to_paper_id,
            "relation": relation,
            "claim_id": claim_id or None,
            "provenance": provenance,
            "note": note,
            "first_seen_at": now,
            "last_seen_at": now,
            "run_ids": [run_id] if run_id else [],
        }
        return True
    existing["last_seen_at"] = now
    if run_id:
        existing["run_ids"] = list(dict.fromkeys([*(existing.get("run_ids") or []), run_id]))
    return False


def _upsert_dependency(
    dependencies_by_key: dict[tuple[str, str, str], dict[str, Any]],
    paper_id: str,
    depends_on_paper_id: str,
    reason: str,
    depth: int,
    provenance: str,
    run_id: str,
    now: str,
) -> bool:
    key = (paper_id, depends_on_paper_id, reason)
    existing = dependencies_by_key.get(key)
    if existing is None:
        dependencies_by_key[key] = {
            "dependency_id": _stable_id("dep", paper_id, depends_on_paper_id, reason),
            "paper_id": paper_id,
            "depends_on_paper_id": depends_on_paper_id,
            "reason": reason,
            "depth": depth,
            "provenance": provenance,
            "first_seen_at": now,
            "last_seen_at": now,
            "run_ids": [run_id] if run_id else [],
        }
        return True
    existing["last_seen_at"] = now
    existing["depth"] = min(int(existing.get("depth") or depth), depth)
    if run_id:
        existing["run_ids"] = list(dict.fromkeys([*(existing.get("run_ids") or []), run_id]))
    return False


def _ingest_openalex_report_payload(
    state: dict[str, list[dict[str, Any]]],
    payload: dict[str, Any],
    report_path: str,
    run_id: str,
    link_relation: str,
    max_dependencies_per_paper: int,
) -> dict[str, int]:
    now = _utcnow_iso()
    papers_by_id = {item["paper_id"]: item for item in state["papers"] if item.get("paper_id")}
    claims_by_id = {item["claim_id"]: item for item in state["claims"] if item.get("claim_id")}
    links_by_key = {
        (item["claim_id"], item["paper_id"], item["relation"]): item
        for item in state["claim_paper_links"]
        if item.get("claim_id") and item.get("paper_id") and item.get("relation")
    }
    edges_by_key = {
        (item["from_paper_id"], item["to_paper_id"], item["relation"], item.get("claim_id") or ""): item
        for item in state["paper_edges"]
        if item.get("from_paper_id") and item.get("to_paper_id") and item.get("relation")
    }
    dependencies_by_key = {
        (item["paper_id"], item["depends_on_paper_id"], item["reason"]): item
        for item in state["dependencies"]
        if item.get("paper_id") and item.get("depends_on_paper_id") and item.get("reason")
    }

    created = {
        "papers": 0,
        "claims": 0,
        "links": 0,
        "edges": 0,
        "dependencies": 0,
        "placeholders": 0,
    }

    for item in payload.get("items", []):
        finding = _strip_markdown(str(item.get("finding") or ""))
        source = str(item.get("source") or "unknown")
        query = str(item.get("query") or "")
        if not finding:
            continue

        claim_id, claim_created = _upsert_claim(claims_by_id, claim_text=finding, source=source, now=now)
        if claim_created:
            created["claims"] += 1

        claim_paper_ids: list[str] = []
        for work in item.get("works", []):
            if not isinstance(work, dict):
                continue
            paper_id, paper_created = _upsert_paper(papers_by_id, work=work, now=now)
            if paper_created:
                created["papers"] += 1
            claim_paper_ids.append(paper_id)

            link_created = _upsert_claim_link(
                links_by_key=links_by_key,
                claim_id=claim_id,
                paper_id=paper_id,
                relation=link_relation,
                run_id=run_id,
                evidence_source=source,
                query=query,
                now=now,
            )
            if link_created:
                created["links"] += 1

            referenced = [ref for ref in (work.get("referenced_works") or []) if isinstance(ref, str)]
            for ref in referenced[:max(0, max_dependencies_per_paper)]:
                depends_on_paper_id, placeholder_created = _ensure_placeholder_paper(
                    papers_by_id=papers_by_id,
                    openalex_id=ref,
                    now=now,
                )
                if placeholder_created:
                    created["placeholders"] += 1

                dep_created = _upsert_dependency(
                    dependencies_by_key=dependencies_by_key,
                    paper_id=paper_id,
                    depends_on_paper_id=depends_on_paper_id,
                    reason="openalex_referenced_work",
                    depth=1,
                    provenance="auto_openalex",
                    run_id=run_id,
                    now=now,
                )
                if dep_created:
                    created["dependencies"] += 1

                edge_created = _upsert_paper_edge(
                    edges_by_key=edges_by_key,
                    from_paper_id=paper_id,
                    to_paper_id=depends_on_paper_id,
                    relation="cites",
                    claim_id="",
                    provenance="auto_openalex",
                    run_id=run_id,
                    note="Derived from OpenAlex referenced_works",
                    now=now,
                )
                if edge_created:
                    created["edges"] += 1

        # Undirected claim-level interaction edges to show related papers per finding.
        unique_claim_papers = sorted(set(claim_paper_ids))
        for left, right in itertools.combinations(unique_claim_papers, 2):
            edge_created = _upsert_paper_edge(
                edges_by_key=edges_by_key,
                from_paper_id=left,
                to_paper_id=right,
                relation="co_supports_claim",
                claim_id=claim_id,
                provenance="auto_claim_cluster",
                run_id=run_id,
                note="Both papers were retrieved for the same finding/claim.",
                now=now,
            )
            if edge_created:
                created["edges"] += 1

    state["papers"] = sorted(papers_by_id.values(), key=lambda item: item["paper_id"])
    state["claims"] = sorted(claims_by_id.values(), key=lambda item: item["claim_id"])
    state["claim_paper_links"] = sorted(
        links_by_key.values(),
        key=lambda item: (item["claim_id"], item["paper_id"], item["relation"]),
    )
    state["paper_edges"] = sorted(
        edges_by_key.values(),
        key=lambda item: (item["from_paper_id"], item["to_paper_id"], item["relation"], item.get("claim_id") or ""),
    )
    state["dependencies"] = sorted(
        dependencies_by_key.values(),
        key=lambda item: (item["paper_id"], item["depends_on_paper_id"], item["reason"]),
    )

    state["ingestions"].append(
        {
            "ingestion_id": _stable_id("ingest", run_id, report_path, now),
            "run_id": run_id,
            "report_path": report_path,
            "ingested_at": now,
            "new_papers": created["papers"],
            "new_claims": created["claims"],
            "new_links": created["links"],
            "new_edges": created["edges"],
            "new_dependencies": created["dependencies"],
            "new_placeholders": created["placeholders"],
        }
    )
    return created


def _rebuild_research_sqlite(local_root: Path, state: dict[str, list[dict[str, Any]]]) -> Path:
    paths = _research_paths(local_root)
    db_path = paths["db"]
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.executescript(
            """
            DROP TABLE IF EXISTS papers;
            DROP TABLE IF EXISTS claims;
            DROP TABLE IF EXISTS claim_paper_links;
            DROP TABLE IF EXISTS paper_edges;
            DROP TABLE IF EXISTS dependencies;
            DROP TABLE IF EXISTS ingestions;
            CREATE TABLE papers (
                paper_id TEXT PRIMARY KEY,
                title TEXT,
                publication_year INTEGER,
                source TEXT,
                doi TEXT,
                openalex_id TEXT,
                cited_by_count INTEGER,
                quality_tier TEXT,
                is_placeholder INTEGER,
                first_seen_at TEXT,
                last_seen_at TEXT,
                payload_json TEXT
            );
            CREATE TABLE claims (
                claim_id TEXT PRIMARY KEY,
                claim_text TEXT,
                origin TEXT,
                first_seen_at TEXT,
                last_seen_at TEXT,
                payload_json TEXT
            );
            CREATE TABLE claim_paper_links (
                link_id TEXT PRIMARY KEY,
                claim_id TEXT,
                paper_id TEXT,
                relation TEXT,
                first_seen_at TEXT,
                last_seen_at TEXT,
                payload_json TEXT
            );
            CREATE TABLE paper_edges (
                edge_id TEXT PRIMARY KEY,
                from_paper_id TEXT,
                to_paper_id TEXT,
                relation TEXT,
                claim_id TEXT,
                provenance TEXT,
                first_seen_at TEXT,
                last_seen_at TEXT,
                payload_json TEXT
            );
            CREATE TABLE dependencies (
                dependency_id TEXT PRIMARY KEY,
                paper_id TEXT,
                depends_on_paper_id TEXT,
                reason TEXT,
                depth INTEGER,
                provenance TEXT,
                first_seen_at TEXT,
                last_seen_at TEXT,
                payload_json TEXT
            );
            CREATE TABLE ingestions (
                ingestion_id TEXT PRIMARY KEY,
                run_id TEXT,
                report_path TEXT,
                ingested_at TEXT,
                payload_json TEXT
            );
            """
        )

        cur.executemany(
            """
            INSERT INTO papers (
                paper_id, title, publication_year, source, doi, openalex_id, cited_by_count, quality_tier, is_placeholder,
                first_seen_at, last_seen_at, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    item.get("paper_id"),
                    item.get("title"),
                    item.get("publication_year"),
                    item.get("source"),
                    item.get("doi"),
                    item.get("openalex_id"),
                    int(item.get("cited_by_count") or 0),
                    item.get("quality_tier") or _paper_quality_tier(int(item.get("cited_by_count") or 0)),
                    1 if item.get("is_placeholder") else 0,
                    item.get("first_seen_at"),
                    item.get("last_seen_at"),
                    json.dumps(item, ensure_ascii=False, sort_keys=True),
                )
                for item in state["papers"]
            ],
        )

        cur.executemany(
            "INSERT INTO claims (claim_id, claim_text, origin, first_seen_at, last_seen_at, payload_json) VALUES (?, ?, ?, ?, ?, ?)",
            [
                (
                    item.get("claim_id"),
                    item.get("claim_text"),
                    item.get("origin"),
                    item.get("first_seen_at"),
                    item.get("last_seen_at"),
                    json.dumps(item, ensure_ascii=False, sort_keys=True),
                )
                for item in state["claims"]
            ],
        )

        cur.executemany(
            """
            INSERT INTO claim_paper_links (
                link_id, claim_id, paper_id, relation, first_seen_at, last_seen_at, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    item.get("link_id"),
                    item.get("claim_id"),
                    item.get("paper_id"),
                    item.get("relation"),
                    item.get("first_seen_at"),
                    item.get("last_seen_at"),
                    json.dumps(item, ensure_ascii=False, sort_keys=True),
                )
                for item in state["claim_paper_links"]
            ],
        )

        cur.executemany(
            """
            INSERT INTO paper_edges (
                edge_id, from_paper_id, to_paper_id, relation, claim_id, provenance, first_seen_at, last_seen_at, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    item.get("edge_id"),
                    item.get("from_paper_id"),
                    item.get("to_paper_id"),
                    item.get("relation"),
                    item.get("claim_id"),
                    item.get("provenance"),
                    item.get("first_seen_at"),
                    item.get("last_seen_at"),
                    json.dumps(item, ensure_ascii=False, sort_keys=True),
                )
                for item in state["paper_edges"]
            ],
        )

        cur.executemany(
            """
            INSERT INTO dependencies (
                dependency_id, paper_id, depends_on_paper_id, reason, depth, provenance, first_seen_at, last_seen_at, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    item.get("dependency_id"),
                    item.get("paper_id"),
                    item.get("depends_on_paper_id"),
                    item.get("reason"),
                    int(item.get("depth") or 1),
                    item.get("provenance"),
                    item.get("first_seen_at"),
                    item.get("last_seen_at"),
                    json.dumps(item, ensure_ascii=False, sort_keys=True),
                )
                for item in state["dependencies"]
            ],
        )

        cur.executemany(
            "INSERT INTO ingestions (ingestion_id, run_id, report_path, ingested_at, payload_json) VALUES (?, ?, ?, ?, ?)",
            [
                (
                    item.get("ingestion_id"),
                    item.get("run_id"),
                    item.get("report_path"),
                    item.get("ingested_at"),
                    json.dumps(item, ensure_ascii=False, sort_keys=True),
                )
                for item in state["ingestions"]
            ],
        )

        cur.executescript(
            """
            CREATE INDEX idx_links_claim ON claim_paper_links(claim_id);
            CREATE INDEX idx_links_paper ON claim_paper_links(paper_id);
            CREATE INDEX idx_edges_from ON paper_edges(from_paper_id);
            CREATE INDEX idx_edges_to ON paper_edges(to_paper_id);
            CREATE INDEX idx_dependencies_paper ON dependencies(paper_id);
            CREATE INDEX idx_dependencies_depends ON dependencies(depends_on_paper_id);
            """
        )
        conn.commit()
    finally:
        conn.close()

    return db_path


def _build_research_overview_markdown(state: dict[str, list[dict[str, Any]]], generated_at: str) -> str:
    papers = state["papers"]
    claims = state["claims"]
    links = state["claim_paper_links"]
    edges = state["paper_edges"]
    dependencies = state["dependencies"]

    papers_by_id = {item.get("paper_id"): item for item in papers}
    links_by_claim: dict[str, list[dict[str, Any]]] = {}
    for link in links:
        links_by_claim.setdefault(link.get("claim_id"), []).append(link)

    lines: list[str] = []
    lines.append("# Thesis Academic Foundation Overview")
    lines.append("")
    lines.append(f"- Generated at: {generated_at}")
    lines.append(f"- Papers: {len(papers)}")
    lines.append(f"- Claims: {len(claims)}")
    lines.append(f"- Claim-paper links: {len(links)}")
    lines.append(f"- Paper edges: {len(edges)}")
    lines.append(f"- Dependencies: {len(dependencies)}")
    lines.append("")

    lines.append("## Top Papers (by citations)")
    lines.append("")
    top_papers = sorted(papers, key=lambda item: int(item.get("cited_by_count") or 0), reverse=True)[:15]
    if not top_papers:
        lines.append("- No papers recorded yet.")
    else:
        for paper in top_papers:
            title = paper.get("title") or "(placeholder)"
            cited_by = int(paper.get("cited_by_count") or 0)
            year = paper.get("publication_year") or "n/a"
            lines.append(f"- {paper.get('paper_id')}: {title} ({year}), cited_by={cited_by}")
    lines.append("")

    lines.append("## Quality Tiers")
    lines.append("")
    quality_counts: dict[str, int] = {}
    for paper in papers:
        tier = str(paper.get("quality_tier") or _paper_quality_tier(int(paper.get("cited_by_count") or 0)))
        quality_counts[tier] = quality_counts.get(tier, 0) + 1
    if not quality_counts:
        lines.append("- None.")
    else:
        for tier in ("foundational", "high", "medium", "emerging"):
            if tier in quality_counts:
                lines.append(f"- {tier}: {quality_counts[tier]}")
    lines.append("")

    lines.append("## Claim Coverage")
    lines.append("")
    if not claims:
        lines.append("- No claims available yet.")
    else:
        for claim in claims:
            claim_id = claim.get("claim_id")
            claim_links = links_by_claim.get(claim_id, [])
            lines.append(f"- {claim_id}: papers={len(claim_links)} | {claim.get('claim_text')}")
    lines.append("")

    lines.append("## Claims Missing Support")
    lines.append("")
    missing = [claim for claim in claims if not links_by_claim.get(claim.get("claim_id"))]
    if not missing:
        lines.append("- None.")
    else:
        for claim in missing:
            lines.append(f"- {claim.get('claim_id')}: {claim.get('claim_text')}")
    lines.append("")

    lines.append("## Interaction Breakdown")
    lines.append("")
    by_relation: dict[str, int] = {}
    for edge in edges:
        relation = edge.get("relation") or "unknown"
        by_relation[relation] = by_relation.get(relation, 0) + 1
    if not by_relation:
        lines.append("- No interaction edges yet.")
    else:
        for relation, count in sorted(by_relation.items()):
            lines.append(f"- {relation}: {count}")
    lines.append("")

    lines.append("## Deep Reading Paths (sample)")
    lines.append("")
    if not claims:
        lines.append("- No claims available.")
    else:
        for claim in claims[:8]:
            claim_id = claim.get("claim_id")
            claim_links = links_by_claim.get(claim_id, [])
            lines.append(f"### {claim_id}")
            lines.append("")
            lines.append(f"- Claim: {claim.get('claim_text')}")
            if not claim_links:
                lines.append("- No linked papers yet.")
                lines.append("")
                continue
            for link in claim_links[:4]:
                paper = papers_by_id.get(link.get("paper_id"), {})
                title = paper.get("title") or link.get("paper_id")
                lines.append(f"- Core: {title} ({link.get('paper_id')})")
                deps = [dep for dep in dependencies if dep.get("paper_id") == link.get("paper_id")][:3]
                if not deps:
                    lines.append("  deeper: none recorded")
                for dep in deps:
                    dep_paper = papers_by_id.get(dep.get("depends_on_paper_id"), {})
                    dep_title = dep_paper.get("title") or dep.get("depends_on_paper_id")
                    lines.append(f"  deeper: {dep_title} ({dep.get('depends_on_paper_id')}) [{dep.get('reason')}]")
            lines.append("")

    return "\n".join(lines)


def build_mapping_propose_spec(args: argparse.Namespace) -> CommandSpec:
    module_args = ["--version_id", args.version_id]
    if args.output_prefix:
        module_args.extend(["--output_prefix", args.output_prefix])
    model = args.model or "local-heuristic"
    module_args.extend(["--model", model])

    return _python_module_spec(
        module="mapping.propose",
        args=module_args,
        cwd=REPO_ROOT / "jobs" / "mapping",
        python_exec=args.python,
        src_dir=REPO_ROOT / "jobs" / "mapping" / "src",
        extra_env=_workflow_env(),
    )


def build_mapping_freeze_spec(args: argparse.Namespace) -> CommandSpec:
    module_args = ["--output_bundle_id", args.output_bundle_id]
    if getattr(args, "auto_approve_all", False):
        module_args.append("--auto_approve_all")

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
        extra_env=_workflow_env(),
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
        extra_env=_workflow_env(),
    )


def build_enrichment_spec(args: argparse.Namespace) -> CommandSpec:
    module_args = ["--version_id", args.version_id, "--source", args.source]
    return _python_module_spec(
        module="enrichment.cli",
        args=module_args,
        cwd=REPO_ROOT / "jobs" / "enrichment",
        python_exec=args.python,
        src_dir=REPO_ROOT / "jobs" / "enrichment" / "src",
        extra_env=_workflow_env(),
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
            "env": [],
            "bins": ["python3"],
            "python_imports": ["pandas"],
        },
        "pipeline": {
            "env": [],
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
    local_root = _resolve_local_storage_root()
    local_files = [p for p in local_root.rglob("*") if p.is_file()] if local_root.exists() else []
    print(f"local_storage_root={local_root}")
    print(f"local_storage_files={len(local_files)}")
    research_paths = _research_paths(local_root)
    print(f"research_papers={_count_jsonl_records(research_paths['papers'])}")
    print(f"research_claims={_count_jsonl_records(research_paths['claims'])}")
    print(f"research_links={_count_jsonl_records(research_paths['claim_paper_links'])}")
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    requirements = _component_requirements()
    components = list(requirements.keys()) if args.component == "all" else [args.component]
    local_root = _resolve_local_storage_root()
    print(f"[info] local storage root: {local_root}")

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


def cmd_analysis_literature(args: argparse.Namespace) -> int:
    _print_header("analysis literature")
    sources = _resolve_literature_sources(args.finding_source)
    if not sources:
        print("No default finding sources were found.", file=sys.stderr)
        return 2

    missing_sources = [path for path in sources if not path.exists()]
    if missing_sources:
        print("Missing finding sources:", file=sys.stderr)
        for path in missing_sources:
            print(f"  - {path}", file=sys.stderr)
        return 2

    findings: list[dict[str, str]] = []
    for source in sources:
        content = source.read_text(encoding="utf-8")
        findings.extend(_extract_findings_from_markdown(content, source_name=str(source.relative_to(REPO_ROOT))))

    if not findings:
        print("No findings extracted from provided markdown sources.", file=sys.stderr)
        return 2

    max_findings = max(1, args.max_findings)
    findings = findings[:max_findings]

    output_items: list[dict[str, Any]] = []
    for finding in findings:
        query = _derive_openalex_query(finding["finding"])
        works: list[dict[str, Any]] = []
        if not args.dry_run:
            try:
                works = _openalex_search(
                    query=query,
                    per_page=args.works_per_finding,
                    mailto=args.mailto,
                    from_year=args.from_year,
                )
            except (urlerror.HTTPError, urlerror.URLError, TimeoutError) as exc:
                print(f"[warn] OpenAlex query failed for finding: {finding['finding']}", file=sys.stderr)
                print(f"[warn] {exc}", file=sys.stderr)
                works = []

        output_items.append(
            {
                "source": finding["source"],
                "finding": finding["finding"],
                "query": query,
                "works": works,
            }
        )

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    generated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    results = {
        "run_id": run_id,
        "generated_at_utc": generated_at,
        "sources": [str(path.relative_to(REPO_ROOT)) for path in sources],
        "dry_run": bool(args.dry_run),
        "items": output_items,
    }

    root = _resolve_local_storage_root(args.local_root)
    out_dir = root / "reports" / "literature"
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f"openalex_{run_id}.json"
    md_path = out_dir / f"openalex_{run_id}.md"
    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(_build_literature_markdown(results), encoding="utf-8")

    print(f"literature_findings={len(output_items)}")
    print(f"json_report={json_path}")
    print(f"markdown_report={md_path}")
    return 0


def cmd_local_init(args: argparse.Namespace) -> int:
    root = _resolve_local_storage_root(args.local_root)
    subdirs = [
        "raw",
        "derived",
        "reports",
        "models",
        "predictions",
        "manifests",
        "mappings",
        "enriched",
        "meta",
        "research",
        "research/runs",
    ]
    for rel in subdirs:
        (root / rel).mkdir(parents=True, exist_ok=True)
    print(f"initialized_local_storage={root}")
    return 0


def _dataset_metadata(path: Path) -> tuple[int, list[str], str]:
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    row_count = 0
    columns: list[str] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        columns = next(reader, [])
        for _ in reader:
            row_count += 1
    return row_count, columns, sha


def cmd_local_register_version(args: argparse.Namespace) -> int:
    root = _resolve_local_storage_root(args.local_root)
    dataset_path = Path(args.dataset_path).expanduser().resolve()
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}", file=sys.stderr)
        return 2

    row_count, columns, sha = _dataset_metadata(dataset_path)
    target_key = Path("raw") / args.version_id / "dataset.csv"
    target_path = (root / target_key).resolve()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(dataset_path, target_path)

    meta_path = root / "meta" / "versions.json"
    records: list[dict[str, Any]] = []
    if meta_path.exists():
        loaded = json.loads(meta_path.read_text(encoding="utf-8"))
        if isinstance(loaded, list):
            records = loaded
    records = [item for item in records if item.get("version_id") != args.version_id]
    records.append(
        {
            "version_id": args.version_id,
            "registered_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "source_dataset_path": str(dataset_path),
            "local_storage_key": target_key.as_posix(),
            "dataset_sha256": sha,
            "row_count": row_count,
            "columns": columns,
        }
    )
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(records, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "version_id": args.version_id,
                "local_storage_root": str(root),
                "dataset_key": target_key.as_posix(),
                "dataset_sha256": sha,
                "row_count": row_count,
                "column_count": len(columns),
            },
            indent=2,
        )
    )
    return 0


def cmd_local_list_versions(args: argparse.Namespace) -> int:
    root = _resolve_local_storage_root(args.local_root)
    meta_path = root / "meta" / "versions.json"
    if not meta_path.exists():
        print("[]")
        return 0
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    print(json.dumps(payload, indent=2))
    return 0


def cmd_research_init(args: argparse.Namespace) -> int:
    root = _resolve_local_storage_root(args.local_root)
    paths = _research_paths(root)
    paths["root"].mkdir(parents=True, exist_ok=True)
    paths["runs_dir"].mkdir(parents=True, exist_ok=True)
    (paths["runs_dir"] / ".gitkeep").touch()
    for key in ("papers", "claims", "claim_paper_links", "paper_edges", "dependencies", "ingestions"):
        _ensure_jsonl_file(paths[key])
    print(f"research_root={paths['root']}")
    print("initialized_files=papers,claims,claim_paper_links,paper_edges,dependencies,ingestions")
    return 0


def cmd_research_ingest_openalex(args: argparse.Namespace) -> int:
    _print_header("research ingest-openalex")
    local_root = _resolve_local_storage_root(args.local_root)
    paths = _research_paths(local_root)
    state = _load_research_state(local_root)

    report_paths = _resolve_openalex_report_paths(local_root=local_root, report_paths=args.report_path, latest=args.latest)
    if not report_paths:
        print("No OpenAlex report files found. Run `analysis literature` first.", file=sys.stderr)
        return 2

    missing = [path for path in report_paths if not path.exists()]
    if missing:
        print("Missing report path(s):", file=sys.stderr)
        for path in missing:
            print(f"  - {path}", file=sys.stderr)
        return 2

    total = {"papers": 0, "claims": 0, "links": 0, "edges": 0, "dependencies": 0, "placeholders": 0}
    for report_path in report_paths:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        run_id = str(payload.get("run_id") or _extract_run_id_from_report_path(report_path))
        created = _ingest_openalex_report_payload(
            state=state,
            payload=payload,
            report_path=str(report_path),
            run_id=run_id,
            link_relation=args.link_relation,
            max_dependencies_per_paper=args.max_dependencies_per_paper,
        )
        for key in total:
            total[key] += created[key]

        if not args.dry_run:
            paths["runs_dir"].mkdir(parents=True, exist_ok=True)
            snapshot_path = paths["runs_dir"] / report_path.name
            shutil.copy2(report_path, snapshot_path)

    if args.dry_run:
        print(f"dry_run_reports={len(report_paths)}")
    else:
        _save_research_state(local_root, state)
        if args.rebuild_sqlite:
            db_path = _rebuild_research_sqlite(local_root, state)
            print(f"sqlite_db={db_path}")

    print(f"reports_processed={len(report_paths)}")
    print(f"new_papers={total['papers']}")
    print(f"new_claims={total['claims']}")
    print(f"new_links={total['links']}")
    print(f"new_edges={total['edges']}")
    print(f"new_dependencies={total['dependencies']}")
    print(f"new_placeholder_papers={total['placeholders']}")
    return 0


def cmd_research_add_edge(args: argparse.Namespace) -> int:
    _print_header("research add-edge")
    local_root = _resolve_local_storage_root(args.local_root)
    state = _load_research_state(local_root)
    papers_by_id = {item.get("paper_id"): item for item in state["papers"]}

    now = _utcnow_iso()
    for paper_id in (args.from_paper_id, args.to_paper_id):
        if paper_id in papers_by_id:
            continue
        if not args.allow_placeholder:
            print(f"Unknown paper_id: {paper_id} (use --allow-placeholder to create it)", file=sys.stderr)
            return 2
        _ensure_placeholder_paper(
            papers_by_id=papers_by_id,
            openalex_id=f"https://openalex.org/{paper_id}",
            now=now,
        )

    edges_by_key = {
        (item["from_paper_id"], item["to_paper_id"], item["relation"], item.get("claim_id") or ""): item
        for item in state["paper_edges"]
        if item.get("from_paper_id") and item.get("to_paper_id") and item.get("relation")
    }

    created = _upsert_paper_edge(
        edges_by_key=edges_by_key,
        from_paper_id=args.from_paper_id,
        to_paper_id=args.to_paper_id,
        relation=args.relation,
        claim_id=args.claim_id or "",
        provenance="manual",
        run_id="",
        note=args.note or "",
        now=now,
    )

    state["papers"] = sorted(papers_by_id.values(), key=lambda item: item["paper_id"])
    state["paper_edges"] = sorted(
        edges_by_key.values(),
        key=lambda item: (item["from_paper_id"], item["to_paper_id"], item["relation"], item.get("claim_id") or ""),
    )

    _save_research_state(local_root, state)
    if args.rebuild_sqlite:
        db_path = _rebuild_research_sqlite(local_root, state)
        print(f"sqlite_db={db_path}")
    print(f"edge_created={1 if created else 0}")
    return 0


def cmd_research_add_dependency(args: argparse.Namespace) -> int:
    _print_header("research add-dependency")
    local_root = _resolve_local_storage_root(args.local_root)
    state = _load_research_state(local_root)
    papers_by_id = {item.get("paper_id"): item for item in state["papers"]}

    now = _utcnow_iso()
    for paper_id in (args.paper_id, args.depends_on_paper_id):
        if paper_id in papers_by_id:
            continue
        if not args.allow_placeholder:
            print(f"Unknown paper_id: {paper_id} (use --allow-placeholder to create it)", file=sys.stderr)
            return 2
        _ensure_placeholder_paper(
            papers_by_id=papers_by_id,
            openalex_id=f"https://openalex.org/{paper_id}",
            now=now,
        )

    dependencies_by_key = {
        (item["paper_id"], item["depends_on_paper_id"], item["reason"]): item
        for item in state["dependencies"]
        if item.get("paper_id") and item.get("depends_on_paper_id") and item.get("reason")
    }
    edges_by_key = {
        (item["from_paper_id"], item["to_paper_id"], item["relation"], item.get("claim_id") or ""): item
        for item in state["paper_edges"]
        if item.get("from_paper_id") and item.get("to_paper_id") and item.get("relation")
    }

    dep_created = _upsert_dependency(
        dependencies_by_key=dependencies_by_key,
        paper_id=args.paper_id,
        depends_on_paper_id=args.depends_on_paper_id,
        reason=args.reason,
        depth=args.depth,
        provenance="manual",
        run_id="",
        now=now,
    )
    edge_created = _upsert_paper_edge(
        edges_by_key=edges_by_key,
        from_paper_id=args.paper_id,
        to_paper_id=args.depends_on_paper_id,
        relation="depends_on",
        claim_id="",
        provenance="manual",
        run_id="",
        note=args.reason,
        now=now,
    )

    state["papers"] = sorted(papers_by_id.values(), key=lambda item: item["paper_id"])
    state["dependencies"] = sorted(
        dependencies_by_key.values(),
        key=lambda item: (item["paper_id"], item["depends_on_paper_id"], item["reason"]),
    )
    state["paper_edges"] = sorted(
        edges_by_key.values(),
        key=lambda item: (item["from_paper_id"], item["to_paper_id"], item["relation"], item.get("claim_id") or ""),
    )

    _save_research_state(local_root, state)
    if args.rebuild_sqlite:
        db_path = _rebuild_research_sqlite(local_root, state)
        print(f"sqlite_db={db_path}")
    print(f"dependency_created={1 if dep_created else 0}")
    print(f"edge_created={1 if edge_created else 0}")
    return 0


def cmd_research_rebuild_sqlite(args: argparse.Namespace) -> int:
    _print_header("research rebuild-sqlite")
    local_root = _resolve_local_storage_root(args.local_root)
    state = _load_research_state(local_root)
    db_path = _rebuild_research_sqlite(local_root, state)
    print(f"sqlite_db={db_path}")
    return 0


def cmd_research_overview(args: argparse.Namespace) -> int:
    _print_header("research overview")
    local_root = _resolve_local_storage_root(args.local_root)
    paths = _research_paths(local_root)
    state = _load_research_state(local_root)
    generated_at = _utcnow_iso()
    markdown = _build_research_overview_markdown(state, generated_at=generated_at)
    paths["overview_md"].write_text(markdown, encoding="utf-8")
    print(f"overview_path={paths['overview_md']}")
    print(f"papers={len(state['papers'])}")
    print(f"claims={len(state['claims'])}")
    print(f"claim_paper_links={len(state['claim_paper_links'])}")
    print(f"paper_edges={len(state['paper_edges'])}")
    print(f"dependencies={len(state['dependencies'])}")
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

    p_analysis_lit = analysis_sub.add_parser(
        "literature",
        help="Find related literature in OpenAlex from analysis findings",
    )
    p_analysis_lit.add_argument(
        "--finding-source",
        action="append",
        help="Markdown source path for finding extraction (repeatable). Defaults to NORTHSTAR + analysis/output/*.md",
    )
    p_analysis_lit.add_argument("--max-findings", type=int, default=8, help="Maximum findings to map")
    p_analysis_lit.add_argument("--works-per-finding", type=int, default=5, help="OpenAlex works per finding")
    p_analysis_lit.add_argument("--from-year", type=int, default=2010, help="Minimum publication year")
    p_analysis_lit.add_argument("--mailto", default=os.getenv("OPENALEX_EMAIL"), help="Optional email for OpenAlex polite pool")
    p_analysis_lit.add_argument("--run-id", help="Optional output run identifier")
    p_analysis_lit.add_argument("--local-root")
    p_analysis_lit.add_argument("--dry-run", action="store_true")
    p_analysis_lit.set_defaults(func=cmd_analysis_literature)

    p_local = sub.add_parser("local", help="Local storage setup and version registration")
    local_sub = p_local.add_subparsers(dest="local_cmd", required=True)

    p_local_init = local_sub.add_parser("init", help="Initialize local storage directories")
    p_local_init.add_argument("--local-root")
    p_local_init.set_defaults(func=cmd_local_init)

    p_local_register = local_sub.add_parser("register-version", help="Register a dataset version in local storage")
    p_local_register.add_argument("--version-id", required=True)
    p_local_register.add_argument("--dataset-path", default=str(REPO_ROOT / "dataset.csv"))
    p_local_register.add_argument("--local-root")
    p_local_register.set_defaults(func=cmd_local_register_version)

    p_local_versions = local_sub.add_parser("list-versions", help="List local registered versions")
    p_local_versions.add_argument("--local-root")
    p_local_versions.set_defaults(func=cmd_local_list_versions)

    p_research = sub.add_parser("research", help="Research foundation database workflows")
    research_sub = p_research.add_subparsers(dest="research_cmd", required=True)

    p_research_init = research_sub.add_parser("init", help="Initialize research database files")
    p_research_init.add_argument("--local-root")
    p_research_init.set_defaults(func=cmd_research_init)

    p_research_ingest = research_sub.add_parser(
        "ingest-openalex",
        help="Ingest OpenAlex literature reports into the research database",
    )
    p_research_ingest.add_argument(
        "--report-path",
        action="append",
        help="Path to OpenAlex JSON report (repeatable). If omitted, ingest all local reports or latest with --latest.",
    )
    p_research_ingest.add_argument("--latest", action="store_true", help="Ingest only the latest OpenAlex report")
    p_research_ingest.add_argument(
        "--link-relation",
        default="supports",
        choices=["supports", "context", "method"],
        help="Relation used for claim-paper links",
    )
    p_research_ingest.add_argument("--max-dependencies-per-paper", type=int, default=8)
    p_research_ingest.add_argument("--local-root")
    p_research_ingest.add_argument(
        "--skip-rebuild-sqlite",
        dest="rebuild_sqlite",
        action="store_false",
        help="Skip SQLite refresh after ingestion",
    )
    p_research_ingest.set_defaults(rebuild_sqlite=True)
    p_research_ingest.add_argument("--dry-run", action="store_true")
    p_research_ingest.set_defaults(func=cmd_research_ingest_openalex)

    p_research_edge = research_sub.add_parser("add-edge", help="Add or update a manual paper interaction edge")
    p_research_edge.add_argument("--from-paper-id", required=True)
    p_research_edge.add_argument("--to-paper-id", required=True)
    p_research_edge.add_argument(
        "--relation",
        required=True,
        choices=["supports", "contradicts", "extends", "uses_method_of", "depends_on", "related"],
    )
    p_research_edge.add_argument("--claim-id")
    p_research_edge.add_argument("--note")
    p_research_edge.add_argument("--allow-placeholder", action="store_true")
    p_research_edge.add_argument("--local-root")
    p_research_edge.add_argument("--rebuild-sqlite", action="store_true")
    p_research_edge.set_defaults(func=cmd_research_add_edge)

    p_research_dependency = research_sub.add_parser(
        "add-dependency",
        help="Add or update a manual reading dependency between papers",
    )
    p_research_dependency.add_argument("--paper-id", required=True)
    p_research_dependency.add_argument("--depends-on-paper-id", required=True)
    p_research_dependency.add_argument("--reason", required=True)
    p_research_dependency.add_argument("--depth", type=int, default=1)
    p_research_dependency.add_argument("--allow-placeholder", action="store_true")
    p_research_dependency.add_argument("--local-root")
    p_research_dependency.add_argument("--rebuild-sqlite", action="store_true")
    p_research_dependency.set_defaults(func=cmd_research_add_dependency)

    p_research_sqlite = research_sub.add_parser("rebuild-sqlite", help="Rebuild derived SQLite database from JSONL")
    p_research_sqlite.add_argument("--local-root")
    p_research_sqlite.set_defaults(func=cmd_research_rebuild_sqlite)

    p_research_overview = research_sub.add_parser("overview", help="Generate thesis foundation overview markdown")
    p_research_overview.add_argument("--local-root")
    p_research_overview.set_defaults(func=cmd_research_overview)

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
    p_mapping_freeze.add_argument("--auto-approve-all", action="store_true")
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
