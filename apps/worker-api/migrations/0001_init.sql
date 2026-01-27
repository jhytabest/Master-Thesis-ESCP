CREATE TABLE IF NOT EXISTS versions (
  version_id TEXT PRIMARY KEY,
  created_at TEXT NOT NULL,
  source_path TEXT NOT NULL,
  dataset_sha256 TEXT NOT NULL,
  row_count INTEGER NOT NULL,
  schema_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
  run_id TEXT PRIMARY KEY,
  version_id TEXT NOT NULL,
  mapping_bundle_id TEXT,
  status TEXT NOT NULL,
  started_at TEXT NOT NULL,
  finished_at TEXT,
  artifact_index_path TEXT,
  error_step TEXT,
  error_message TEXT
);

CREATE INDEX IF NOT EXISTS runs_version_id_idx ON runs(version_id);
CREATE INDEX IF NOT EXISTS runs_status_idx ON runs(status);

CREATE TABLE IF NOT EXISTS mapping_bundles (
  mapping_bundle_id TEXT PRIMARY KEY,
  created_at TEXT NOT NULL,
  source_version_id TEXT,
  columns_json TEXT,
  artifact_path TEXT NOT NULL,
  manifest_sha256 TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS models (
  model_id TEXT PRIMARY KEY,
  version_id TEXT NOT NULL,
  mapping_bundle_id TEXT,
  trained_at TEXT NOT NULL,
  target_name TEXT NOT NULL,
  metrics_path TEXT NOT NULL,
  artifact_path TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS enrichment_runs (
  enrichment_run_id TEXT PRIMARY KEY,
  version_id TEXT NOT NULL,
  source TEXT NOT NULL,
  status TEXT NOT NULL,
  artifact_path TEXT NOT NULL,
  coverage_path TEXT NOT NULL,
  created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS enrichment_version_id_idx ON enrichment_runs(version_id);
