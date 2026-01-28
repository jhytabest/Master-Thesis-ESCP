interface Env {
  DB: D1Database;
  R2: R2Bucket;
  AUTH_TOKEN: string;
}

type JsonRecord = Record<string, unknown>;

type RunRow = {
  run_id: string;
  version_id: string;
  mapping_bundle_id: string | null;
  status: string;
  started_at: string;
  finished_at: string | null;
  artifact_index_path: string | null;
  error_step: string | null;
  error_message: string | null;
};

type VersionRow = {
  version_id: string;
  created_at: string;
  source_path: string;
  dataset_sha256: string;
  row_count: number;
  schema_json: string;
};

type MappingBundleRow = {
  mapping_bundle_id: string;
  created_at: string;
  source_version_id: string | null;
  columns_json: string | null;
  artifact_path: string;
  manifest_sha256: string;
};

function jsonResponse(body: JsonRecord | JsonRecord[], status = 200): Response {
  return new Response(JSON.stringify(body, null, 2), {
    status,
    headers: { "content-type": "application/json" }
  });
}

function unauthorized(): Response {
  return jsonResponse({ error: "unauthorized" }, 401);
}

function badRequest(message: string): Response {
  return jsonResponse({ error: message }, 400);
}

function requireAuth(request: Request, env: Env): boolean {
  if (!env.AUTH_TOKEN) {
    return true;
  }
  const authHeader = request.headers.get("authorization");
  if (!authHeader) {
    return false;
  }
  const [scheme, token] = authHeader.split(" ");
  return scheme === "Bearer" && token === env.AUTH_TOKEN;
}

function parsePath(pathname: string): string[] {
  return pathname.replace(/\/+$/, "").split("/").filter(Boolean);
}

function bytesToHex(bytes: ArrayBuffer): string {
  const view = new Uint8Array(bytes);
  let hex = "";
  for (const byte of view) {
    hex += byte.toString(16).padStart(2, "0");
  }
  return hex;
}

async function sha256Hex(buffer: ArrayBuffer): Promise<string> {
  const hash = await crypto.subtle.digest("SHA-256", buffer);
  return bytesToHex(hash);
}

function parseCsvHeader(headerLine: string): string[] {
  const result: string[] = [];
  let current = "";
  let inQuotes = false;
  for (let i = 0; i < headerLine.length; i += 1) {
    const char = headerLine[i];
    if (char === "\"") {
      if (inQuotes && headerLine[i + 1] === "\"") {
        current += "\"";
        i += 1;
        continue;
      }
      inQuotes = !inQuotes;
      continue;
    }
    if (char === "," && !inQuotes) {
      result.push(current.trim());
      current = "";
      continue;
    }
    current += char;
  }
  if (current.length > 0) {
    result.push(current.trim());
  }
  return result.filter(Boolean);
}

async function extractCsvMetadata(buffer: ArrayBuffer): Promise<{ rowCount: number; schema: JsonRecord }> {
  const text = new TextDecoder("utf-8").decode(buffer);
  const lines = text.split(/\r?\n/);
  if (lines.length === 0) {
    return { rowCount: 0, schema: { columns: [], column_count: 0 } };
  }
  if (lines[lines.length - 1]?.trim() === "") {
    lines.pop();
  }
  const headerLine = lines[0] ?? "";
  const columns = parseCsvHeader(headerLine);
  const rowCount = Math.max(0, lines.length - 1);
  return { rowCount, schema: { columns, column_count: columns.length } };
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url);
    const segments = parsePath(url.pathname);

    if (url.pathname === "/health") {
      return jsonResponse({ ok: true });
    }

    if (!requireAuth(request, env)) {
      return unauthorized();
    }

    if (segments.length === 2 && segments[0] === "versions" && segments[1] === "register") {
      if (request.method !== "POST") {
        return badRequest("method_not_allowed");
      }
      const contentType = request.headers.get("content-type") ?? "";
      const createdAt = new Date().toISOString();
      if (contentType.includes("application/json")) {
        const body = (await request.json().catch(() => null)) as JsonRecord | null;
        if (!body) {
          return badRequest("invalid_json");
        }
        const sourcePath = body.source_path as string | undefined;
        const datasetSha = body.dataset_sha256 as string | undefined;
        const rowCount = body.row_count as number | undefined;
        const schemaJson = body.schema_json as JsonRecord | undefined;
        if (!sourcePath || !datasetSha || typeof rowCount !== "number" || !schemaJson) {
          return badRequest("missing_required_fields");
        }
        const versionId = (body.version_id as string | undefined) ?? crypto.randomUUID();

        await env.DB.prepare(
          "INSERT INTO versions (version_id, created_at, source_path, dataset_sha256, row_count, schema_json) VALUES (?, ?, ?, ?, ?, ?)"
        )
          .bind(versionId, createdAt, sourcePath, datasetSha, rowCount, JSON.stringify(schemaJson))
          .run();

        return jsonResponse({ version_id: versionId, created_at: createdAt });
      }

      let buffer: ArrayBuffer | null = null;
      let versionId: string | null = null;
      if (contentType.includes("multipart/form-data")) {
        const form = await request.formData();
        const file = form.get("file");
        if (file instanceof File) {
          buffer = await file.arrayBuffer();
        }
        const suppliedVersion = form.get("version_id");
        if (typeof suppliedVersion === "string" && suppliedVersion.trim()) {
          versionId = suppliedVersion.trim();
        }
      } else {
        buffer = await request.arrayBuffer();
        const suppliedVersion = request.headers.get("x-version-id");
        if (suppliedVersion && suppliedVersion.trim()) {
          versionId = suppliedVersion.trim();
        }
      }

      if (!buffer) {
        return badRequest("missing_file");
      }
      const finalVersionId = versionId ?? crypto.randomUUID();
      const datasetSha = await sha256Hex(buffer);
      const { rowCount, schema } = await extractCsvMetadata(buffer);
      const sourcePath = `raw/${finalVersionId}/dataset.csv`;

      await env.R2.put(sourcePath, buffer, {
        httpMetadata: { contentType: "text/csv" },
        customMetadata: {
          dataset_sha256: datasetSha,
          row_count: rowCount.toString()
        }
      });

      await env.DB.prepare(
        "INSERT INTO versions (version_id, created_at, source_path, dataset_sha256, row_count, schema_json) VALUES (?, ?, ?, ?, ?, ?)"
      )
        .bind(finalVersionId, createdAt, sourcePath, datasetSha, rowCount, JSON.stringify(schema))
        .run();

      return jsonResponse({
        version_id: finalVersionId,
        created_at: createdAt,
        source_path: sourcePath,
        dataset_sha256: datasetSha,
        row_count: rowCount,
        schema_json: schema
      });
    }

    if (segments.length === 2 && segments[0] === "runs" && segments[1] === "trigger") {
      if (request.method !== "POST") {
        return badRequest("method_not_allowed");
      }
      const body = (await request.json().catch(() => null)) as JsonRecord | null;
      if (!body) {
        return badRequest("invalid_json");
      }
      const versionId = body.version_id as string | undefined;
      if (!versionId) {
        return badRequest("missing_version_id");
      }
      const mappingBundleId = (body.mapping_bundle_id as string | undefined) ?? null;
      const runId = (body.run_id as string | undefined) ?? crypto.randomUUID();
      const startedAt = new Date().toISOString();

      await env.DB.prepare(
        "INSERT INTO runs (run_id, version_id, mapping_bundle_id, status, started_at) VALUES (?, ?, ?, ?, ?)"
      )
        .bind(runId, versionId, mappingBundleId, "STARTED", startedAt)
        .run();

      return jsonResponse({ run_id: runId, status: "STARTED" });
    }

    if (segments.length === 2 && segments[0] === "runs" && (request.method === "PATCH" || request.method === "POST")) {
      const runId = segments[1];
      const body = (await request.json().catch(() => null)) as JsonRecord | null;
      if (!body) {
        return badRequest("invalid_json");
      }
      const updates: string[] = [];
      const values: unknown[] = [];
      const fields: Array<[string, string]> = [
        ["status", "status"],
        ["started_at", "started_at"],
        ["finished_at", "finished_at"],
        ["artifact_index_path", "artifact_index_path"],
        ["error_step", "error_step"],
        ["error_message", "error_message"]
      ];
      for (const [key, column] of fields) {
        if (Object.prototype.hasOwnProperty.call(body, key)) {
          updates.push(`${column} = ?`);
          values.push(body[key]);
        }
      }
      if (updates.length === 0) {
        return badRequest("no_updates");
      }
      values.push(runId);
      await env.DB.prepare(`UPDATE runs SET ${updates.join(", ")} WHERE run_id = ?`).bind(...values).run();
      return jsonResponse({ run_id: runId, updated: updates.map((entry) => entry.split(" = ")[0]) });
    }

    if (segments.length === 1 && segments[0] === "runs" && request.method === "GET") {
      const result = await env.DB.prepare(
        "SELECT * FROM runs ORDER BY started_at DESC LIMIT 100"
      ).all<RunRow>();
      return jsonResponse({ runs: result.results ?? [] });
    }

    if (segments.length === 1 && segments[0] === "versions" && request.method === "GET") {
      const result = await env.DB.prepare(
        "SELECT * FROM versions ORDER BY created_at DESC LIMIT 100"
      ).all<VersionRow>();
      return jsonResponse({ versions: result.results ?? [] });
    }

    if (segments.length === 2 && segments[0] === "mapping_bundles" && segments[1] === "register") {
      if (request.method !== "POST") {
        return badRequest("method_not_allowed");
      }
      const body = (await request.json().catch(() => null)) as JsonRecord | null;
      if (!body) {
        return badRequest("invalid_json");
      }
      const mappingBundleId = body.mapping_bundle_id as string | undefined;
      const createdAt = (body.created_at as string | undefined) ?? new Date().toISOString();
      const sourceVersionId = (body.source_version_id as string | undefined) ?? null;
      const columnsJson = (body.columns_json as string | undefined) ?? null;
      const artifactPath = body.artifact_path as string | undefined;
      const manifestSha = body.manifest_sha256 as string | undefined;
      if (!mappingBundleId || !artifactPath || !manifestSha) {
        return badRequest("missing_required_fields");
      }
      await env.DB.prepare(
        "INSERT OR REPLACE INTO mapping_bundles (mapping_bundle_id, created_at, source_version_id, columns_json, artifact_path, manifest_sha256) VALUES (?, ?, ?, ?, ?, ?)"
      )
        .bind(mappingBundleId, createdAt, sourceVersionId, columnsJson, artifactPath, manifestSha)
        .run();
      return jsonResponse({ mapping_bundle_id: mappingBundleId, created_at: createdAt });
    }

    if (segments.length === 1 && segments[0] === "mapping_bundles" && request.method === "GET") {
      const result = await env.DB.prepare(
        "SELECT * FROM mapping_bundles ORDER BY created_at DESC LIMIT 100"
      ).all<MappingBundleRow>();
      return jsonResponse({ mapping_bundles: result.results ?? [] });
    }

    if (segments.length === 2 && segments[0] === "mapping_bundles" && request.method === "GET") {
      const bundleId = segments[1];
      const result = await env.DB.prepare("SELECT * FROM mapping_bundles WHERE mapping_bundle_id = ?")
        .bind(bundleId)
        .all<MappingBundleRow>();
      if (!result.results || result.results.length === 0) {
        return jsonResponse({ error: "not_found" }, 404);
      }
      return jsonResponse({ mapping_bundle: result.results[0] });
    }

    if (segments.length === 2 && segments[0] === "runs" && request.method === "GET") {
      const runId = segments[1];
      const result = await env.DB.prepare("SELECT * FROM runs WHERE run_id = ?")
        .bind(runId)
        .all<RunRow>();
      if (!result.results || result.results.length === 0) {
        return jsonResponse({ error: "not_found" }, 404);
      }
      return jsonResponse({ run: result.results[0] });
    }

    if (segments.length === 2 && segments[0] === "versions" && request.method === "GET") {
      const versionId = segments[1];
      const result = await env.DB.prepare("SELECT * FROM versions WHERE version_id = ?")
        .bind(versionId)
        .all<VersionRow>();
      if (!result.results || result.results.length === 0) {
        return jsonResponse({ error: "not_found" }, 404);
      }
      return jsonResponse({ version: result.results[0] });
    }

    return jsonResponse({ error: "not_found" }, 404);
  }
};
