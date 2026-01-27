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
      const createdAt = new Date().toISOString();

      await env.DB.prepare(
        "INSERT INTO versions (version_id, created_at, source_path, dataset_sha256, row_count, schema_json) VALUES (?, ?, ?, ?, ?, ?)"
      )
        .bind(versionId, createdAt, sourcePath, datasetSha, rowCount, JSON.stringify(schemaJson))
        .run();

      return jsonResponse({ version_id: versionId, created_at: createdAt });
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
      const runId = crypto.randomUUID();
      const startedAt = new Date().toISOString();

      await env.DB.prepare(
        "INSERT INTO runs (run_id, version_id, mapping_bundle_id, status, started_at) VALUES (?, ?, ?, ?, ?)"
      )
        .bind(runId, versionId, mappingBundleId, "STARTED", startedAt)
        .run();

      return jsonResponse({ run_id: runId, status: "STARTED" });
    }

    if (segments.length === 1 && segments[0] === "runs" && request.method === "GET") {
      const result = await env.DB.prepare(
        "SELECT * FROM runs ORDER BY started_at DESC LIMIT 100"
      ).all<RunRow>();
      return jsonResponse({ runs: result.results ?? [] });
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
