# Worker API

Cloudflare Worker control plane for versions, runs, and mapping bundles.

## Endpoints (MVP)

- `POST /versions/register` (multipart `file` or JSON metadata)
- `POST /runs/trigger`
- `PATCH /runs/{run_id}` (update status/artifact paths)
- `GET /runs`
- `GET /runs/{run_id}`
- `GET /versions/{version_id}`

## Auth

Send `Authorization: Bearer <token>` for all endpoints except `/health`.
Set the token with `wrangler secret put AUTH_TOKEN`.

## Uploading a CSV

```
curl -X POST \\
  -H "Authorization: Bearer <token>" \\
  -F "file=@/path/to/dataset.csv" \\
  https://<worker-url>/versions/register
```
