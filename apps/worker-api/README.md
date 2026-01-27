# Worker API

Cloudflare Worker control plane for versions, runs, and mapping bundles.

## Endpoints (MVP)

- `POST /versions/register`
- `POST /runs/trigger`
- `GET /runs`
- `GET /runs/{run_id}`
- `GET /versions/{version_id}`

## Auth

Send `Authorization: Bearer <token>` for all endpoints except `/health`.
Set the token in `wrangler.toml` as `AUTH_TOKEN`.

