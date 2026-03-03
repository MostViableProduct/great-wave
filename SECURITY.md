# Security Considerations

The contextual-compiler is a **library and reference server**. Certain security responsibilities are deliberately left to the caller. This document describes those boundaries and known trade-offs.

## Caller Responsibilities

### D1: Authentication and Authorization

The reference HTTP server (`api/` package) does **not** implement authentication or authorization. All state-mutating endpoints are unprotected by default:

- `POST /v1/classify`
- `POST /v1/classify/signal`
- `POST /v1/health/{tenant_id}/{entity_id}/events`
- `POST /v1/keywords/promote`
- `POST /v1/state/flush`

**Action:** Callers must add auth middleware before registering routes.

### D2: TLS Termination

The reference server listens on plain HTTP. In production, terminate TLS at a reverse proxy or load balancer in front of the server.

### D3: Rate Limiting

Classification endpoints that invoke LLM adapters (`/v1/classify`, `/v1/classify/signal`) can be expensive. Callers should apply per-IP or per-tenant rate limiting at the gateway layer.

### D4: Keyword Learning Scope

The self-improving keyword loop is **global** — learned keywords are not scoped to a specific tenant. In multi-tenant deployments, a keyword learned from tenant A's signal will influence classification for tenant B.

### D5: Vector Store Scope

Vector search for entity resolution is **global** — all tenants share the same embedding space. Callers requiring tenant isolation should use separate vector store instances.

### D6: Gemini API Key in URL

The Gemini adapter transmits the API key as a URL query parameter (`?key=...`) because Google's API requires it. This means the key may appear in:

- Reverse proxy access logs
- CDN logs
- Network monitoring tools

Redirect following is disabled on all HTTP clients to prevent key leakage via redirects. Callers should ensure logging infrastructure does not persist full query strings for the Gemini endpoint.

## Reporting Vulnerabilities

If you discover a security vulnerability, please report it responsibly by opening a GitHub security advisory on this repository.
