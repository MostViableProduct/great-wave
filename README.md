# Contextual Compiler

A cascade classification engine for product signals. Routes signals (metrics, errors, deploys, audits) into categories using a multi-stage pipeline: heuristic keywords, Bayesian gating, optional LLM classification, and self-improving keyword learning.

```
Signal ──► Heuristic Classifier ──► Bayesian Gate ──► LLM Classifier
                │                        │                  │
                │                        │                  ▼
                │                        │           Keyword Learning
                │                        │                  │
                ▼                        ▼                  ▼
           Category + Score ◄── Gate Skip (fast) ◄── LLM Category
                │
                ▼
         Vector Search (entity resolution)
                │
                ▼
         ClassifyResult
```

## Quick Start

### Binary

```bash
go install github.com/MostViableProduct/contextual-compiler/cmd/compiler@latest
compiler
# Listening on :8200
```

### Docker

```bash
docker build -t contextual-compiler .
docker run -p 8200:8200 contextual-compiler
```

### Docker Compose

```bash
docker-compose up -d
# compiler on :8200, compiler-with-postgres on :8201
```

### Classify a signal

```bash
curl -s http://localhost:8200/v1/classify \
  -H 'Content-Type: application/json' \
  -d '{"type":"metric","content":"High p99 latency detected in API gateway","source":"prometheus"}' | jq
```

```json
{
  "category": "performance",
  "relevance_score": 0.6,
  "classification_source": "heuristic",
  "confidence": 0,
  "signal_class": "SEMANTIC"
}
```

## Configuration

Configuration is loaded from `CONFIG_PATH` (default: `config.yaml`). Supports JSON and YAML.

```yaml
classifier:
  categories:
    - name: performance
      keywords: [latency, throughput, p99, slow, timeout]
      weights: {p99: 2.0, timeout: 2.0}
    - name: reliability
      keywords: [error, failure, crash, outage]
      weights: {outage: 2.0}
  source_priors:
    sentry: {reliability: 3.0}
    prometheus: {performance: 3.0}
  type_to_category:
    metric: performance
    error: reliability

gate:
  agreement_threshold: 0.75
  uncertainty_max: 0.10
  shadow_mode: false

health:
  severities:
    - name: critical
      weight: 0.40
      direction: negative
      default_alpha: 5.0
      default_beta: 1.0

keywords:
  min_confidence: 0.7
  min_observations: 10
```

### Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `CONFIG_PATH` | Path to YAML/JSON config | `config.yaml` |
| `PORT` | HTTP server port | `8200` |
| `DATABASE_URL` | PostgreSQL DSN | (in-memory) |
| `SQLITE_PATH` | SQLite file path | (in-memory) |
| `ANTHROPIC_API_KEY` | Anthropic API key | (heuristic only) |
| `OPENAI_API_KEY` | OpenAI API key (LLM + embeddings) | (heuristic only) |
| `GEMINI_API_KEY` | Google Gemini API key | (heuristic only) |
| `LOG_EVENTS` | Enable stdout event logging | (disabled) |

## API Reference

### `GET /health`

Liveness probe with dependency status.

### `GET /metrics`

Prometheus-format counters (requires `WithMetrics` in handler setup).

### `POST /v1/classify`

Classify signal content directly.

```json
{"type": "metric", "content": "p99 latency spike", "source": "prometheus", "tenant_id": "t1"}
```

### `POST /v1/classify/signal`

Classify by flattening a JSON payload into searchable text.

```json
{"type": "error", "payload": {"message": "connection refused"}, "source": "sentry"}
```

### `GET /v1/health/{tenant_id}/{entity_id}`

Bayesian health score for an entity.

### `POST /v1/health/{tenant_id}/{entity_id}/events`

Record a health-affecting event.

```json
{"severity": "critical", "category": "reliability", "confidence": 0.9}
```

### `POST /v1/keywords/promote`

Promote high-confidence learned keywords to the heuristic classifier.

### `POST /v1/state/flush`

Persist state (gate entries, health priors) to storage adapters.

## Adapters

| Category | Adapter | Package |
|----------|---------|---------|
| LLM | Anthropic (Claude) | `adapters/llm/anthropic` |
| LLM | OpenAI (GPT) | `adapters/llm/openai` |
| LLM | Google Gemini | `adapters/llm/gemini` |
| Embeddings | OpenAI | `adapters/embeddings/openai` |
| Vector | In-Memory (cosine) | `adapters/vector/memory` |
| Storage | PostgreSQL | `adapters/storage/postgres` |
| Storage | SQLite | `adapters/storage/sqlite` |
| Events | Log Writer (stdout) | `adapters/events/logwriter` |

All adapters are optional. The compiler gracefully degrades to in-memory, heuristic-only mode when adapters are nil.

## Library Usage

```go
import (
    "github.com/MostViableProduct/contextual-compiler/pkg/compiler"
    "github.com/MostViableProduct/contextual-compiler/pkg/classifier"
)

cfg := compiler.DefaultConfig()
cfg.Classifier = classifier.Config{
    Categories: []classifier.CategoryConfig{
        {Name: "performance", Keywords: []string{"latency", "p99", "slow"}},
        {Name: "reliability", Keywords: []string{"error", "crash", "outage"}},
    },
}

c := compiler.New(cfg, compiler.Deps{}, compiler.Callbacks{})

result, err := c.Classify(ctx, compiler.Signal{
    Source:  "prometheus",
    Type:    "metric",
    Content: "High p99 latency detected",
})
```

## Deployment

### Docker

```bash
docker build -t contextual-compiler .
docker run -p 8200:8200 \
  -e ANTHROPIC_API_KEY=sk-... \
  -e DATABASE_URL=postgres://... \
  contextual-compiler
```

### Docker Compose

```bash
docker-compose up -d
```

Services:
- `compiler` — in-memory mode on port 8200
- `postgres` — PostgreSQL 17 on port 5432
- `compiler-with-postgres` — persistent mode on port 8201

## Testing & Benchmarks

```bash
# All tests with race detector
make test

# Short (unit) tests only
make test-unit

# Benchmarks
make bench

# Lint
make lint
```

## Architecture

### Core Packages

| Package | Purpose |
|---------|---------|
| `pkg/compiler` | Top-level orchestrator — ties all stages together |
| `pkg/classifier` | Heuristic cascade classifier with keyword matching |
| `pkg/gate` | Bayesian Beta-Binomial gate for LLM skip decisions |
| `pkg/health` | Per-entity Bayesian health model |
| `pkg/keywords` | Self-improving keyword extraction and promotion |
| `pkg/belief` | Dempster-Shafer evidence theory for confidence fusion |

### Classification Pipeline

1. **Heuristic** — keyword matching with configurable weights and source priors
2. **Bayesian Gate** — learns P(heuristic agrees with LLM) per (tenant, category, source); skips LLM when confident
3. **LLM** — deep classification when heuristics are unreliable
4. **Keyword Learning** — extracts novel keywords from heuristic/LLM disagreements
5. **Vector Search** — optional entity resolution via cosine similarity
