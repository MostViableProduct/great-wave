-- Contextual Compiler: Postgres schema
-- Bayesian gate entries
CREATE TABLE IF NOT EXISTS compiler_gates (
    tenant_id    TEXT NOT NULL DEFAULT '',
    category     TEXT NOT NULL,
    source_type  TEXT NOT NULL,
    alpha        DOUBLE PRECISION NOT NULL DEFAULT 2.0,
    beta         DOUBLE PRECISION NOT NULL DEFAULT 2.0,
    observations INTEGER NOT NULL DEFAULT 0,
    updated_at   TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (tenant_id, category, source_type)
);
CREATE INDEX IF NOT EXISTS idx_compiler_gates_updated
    ON compiler_gates (updated_at DESC);

-- Entity health priors (JSONB for arbitrary severity levels)
CREATE TABLE IF NOT EXISTS compiler_health_priors (
    tenant_id    TEXT NOT NULL DEFAULT '',
    entity_id    TEXT NOT NULL,
    priors       JSONB NOT NULL DEFAULT '{}',
    updated_at   TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (tenant_id, entity_id)
);

-- Learned keywords
CREATE TABLE IF NOT EXISTS compiler_learned_keywords (
    keyword               TEXT NOT NULL,
    category              TEXT NOT NULL,
    weight                DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    confidence            DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    total_observations    INTEGER NOT NULL DEFAULT 0,
    positive_observations INTEGER NOT NULL DEFAULT 0,
    promoted              BOOLEAN NOT NULL DEFAULT false,
    promoted_at           TIMESTAMPTZ,
    created_at            TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at            TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (keyword, category)
);
CREATE INDEX IF NOT EXISTS idx_compiler_keywords_promoted
    ON compiler_learned_keywords (promoted) WHERE promoted = true;
