-- Contextual Compiler: SQLite schema
-- Bayesian gate entries
CREATE TABLE IF NOT EXISTS compiler_gates (
    tenant_id    TEXT NOT NULL DEFAULT '',
    category     TEXT NOT NULL,
    source_type  TEXT NOT NULL,
    alpha        REAL NOT NULL DEFAULT 2.0,
    beta         REAL NOT NULL DEFAULT 2.0,
    observations INTEGER NOT NULL DEFAULT 0,
    updated_at   TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (tenant_id, category, source_type)
);
CREATE INDEX IF NOT EXISTS idx_compiler_gates_updated
    ON compiler_gates (updated_at);

-- Entity health priors (JSON stored as TEXT)
CREATE TABLE IF NOT EXISTS compiler_health_priors (
    tenant_id    TEXT NOT NULL DEFAULT '',
    entity_id    TEXT NOT NULL,
    priors       TEXT NOT NULL DEFAULT '{}',
    updated_at   TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (tenant_id, entity_id)
);

-- Learned keywords
CREATE TABLE IF NOT EXISTS compiler_learned_keywords (
    keyword               TEXT NOT NULL,
    category              TEXT NOT NULL,
    weight                REAL NOT NULL DEFAULT 1.0,
    confidence            REAL NOT NULL DEFAULT 0.0,
    total_observations    INTEGER NOT NULL DEFAULT 0,
    positive_observations INTEGER NOT NULL DEFAULT 0,
    promoted              INTEGER NOT NULL DEFAULT 0,
    promoted_at           TEXT,
    created_at            TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at            TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (keyword, category)
);
CREATE INDEX IF NOT EXISTS idx_compiler_keywords_promoted
    ON compiler_learned_keywords (promoted);
