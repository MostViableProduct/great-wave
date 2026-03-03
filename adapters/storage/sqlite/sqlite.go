// Package sqlite provides a SQLite storage adapter for the contextual
// compiler's gate, health, and keyword stores.
//
// The caller is responsible for importing a SQLite driver (e.g.,
// _ "modernc.org/sqlite") and providing an open *sql.DB connection.
// For in-memory databases, use the DSN ":memory:".
package sqlite

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"time"

	"github.com/MostViableProduct/great-wave/pkg/classifier"
	"github.com/MostViableProduct/great-wave/pkg/gate"
	"github.com/MostViableProduct/great-wave/pkg/health"
)

// Store implements gate.GateStore, health.HealthStore, and keywords.KeywordStore
// backed by SQLite.
type Store struct {
	db *sql.DB
}

// New creates a Store from an existing database connection.
func New(db *sql.DB) *Store {
	return &Store{db: db}
}

// EnsureSchema creates the required tables if they don't exist.
func (s *Store) EnsureSchema() error {
	for _, stmt := range schemaStatements {
		if _, err := s.db.Exec(stmt); err != nil {
			return fmt.Errorf("ensure schema: %w", err)
		}
	}
	return nil
}

// --- gate.GateStore ---

// LoadGateEntries loads up to maxEntries gate entries ordered by most recently updated.
func (s *Store) LoadGateEntries(maxEntries int) ([]gate.GateEntry, error) {
	rows, err := s.db.Query(`
		SELECT tenant_id, category, source_type, alpha, beta, observations, updated_at
		FROM compiler_gates
		ORDER BY updated_at DESC
		LIMIT ?
	`, maxEntries)
	if err != nil {
		return nil, fmt.Errorf("load gate entries: %w", err)
	}
	defer rows.Close()

	var entries []gate.GateEntry
	for rows.Next() {
		var e gate.GateEntry
		var updatedAt string
		if err := rows.Scan(
			&e.Key.TenantID,
			&e.Key.Category,
			&e.Key.SourceType,
			&e.Prior.Alpha,
			&e.Prior.Beta,
			&e.Prior.Observations,
			&updatedAt,
		); err != nil {
			return nil, fmt.Errorf("scan gate entry: %w", err)
		}
		if t, err := time.Parse("2006-01-02 15:04:05", updatedAt); err == nil {
			e.Prior.UpdatedAt = t
		}
		entries = append(entries, e)
	}
	return entries, rows.Err()
}

// FlushGateEntries upserts gate entries into the database.
func (s *Store) FlushGateEntries(entries []gate.GateEntry) error {
	tx, err := s.db.Begin()
	if err != nil {
		return fmt.Errorf("begin gate flush tx: %w", err)
	}
	defer tx.Rollback()

	stmt, err := tx.Prepare(`
		INSERT INTO compiler_gates (tenant_id, category, source_type, alpha, beta, observations, updated_at)
		VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
		ON CONFLICT (tenant_id, category, source_type)
		DO UPDATE SET alpha = excluded.alpha, beta = excluded.beta,
		              observations = excluded.observations, updated_at = datetime('now')
	`)
	if err != nil {
		return fmt.Errorf("prepare gate flush: %w", err)
	}
	defer stmt.Close()

	for _, e := range entries {
		if _, err := stmt.Exec(
			e.Key.TenantID, e.Key.Category, e.Key.SourceType,
			e.Prior.Alpha, e.Prior.Beta, e.Prior.Observations,
		); err != nil {
			return fmt.Errorf("flush gate entry: %w", err)
		}
	}

	return tx.Commit()
}

// --- health.HealthStore ---

// LoadHealthPriors loads up to maxEntries entity health priors.
func (s *Store) LoadHealthPriors(maxEntries int) ([]health.EntityPriors, error) {
	rows, err := s.db.Query(`
		SELECT tenant_id, entity_id, priors FROM compiler_health_priors LIMIT ?
	`, maxEntries)
	if err != nil {
		return nil, fmt.Errorf("load health priors: %w", err)
	}
	defer rows.Close()

	var entries []health.EntityPriors
	for rows.Next() {
		var ep health.EntityPriors
		var priorsText string
		if err := rows.Scan(&ep.TenantID, &ep.EntityID, &priorsText); err != nil {
			return nil, fmt.Errorf("scan health prior: %w", err)
		}

		priors, err := decodePriors([]byte(priorsText))
		if err != nil {
			return nil, fmt.Errorf("decode priors for %s/%s: %w", ep.TenantID, ep.EntityID, err)
		}
		ep.Priors = priors
		entries = append(entries, ep)
	}
	return entries, rows.Err()
}

// FlushHealthPriors upserts entity health priors.
func (s *Store) FlushHealthPriors(entries []health.EntityPriors) error {
	tx, err := s.db.Begin()
	if err != nil {
		return fmt.Errorf("begin health flush tx: %w", err)
	}
	defer tx.Rollback()

	stmt, err := tx.Prepare(`
		INSERT INTO compiler_health_priors (tenant_id, entity_id, priors, updated_at)
		VALUES (?, ?, ?, datetime('now'))
		ON CONFLICT (tenant_id, entity_id)
		DO UPDATE SET priors = excluded.priors, updated_at = datetime('now')
	`)
	if err != nil {
		return fmt.Errorf("prepare health flush: %w", err)
	}
	defer stmt.Close()

	for _, ep := range entries {
		priorsJSON, err := encodePriors(ep.Priors)
		if err != nil {
			return fmt.Errorf("encode priors for %s/%s: %w", ep.TenantID, ep.EntityID, err)
		}
		if _, err := stmt.Exec(ep.TenantID, ep.EntityID, string(priorsJSON)); err != nil {
			return fmt.Errorf("flush health prior: %w", err)
		}
	}

	return tx.Commit()
}

// --- keywords.KeywordStore ---

// UpsertKeyword records an observation for a keyword/category pair.
func (s *Store) UpsertKeyword(keyword, category string) error {
	_, err := s.db.Exec(`
		INSERT INTO compiler_learned_keywords (keyword, category, total_observations, positive_observations, confidence, weight, updated_at)
		VALUES (?, ?, 1, 1, 1.0, 1.0, datetime('now'))
		ON CONFLICT (keyword, category) DO UPDATE SET
			total_observations = compiler_learned_keywords.total_observations + 1,
			positive_observations = compiler_learned_keywords.positive_observations + 1,
			confidence = CAST(compiler_learned_keywords.positive_observations + 1 AS REAL) /
			             CAST(compiler_learned_keywords.total_observations + 1 AS REAL),
			weight = CASE
				WHEN CAST(compiler_learned_keywords.positive_observations + 1 AS REAL) /
				     CAST(compiler_learned_keywords.total_observations + 1 AS REAL) >= 0.7
				THEN 2.0
				ELSE 1.0
			END,
			updated_at = datetime('now')
	`, keyword, category)
	if err != nil {
		return fmt.Errorf("upsert keyword %q/%q: %w", keyword, category, err)
	}
	return nil
}

// PromoteKeywords marks keywords with sufficient confidence and observations as promoted.
func (s *Store) PromoteKeywords(minConfidence float64, minObservations int) ([]classifier.LearnedKeyword, error) {
	// SQLite doesn't support UPDATE ... RETURNING, so we do it in two steps.
	tx, err := s.db.Begin()
	if err != nil {
		return nil, fmt.Errorf("begin promote tx: %w", err)
	}
	defer tx.Rollback()

	_, err = tx.Exec(`
		UPDATE compiler_learned_keywords
		SET promoted = 1, promoted_at = datetime('now')
		WHERE confidence >= ? AND total_observations >= ?
	`, minConfidence, minObservations)
	if err != nil {
		return nil, fmt.Errorf("promote keywords: %w", err)
	}

	rows, err := tx.Query(`
		SELECT keyword, category, weight, confidence
		FROM compiler_learned_keywords
		WHERE promoted = 1 AND confidence >= ? AND total_observations >= ?
	`, minConfidence, minObservations)
	if err != nil {
		return nil, fmt.Errorf("select promoted: %w", err)
	}
	defer rows.Close()

	var promoted []classifier.LearnedKeyword
	for rows.Next() {
		var kw classifier.LearnedKeyword
		if err := rows.Scan(&kw.Keyword, &kw.Category, &kw.Weight, &kw.Confidence); err != nil {
			return nil, fmt.Errorf("scan promoted keyword: %w", err)
		}
		promoted = append(promoted, kw)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}

	return promoted, tx.Commit()
}

// DemoteKeywords un-promotes keywords that no longer meet the threshold.
func (s *Store) DemoteKeywords(minConfidence float64, minObservations int) error {
	_, err := s.db.Exec(`
		UPDATE compiler_learned_keywords
		SET promoted = 0
		WHERE promoted = 1 AND (confidence < ? OR total_observations < ?)
	`, minConfidence, minObservations)
	if err != nil {
		return fmt.Errorf("demote keywords: %w", err)
	}
	return nil
}

// LoadPromotedKeywords returns up to limit promoted keywords ordered by confidence.
func (s *Store) LoadPromotedKeywords(limit int) ([]classifier.LearnedKeyword, error) {
	rows, err := s.db.Query(`
		SELECT keyword, category, weight, confidence
		FROM compiler_learned_keywords
		WHERE promoted = 1
		ORDER BY confidence DESC
		LIMIT ?
	`, limit)
	if err != nil {
		return nil, fmt.Errorf("load promoted keywords: %w", err)
	}
	defer rows.Close()

	var keywords []classifier.LearnedKeyword
	for rows.Next() {
		var kw classifier.LearnedKeyword
		if err := rows.Scan(&kw.Keyword, &kw.Category, &kw.Weight, &kw.Confidence); err != nil {
			return nil, fmt.Errorf("scan promoted keyword: %w", err)
		}
		keywords = append(keywords, kw)
	}
	return keywords, rows.Err()
}

// --- helpers ---

func encodePriors(priors map[string][2]float64) ([]byte, error) {
	return json.Marshal(priors)
}

func decodePriors(data []byte) (map[string][2]float64, error) {
	var priors map[string][2]float64
	if err := json.Unmarshal(data, &priors); err != nil {
		return nil, err
	}
	return priors, nil
}

var schemaStatements = []string{
	`CREATE TABLE IF NOT EXISTS compiler_gates (
		tenant_id    TEXT NOT NULL DEFAULT '',
		category     TEXT NOT NULL,
		source_type  TEXT NOT NULL,
		alpha        REAL NOT NULL DEFAULT 2.0,
		beta         REAL NOT NULL DEFAULT 2.0,
		observations INTEGER NOT NULL DEFAULT 0,
		updated_at   TEXT DEFAULT (datetime('now')),
		PRIMARY KEY (tenant_id, category, source_type)
	)`,
	`CREATE INDEX IF NOT EXISTS idx_compiler_gates_updated
		ON compiler_gates (updated_at)`,
	`CREATE TABLE IF NOT EXISTS compiler_health_priors (
		tenant_id    TEXT NOT NULL DEFAULT '',
		entity_id    TEXT NOT NULL,
		priors       TEXT NOT NULL DEFAULT '{}',
		updated_at   TEXT DEFAULT (datetime('now')),
		PRIMARY KEY (tenant_id, entity_id)
	)`,
	`CREATE TABLE IF NOT EXISTS compiler_learned_keywords (
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
	)`,
	`CREATE INDEX IF NOT EXISTS idx_compiler_keywords_promoted
		ON compiler_learned_keywords (promoted)`,
}
