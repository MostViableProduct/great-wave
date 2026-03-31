package postgres

import (
	"database/sql"
	"os"
	"testing"

	_ "github.com/lib/pq"

	"github.com/MostViableProduct/great-wave/pkg/gate"
	"github.com/MostViableProduct/great-wave/pkg/health"
)

func testDB(t *testing.T) *sql.DB {
	t.Helper()

	dsn := os.Getenv("TEST_POSTGRES_DSN")
	if dsn == "" {
		t.Skip("TEST_POSTGRES_DSN not set, skipping Postgres tests")
	}

	db, err := sql.Open("postgres", dsn)
	if err != nil {
		t.Fatalf("open postgres: %v", err)
	}
	t.Cleanup(func() {
		// Clean up test tables
		db.Exec("DROP TABLE IF EXISTS compiler_gates")
		db.Exec("DROP TABLE IF EXISTS compiler_health_priors")
		db.Exec("DROP TABLE IF EXISTS compiler_learned_keywords")
		db.Close()
	})

	if err := db.Ping(); err != nil {
		t.Fatalf("ping postgres: %v", err)
	}

	return db
}

func TestEnsureSchema(t *testing.T) {
	db := testDB(t)
	store := New(db)

	if err := store.EnsureSchema(); err != nil {
		t.Fatalf("EnsureSchema: %v", err)
	}

	// Calling twice should be idempotent
	if err := store.EnsureSchema(); err != nil {
		t.Fatalf("EnsureSchema (second call): %v", err)
	}
}

func TestGateStore(t *testing.T) {
	db := testDB(t)
	store := New(db)
	if err := store.EnsureSchema(); err != nil {
		t.Fatalf("EnsureSchema: %v", err)
	}

	entries := []gate.GateEntry{
		{
			Key:   gate.GateKey{TenantID: "t1", Category: "performance", SourceType: "prometheus"},
			Prior: gate.BetaPrior{Alpha: 10.0, Beta: 2.0, Observations: 12},
		},
		{
			Key:   gate.GateKey{TenantID: "t1", Category: "reliability", SourceType: "sentry"},
			Prior: gate.BetaPrior{Alpha: 8.0, Beta: 4.0, Observations: 12},
		},
	}

	// Flush
	if err := store.FlushGateEntries(entries); err != nil {
		t.Fatalf("FlushGateEntries: %v", err)
	}

	// Load
	loaded, err := store.LoadGateEntries(100)
	if err != nil {
		t.Fatalf("LoadGateEntries: %v", err)
	}
	if len(loaded) != 2 {
		t.Fatalf("expected 2 entries, got %d", len(loaded))
	}

	// Verify values
	found := false
	for _, e := range loaded {
		if e.Key.Category == "performance" && e.Key.SourceType == "prometheus" {
			found = true
			if e.Prior.Alpha != 10.0 {
				t.Errorf("alpha = %v, want 10.0", e.Prior.Alpha)
			}
			if e.Prior.Beta != 2.0 {
				t.Errorf("beta = %v, want 2.0", e.Prior.Beta)
			}
			if e.Prior.Observations != 12 {
				t.Errorf("observations = %d, want 12", e.Prior.Observations)
			}
		}
	}
	if !found {
		t.Error("performance/prometheus entry not found")
	}

	// Upsert idempotency
	entries[0].Prior.Alpha = 15.0
	entries[0].Prior.Observations = 17
	if err := store.FlushGateEntries(entries[:1]); err != nil {
		t.Fatalf("FlushGateEntries (upsert): %v", err)
	}

	loaded, err = store.LoadGateEntries(100)
	if err != nil {
		t.Fatalf("LoadGateEntries after upsert: %v", err)
	}
	for _, e := range loaded {
		if e.Key.Category == "performance" {
			if e.Prior.Alpha != 15.0 {
				t.Errorf("after upsert: alpha = %v, want 15.0", e.Prior.Alpha)
			}
		}
	}

	// Test limit
	limited, err := store.LoadGateEntries(1)
	if err != nil {
		t.Fatalf("LoadGateEntries with limit: %v", err)
	}
	if len(limited) != 1 {
		t.Errorf("expected 1 entry with limit, got %d", len(limited))
	}
}

func TestHealthStore(t *testing.T) {
	db := testDB(t)
	store := New(db)
	if err := store.EnsureSchema(); err != nil {
		t.Fatalf("EnsureSchema: %v", err)
	}

	entries := []health.EntityPriors{
		{
			TenantID: "t1",
			EntityID: "api-gateway",
			Priors: map[string][2]float64{
				"critical":    {5.0, 1.0},
				"regression":  {5.0, 1.0},
				"improvement": {1.0, 3.0},
			},
		},
		{
			TenantID: "t1",
			EntityID: "auth-service",
			Priors: map[string][2]float64{
				"critical": {3.0, 2.0},
			},
		},
	}

	// Flush
	if err := store.FlushHealthPriors(entries); err != nil {
		t.Fatalf("FlushHealthPriors: %v", err)
	}

	// Load
	loaded, err := store.LoadHealthPriors(1000)
	if err != nil {
		t.Fatalf("LoadHealthPriors: %v", err)
	}
	if len(loaded) != 2 {
		t.Fatalf("expected 2 entries, got %d", len(loaded))
	}

	// Verify
	for _, ep := range loaded {
		if ep.EntityID == "api-gateway" {
			if ab, ok := ep.Priors["critical"]; !ok || ab[0] != 5.0 || ab[1] != 1.0 {
				t.Errorf("api-gateway critical priors = %v, want [5.0, 1.0]", ab)
			}
			if ab, ok := ep.Priors["improvement"]; !ok || ab[0] != 1.0 || ab[1] != 3.0 {
				t.Errorf("api-gateway improvement priors = %v, want [1.0, 3.0]", ab)
			}
		}
	}

	// Upsert
	entries[0].Priors["critical"] = [2]float64{6.0, 2.0}
	if err := store.FlushHealthPriors(entries[:1]); err != nil {
		t.Fatalf("FlushHealthPriors (upsert): %v", err)
	}

	loaded, err = store.LoadHealthPriors(1000)
	if err != nil {
		t.Fatalf("LoadHealthPriors after upsert: %v", err)
	}
	for _, ep := range loaded {
		if ep.EntityID == "api-gateway" {
			if ab := ep.Priors["critical"]; ab[0] != 6.0 || ab[1] != 2.0 {
				t.Errorf("after upsert: critical = %v, want [6.0, 2.0]", ab)
			}
		}
	}
}

func TestWeakenKeyword(t *testing.T) {
	db := testDB(t)
	store := New(db)
	if err := store.EnsureSchema(); err != nil {
		t.Fatalf("EnsureSchema: %v", err)
	}

	// Upsert a keyword with 15 observations
	for i := 0; i < 15; i++ {
		if err := store.UpsertKeyword("latency_spike", "performance"); err != nil {
			t.Fatalf("UpsertKeyword iteration %d: %v", i, err)
		}
	}

	// Promote it first
	promoted, err := store.PromoteKeywords(0.5, 10)
	if err != nil {
		t.Fatalf("PromoteKeywords: %v", err)
	}
	if len(promoted) != 1 {
		t.Fatalf("expected 1 promoted, got %d", len(promoted))
	}

	// Weaken the keyword 6 times — drops observations from 15 to 9
	for i := 0; i < 6; i++ {
		if err := store.WeakenKeyword("latency_spike"); err != nil {
			t.Fatalf("WeakenKeyword iteration %d: %v", i, err)
		}
	}

	// Demote with threshold that the keyword no longer meets
	if err := store.DemoteKeywords(0.5, 10); err != nil {
		t.Fatalf("DemoteKeywords: %v", err)
	}

	// Should be demoted now (observations < 10)
	loaded, err := store.LoadPromotedKeywords(100)
	if err != nil {
		t.Fatalf("LoadPromotedKeywords after weaken: %v", err)
	}
	if len(loaded) != 0 {
		t.Errorf("expected 0 promoted after weakening, got %d", len(loaded))
	}

	// Weaken a non-existent keyword — should not error
	if err := store.WeakenKeyword("nonexistent_keyword"); err != nil {
		t.Fatalf("WeakenKeyword on missing keyword: %v", err)
	}
}

func TestKeywordStore(t *testing.T) {
	db := testDB(t)
	store := New(db)
	if err := store.EnsureSchema(); err != nil {
		t.Fatalf("EnsureSchema: %v", err)
	}

	// Upsert keyword
	if err := store.UpsertKeyword("latency_spike", "performance"); err != nil {
		t.Fatalf("UpsertKeyword: %v", err)
	}

	// Upsert same keyword again (should increment observations)
	for i := 0; i < 14; i++ {
		if err := store.UpsertKeyword("latency_spike", "performance"); err != nil {
			t.Fatalf("UpsertKeyword iteration %d: %v", i, err)
		}
	}

	// Promote — should return this keyword (15 observations, high confidence)
	promoted, err := store.PromoteKeywords(0.5, 10)
	if err != nil {
		t.Fatalf("PromoteKeywords: %v", err)
	}
	if len(promoted) != 1 {
		t.Fatalf("expected 1 promoted keyword, got %d", len(promoted))
	}
	if promoted[0].Keyword != "latency_spike" {
		t.Errorf("promoted keyword = %q, want %q", promoted[0].Keyword, "latency_spike")
	}

	// Load promoted
	loaded, err := store.LoadPromotedKeywords(100)
	if err != nil {
		t.Fatalf("LoadPromotedKeywords: %v", err)
	}
	if len(loaded) != 1 {
		t.Fatalf("expected 1 loaded promoted keyword, got %d", len(loaded))
	}

	// Demote with very high threshold
	if err := store.DemoteKeywords(0.99, 100); err != nil {
		t.Fatalf("DemoteKeywords: %v", err)
	}

	// Should be demoted now
	loaded, err = store.LoadPromotedKeywords(100)
	if err != nil {
		t.Fatalf("LoadPromotedKeywords after demote: %v", err)
	}
	if len(loaded) != 0 {
		t.Errorf("expected 0 promoted after demote, got %d", len(loaded))
	}
}
