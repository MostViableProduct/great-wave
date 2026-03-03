package gate

import (
	"testing"
)

func TestNewBayesianGate(t *testing.T) {
	g := New(DefaultConfig(), nil)
	if g.EntryCount() != 0 {
		t.Errorf("New gate should be empty, got %d entries", g.EntryCount())
	}
}

func TestShouldSkipLLM_LowConfidence(t *testing.T) {
	g := New(DefaultConfig(), nil)
	// Low heuristic confidence should never skip
	if g.ShouldSkipLLM("t1", "perf", "metric", 0.5) {
		t.Error("Should not skip with low heuristic confidence")
	}
}

func TestShouldSkipLLM_NoObservations(t *testing.T) {
	g := New(DefaultConfig(), nil)
	// No observations — uninformative prior (2,2) has agreement=0.5 < 0.75
	if g.ShouldSkipLLM("t1", "perf", "metric", 0.9) {
		t.Error("Should not skip with no observations")
	}
}

func TestShouldSkipLLM_HighAgreement(t *testing.T) {
	cfg := DefaultConfig()
	g := New(cfg, nil)

	// Record many agreements to build confidence
	for i := 0; i < 100; i++ {
		g.Update("t1", "performance", "metric", true)
	}

	if !g.ShouldSkipLLM("t1", "performance", "metric", 0.9) {
		t.Error("Should skip after 100 agreements")
	}
}

func TestShouldSkipLLM_ShadowMode(t *testing.T) {
	cfg := DefaultConfig()
	cfg.ShadowMode = true
	shadowSkips := 0
	g := New(cfg, nil)
	g.OnShadowSkip = func() { shadowSkips++ }

	for i := 0; i < 100; i++ {
		g.Update("t1", "performance", "metric", true)
	}

	// Shadow mode: gate evaluates but never skips
	if g.ShouldSkipLLM("t1", "performance", "metric", 0.9) {
		t.Error("Shadow mode should never return true")
	}
	if shadowSkips != 1 {
		t.Errorf("Expected 1 shadow skip, got %d", shadowSkips)
	}
}

func TestUpdate(t *testing.T) {
	g := New(DefaultConfig(), nil)
	g.Update("t1", "reliability", "error", true)
	g.Update("t1", "reliability", "error", true)
	g.Update("t1", "reliability", "error", false)

	if g.EntryCount() != 1 {
		t.Errorf("Expected 1 entry, got %d", g.EntryCount())
	}

	agreement := g.GetAgreement("t1", "reliability", "error")
	// Prior: Alpha=2, Beta=2, plus 2 agreements and 1 disagreement
	// Alpha=4, Beta=3, Agreement = 4/7 ≈ 0.571
	expected := 4.0 / 7.0
	if abs(agreement-expected) > 0.01 {
		t.Errorf("Agreement = %v, want ~%v", agreement, expected)
	}
}

func TestLRUEviction(t *testing.T) {
	cfg := DefaultConfig()
	cfg.MaxEntries = 3
	g := New(cfg, nil)

	g.Update("t1", "a", "s1", true)
	g.Update("t1", "b", "s1", true)
	g.Update("t1", "c", "s1", true)
	g.Update("t1", "d", "s1", true) // should evict "a"

	if g.EntryCount() != 3 {
		t.Errorf("Expected 3 entries after eviction, got %d", g.EntryCount())
	}
}

func TestHierarchicalFallback(t *testing.T) {
	g := New(DefaultConfig(), nil)

	// Add observations to one source type
	for i := 0; i < 50; i++ {
		g.Update("t1", "performance", "metric", true)
	}

	// A new source type in the same category should get pooled prior
	agreement := g.GetAgreement("t1", "performance", "trace")
	// Pooled prior should be better than uninformative (0.5)
	if agreement <= 0.5 {
		t.Errorf("Hierarchical fallback agreement %v should be > 0.5", agreement)
	}
}

func TestMetricsCallbacks(t *testing.T) {
	evals := 0
	skips := 0

	cfg := DefaultConfig()
	g := New(cfg, nil)
	g.OnEvaluation = func() { evals++ }
	g.OnSkip = func() { skips++ }

	g.ShouldSkipLLM("t1", "perf", "m", 0.9) // eval, no skip
	if evals != 1 {
		t.Errorf("Expected 1 evaluation, got %d", evals)
	}
	if skips != 0 {
		t.Errorf("Expected 0 skips, got %d", skips)
	}
}

// In-memory store for testing persistence
type memStore struct {
	entries []GateEntry
}

func (m *memStore) LoadGateEntries(maxEntries int) ([]GateEntry, error) {
	if len(m.entries) > maxEntries {
		return m.entries[:maxEntries], nil
	}
	return m.entries, nil
}

func (m *memStore) FlushGateEntries(entries []GateEntry) error {
	m.entries = append(m.entries, entries...)
	return nil
}

func TestLoadFlush(t *testing.T) {
	store := &memStore{}
	g := New(DefaultConfig(), store)

	g.Update("t1", "security", "alert", true)
	g.Update("t1", "security", "alert", true)

	if err := g.Flush(); err != nil {
		t.Fatalf("Flush failed: %v", err)
	}

	if len(store.entries) != 1 {
		t.Errorf("Expected 1 flushed entry, got %d", len(store.entries))
	}

	// Load into a new gate
	g2 := New(DefaultConfig(), store)
	if err := g2.Load(); err != nil {
		t.Fatalf("Load failed: %v", err)
	}

	if g2.EntryCount() != 1 {
		t.Errorf("Expected 1 loaded entry, got %d", g2.EntryCount())
	}
}

func abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}
