package health

import (
	"math"
	"testing"
)

const epsilon = 1e-6

func TestDefaultConfig(t *testing.T) {
	cfg := DefaultConfig()
	if len(cfg.Severities) != 4 {
		t.Errorf("DefaultConfig should have 4 severity levels, got %d", len(cfg.Severities))
	}

	// Weights should sum to 1.0
	total := 0.0
	for _, s := range cfg.Severities {
		total += s.Weight
	}
	if math.Abs(total-1.0) > epsilon {
		t.Errorf("Severity weights sum to %v, should be 1.0", total)
	}
}

func TestNewModel(t *testing.T) {
	m := NewModel(DefaultConfig(), nil)
	if m.EntryCount() != 0 {
		t.Errorf("New model should have 0 entries, got %d", m.EntryCount())
	}
}

func TestGetOrCreate(t *testing.T) {
	m := NewModel(DefaultConfig(), nil)

	p := m.GetOrCreate("t1", "entity-1")
	if p.TenantID != "t1" || p.EntityID != "entity-1" {
		t.Errorf("Unexpected priors: %+v", p)
	}
	if len(p.Priors) != 4 {
		t.Errorf("Expected 4 severity priors, got %d", len(p.Priors))
	}
	if m.EntryCount() != 1 {
		t.Errorf("Expected 1 entry, got %d", m.EntryCount())
	}

	// Second call should return same priors
	p2 := m.GetOrCreate("t1", "entity-1")
	if p != p2 {
		t.Error("GetOrCreate should return same pointer for same entity")
	}
}

func TestScoreDefaultPriors(t *testing.T) {
	m := NewModel(DefaultConfig(), nil)
	score := m.Score("t1", "entity-1")

	// Default priors should give a high health score
	// critical: 5/(5+1)=0.833, regression: 5/(5+1)=0.833,
	// warning: 3/(3+1)=0.75, improvement: 1/(1+3)=0.25
	// weighted: 0.4*0.833 + 0.25*0.833 + 0.15*0.75 + 0.20*0.25 = 0.705
	// * 100 = 70.5
	if score.Score < 65 || score.Score > 75 {
		t.Errorf("Default score = %v, expected ~70", score.Score)
	}

	// Confidence interval should exist
	if score.ConfidenceIntervalLower >= score.ConfidenceIntervalUpper {
		t.Errorf("CI invalid: lower=%v, upper=%v", score.ConfidenceIntervalLower, score.ConfidenceIntervalUpper)
	}
}

func TestUpdateFromEvent_NegativeSeverity(t *testing.T) {
	m := NewModel(DefaultConfig(), nil)
	initialScore := m.Score("t1", "e1").Score

	// Record critical events
	for i := 0; i < 10; i++ {
		m.UpdateFromEvent("t1", "e1", "critical", "critical", 1.0)
	}

	newScore := m.Score("t1", "e1").Score
	if newScore >= initialScore {
		t.Errorf("Critical events should decrease score: %v -> %v", initialScore, newScore)
	}
}

func TestUpdateFromEvent_PositiveSeverity(t *testing.T) {
	m := NewModel(DefaultConfig(), nil)
	initialScore := m.Score("t1", "e1").Score

	// Record improvement events
	for i := 0; i < 10; i++ {
		m.UpdateFromEvent("t1", "e1", "improvement", "improvement", 1.0)
	}

	newScore := m.Score("t1", "e1").Score
	if newScore <= initialScore {
		t.Errorf("Improvement events should increase score: %v -> %v", initialScore, newScore)
	}
}

func TestConfidenceIntervalNarrows(t *testing.T) {
	m := NewModel(DefaultConfig(), nil)
	initialCI := m.Score("t1", "e1")
	initialWidth := initialCI.ConfidenceIntervalUpper - initialCI.ConfidenceIntervalLower

	// Add many observations
	for i := 0; i < 50; i++ {
		m.UpdateFromEvent("t1", "e1", "warning", "warning", 0.5)
	}

	newCI := m.Score("t1", "e1")
	newWidth := newCI.ConfidenceIntervalUpper - newCI.ConfidenceIntervalLower

	if newWidth >= initialWidth {
		t.Errorf("CI should narrow with more observations: %v -> %v", initialWidth, newWidth)
	}
}

func TestEviction(t *testing.T) {
	cfg := DefaultConfig()
	cfg.MaxEntries = 5
	m := NewModel(cfg, nil)

	for i := 0; i < 10; i++ {
		m.GetOrCreate("t1", string(rune('a'+i)))
	}

	// Should have evicted to 90% of max = 4, then added the new one = 5
	if m.EntryCount() > 5 {
		t.Errorf("Expected at most 5 entries after eviction, got %d", m.EntryCount())
	}
}

func TestCustomSeverityLevels(t *testing.T) {
	cfg := Config{
		Severities: []SeverityLevel{
			{Name: "blocker", Weight: 0.5, Direction: "negative", DefaultAlpha: 5.0, DefaultBeta: 1.0},
			{Name: "enhancement", Weight: 0.5, Direction: "positive", DefaultAlpha: 1.0, DefaultBeta: 3.0},
		},
		MaxEntries: 100,
	}

	m := NewModel(cfg, nil)
	score := m.Score("t1", "e1")

	// blocker: 5/6 = 0.833, enhancement: 1/4 = 0.25
	// weighted: 0.5*0.833 + 0.5*0.25 = 0.542 * 100 = 54.2
	if score.Score < 50 || score.Score > 60 {
		t.Errorf("Custom severity score = %v, expected ~54", score.Score)
	}
}

// In-memory store for testing persistence
type memHealthStore struct {
	entries []EntityPriors
}

func (m *memHealthStore) LoadHealthPriors(maxEntries int) ([]EntityPriors, error) {
	if maxEntries > 0 && len(m.entries) > maxEntries {
		return m.entries[:maxEntries], nil
	}
	return m.entries, nil
}

func (m *memHealthStore) FlushHealthPriors(entries []EntityPriors) error {
	m.entries = entries
	return nil
}

func TestLoadFlush(t *testing.T) {
	store := &memHealthStore{}
	m := NewModel(DefaultConfig(), store)

	m.UpdateFromEvent("t1", "e1", "critical", "critical", 1.0)

	if err := m.Flush(); err != nil {
		t.Fatalf("Flush failed: %v", err)
	}
	if len(store.entries) != 1 {
		t.Errorf("Expected 1 flushed entry, got %d", len(store.entries))
	}

	// Load into new model
	m2 := NewModel(DefaultConfig(), store)
	if err := m2.Load(); err != nil {
		t.Fatalf("Load failed: %v", err)
	}
	if m2.EntryCount() != 1 {
		t.Errorf("Expected 1 loaded entry, got %d", m2.EntryCount())
	}
}

func TestScoreBounds(t *testing.T) {
	m := NewModel(DefaultConfig(), nil)

	// Flood with critical events
	for i := 0; i < 1000; i++ {
		m.UpdateFromEvent("t1", "e1", "critical", "critical", 1.0)
	}

	score := m.Score("t1", "e1")
	if score.Score < 0 || score.Score > 100 {
		t.Errorf("Score out of bounds: %v", score.Score)
	}
	if score.ConfidenceIntervalLower < 0 {
		t.Errorf("CI lower out of bounds: %v", score.ConfidenceIntervalLower)
	}
	if score.ConfidenceIntervalUpper > 100 {
		t.Errorf("CI upper out of bounds: %v", score.ConfidenceIntervalUpper)
	}
}
