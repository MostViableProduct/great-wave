// Package health implements a Bayesian per-entity health model using
// per-severity Beta distributions.
//
// Each entity's health is tracked as four independent Beta distributions,
// one per severity level (configurable). The composite health score is a
// weighted combination of posterior means from each distribution.
//
// This enables uncertainty-aware health scoring: new entities start with
// wide confidence intervals that narrow as more events are observed.
package health

import (
	"math"
	"sort"
	"sync"
	"time"
)

// SeverityLevel defines a configurable severity dimension for health tracking.
type SeverityLevel struct {
	// Name identifies this severity (e.g., "critical", "regression").
	Name string `json:"name" yaml:"name"`
	// Weight is the relative importance in the composite score (should sum to 1.0).
	Weight float64 `json:"weight" yaml:"weight"`
	// Direction: "negative" means events decrease health, "positive" means they increase it.
	Direction string `json:"direction" yaml:"direction"`
	// DefaultAlpha is the initial Beta alpha parameter.
	DefaultAlpha float64 `json:"default_alpha" yaml:"default_alpha"`
	// DefaultBeta is the initial Beta beta parameter.
	DefaultBeta float64 `json:"default_beta" yaml:"default_beta"`
}

// Config holds the health model configuration.
type Config struct {
	// Severities defines the severity levels to track.
	Severities []SeverityLevel `json:"severities" yaml:"severities"`
	// MaxEntries is the maximum entities to keep in memory (default: 10000).
	MaxEntries int `json:"max_entries" yaml:"max_entries"`
}

// DefaultConfig returns a health config matching the original Big Wave model.
func DefaultConfig() Config {
	return Config{
		Severities: []SeverityLevel{
			{Name: "critical", Weight: 0.40, Direction: "negative", DefaultAlpha: 5.0, DefaultBeta: 1.0},
			{Name: "regression", Weight: 0.25, Direction: "negative", DefaultAlpha: 5.0, DefaultBeta: 1.0},
			{Name: "warning", Weight: 0.15, Direction: "negative", DefaultAlpha: 3.0, DefaultBeta: 1.0},
			{Name: "improvement", Weight: 0.20, Direction: "positive", DefaultAlpha: 1.0, DefaultBeta: 3.0},
		},
		MaxEntries: 10000,
	}
}

// EntityPriors holds the Beta distribution parameters for each severity level of an entity.
type EntityPriors struct {
	TenantID  string
	EntityID  string
	// Priors maps severity name → (alpha, beta) pair.
	Priors    map[string][2]float64 // [0]=alpha, [1]=beta
	UpdatedAt time.Time
}

// HealthStore defines the persistence interface for health model entries.
type HealthStore interface {
	LoadHealthPriors(maxEntries int) ([]EntityPriors, error)
	FlushHealthPriors(entries []EntityPriors) error
}

// HealthScore is the computed health score for an entity.
type HealthScore struct {
	Score                   float64 `json:"score"`
	ConfidenceIntervalLower float64 `json:"confidence_interval_lower"`
	ConfidenceIntervalUpper float64 `json:"confidence_interval_upper"`
}

// Model manages per-entity severity priors with an in-memory cache.
type Model struct {
	mu              sync.RWMutex
	entries         map[string]*EntityPriors // key: "tenant_id:entity_id"
	config          Config
	store           HealthStore // optional
	severities      map[string]SeverityLevel
	validSeveritySet map[string]bool // pre-computed, immutable after construction
}

// NewModel creates a new health model.
func NewModel(cfg Config, store HealthStore) *Model {
	if cfg.MaxEntries == 0 {
		cfg.MaxEntries = 10000
	}

	sevMap := make(map[string]SeverityLevel, len(cfg.Severities))
	validSet := make(map[string]bool, len(cfg.Severities))
	for _, s := range cfg.Severities {
		sevMap[s.Name] = s
		validSet[s.Name] = true
	}

	return &Model{
		entries:          make(map[string]*EntityPriors),
		config:           cfg,
		store:            store,
		severities:       sevMap,
		validSeveritySet: validSet,
	}
}

func entityKey(tenantID, entityID string) string {
	return tenantID + ":" + entityID
}

// GetOrCreate returns the priors for an entity, creating defaults if needed.
func (m *Model) GetOrCreate(tenantID, entityID string) *EntityPriors {
	key := entityKey(tenantID, entityID)

	m.mu.RLock()
	entry, exists := m.entries[key]
	m.mu.RUnlock()

	if exists {
		return entry
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	// Double-check
	if entry, exists := m.entries[key]; exists {
		return entry
	}

	// Evict oldest if at capacity. Sort all entries by UpdatedAt once and
	// delete the bottom 10% in a single O(n log n) pass instead of the
	// previous O(n²) repeated linear-scan approach.
	if len(m.entries) >= m.config.MaxEntries {
		evictTarget := int(float64(m.config.MaxEntries) * 0.9)
		toEvict := len(m.entries) - evictTarget
		if toEvict > 0 {
			type kv struct {
				key string
				t   time.Time
			}
			keys := make([]kv, 0, len(m.entries))
			for k, v := range m.entries {
				keys = append(keys, kv{k, v.UpdatedAt})
			}
			sort.Slice(keys, func(i, j int) bool {
				return keys[i].t.Before(keys[j].t)
			})
			for i := 0; i < toEvict; i++ {
				delete(m.entries, keys[i].key)
			}
		}
	}

	priors := m.defaultPriors(tenantID, entityID)
	m.entries[key] = priors
	return priors
}

func (m *Model) defaultPriors(tenantID, entityID string) *EntityPriors {
	p := &EntityPriors{
		TenantID: tenantID,
		EntityID: entityID,
		Priors:   make(map[string][2]float64, len(m.config.Severities)),
	}
	for _, s := range m.config.Severities {
		p.Priors[s.Name] = [2]float64{s.DefaultAlpha, s.DefaultBeta}
	}
	return p
}

// Score computes the composite health score for an entity.
func (m *Model) Score(tenantID, entityID string) HealthScore {
	priors := m.GetOrCreate(tenantID, entityID)
	return m.ComputeScore(priors)
}

// ComputeScore computes the health score from entity priors.
func (m *Model) ComputeScore(priors *EntityPriors) HealthScore {
	score := 0.0
	compositeVar := 0.0

	for _, sev := range m.config.Severities {
		ab, ok := priors.Priors[sev.Name]
		if !ok {
			ab = [2]float64{sev.DefaultAlpha, sev.DefaultBeta}
		}
		alpha, beta := ab[0], ab[1]

		mean := alpha / (alpha + beta)
		variance := betaVariance(alpha, beta)

		score += sev.Weight * mean
		compositeVar += (sev.Weight * sev.Weight) * variance
	}

	score *= 100
	compositeStd := math.Sqrt(compositeVar) * 100

	lower := math.Max(0, score-1.96*compositeStd)
	upper := math.Min(100, score+1.96*compositeStd)

	return HealthScore{
		Score:                   math.Max(0, math.Min(100, score)),
		ConfidenceIntervalLower: lower,
		ConfidenceIntervalUpper: upper,
	}
}

// UpdateFromEvent updates the appropriate severity prior based on an event.
func (m *Model) UpdateFromEvent(tenantID, entityID, severity, category string, confidence float64) {
	priors := m.GetOrCreate(tenantID, entityID)

	m.mu.Lock()
	defer m.mu.Unlock()

	// Update the severity-specific prior
	if ab, ok := priors.Priors[severity]; ok {
		sev := m.severities[severity]
		if sev.Direction == "negative" {
			ab[1] += confidence // increase beta (more events = lower health)
		} else {
			ab[0] += confidence // increase alpha (more events = higher health)
		}
		priors.Priors[severity] = ab
	}

	// Also check if category maps to a severity
	if severity != category {
		if ab, ok := priors.Priors[category]; ok {
			sev := m.severities[category]
			if sev.Direction == "negative" {
				ab[1] += confidence
			} else {
				ab[0] += confidence
			}
			priors.Priors[category] = ab
		}
	}

	priors.UpdatedAt = time.Now()
}

// Load reads health priors from the store.
func (m *Model) Load() error {
	if m.store == nil {
		return nil
	}

	entries, err := m.store.LoadHealthPriors(m.config.MaxEntries)
	if err != nil {
		return err
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	for _, e := range entries {
		key := entityKey(e.TenantID, e.EntityID)
		ep := e
		m.entries[key] = &ep
	}

	return nil
}

// Flush writes all entries to the store.
func (m *Model) Flush() error {
	if m.store == nil {
		return nil
	}

	m.mu.RLock()
	entries := make([]EntityPriors, 0, len(m.entries))
	for _, p := range m.entries {
		entries = append(entries, *p)
	}
	m.mu.RUnlock()

	if len(entries) == 0 {
		return nil
	}

	return m.store.FlushHealthPriors(entries)
}

// EntryCount returns the number of entities tracked.
func (m *Model) EntryCount() int {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return len(m.entries)
}

// IsValidSeverity reports whether the given name is a configured severity level.
func (m *Model) IsValidSeverity(name string) bool {
	_, ok := m.severities[name]
	return ok
}

// ValidSeverities returns the set of configured severity level names.
// The returned map is shared and must not be modified by the caller.
func (m *Model) ValidSeverities() map[string]bool {
	return m.validSeveritySet
}

func betaVariance(alpha, beta float64) float64 {
	return (alpha * beta) / ((alpha + beta) * (alpha + beta) * (alpha + beta + 1))
}
