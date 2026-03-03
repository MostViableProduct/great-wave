// Package gate implements a Bayesian Beta-Binomial gate that learns when a
// heuristic classifier is reliable enough to skip expensive LLM calls.
//
// The gate maintains per-(tenant, category, source_type) Beta priors that track
// the historical agreement rate between heuristic and LLM classifiers. When the
// agreement probability exceeds a threshold with low uncertainty, the gate
// recommends skipping the LLM call entirely.
//
// Features:
//   - Hierarchical fallback: pools observations across source types for rare combinations
//   - LRU eviction: maintains bounded memory via least-recently-used eviction
//   - Shadow mode: evaluates but never actually skips, for safe initial rollout
//   - Persistence: load/flush to any storage backend via GateStore interface
package gate

import (
	"math"
	"sync"
	"time"
)

// BetaPrior holds the parameters of a Beta distribution.
type BetaPrior struct {
	Alpha        float64
	Beta         float64
	Observations int
	UpdatedAt    time.Time
}

// Agreement returns the posterior mean P(agreement) = alpha / (alpha + beta).
func (b *BetaPrior) Agreement() float64 {
	return b.Alpha / (b.Alpha + b.Beta)
}

// Uncertainty returns the standard deviation of the Beta distribution.
func (b *BetaPrior) Uncertainty() float64 {
	a, bt := b.Alpha, b.Beta
	return math.Sqrt(a * bt / ((a + bt) * (a + bt) * (a + bt + 1)))
}

// Update increments alpha (agreement) or beta (disagreement).
func (b *BetaPrior) Update(agreed bool) {
	if agreed {
		b.Alpha++
	} else {
		b.Beta++
	}
	b.Observations++
	b.UpdatedAt = time.Now()
}

// GateKey is the composite key for a Bayesian gate entry.
type GateKey struct {
	TenantID   string
	Category   string
	SourceType string
}

// GateEntry wraps a BetaPrior with serialization-friendly fields.
type GateEntry struct {
	Key          GateKey
	Prior        BetaPrior
	LastAccess   time.Time
	Dirty        bool
}

// GateStore defines the persistence interface for gate entries.
type GateStore interface {
	LoadGateEntries(maxEntries int) ([]GateEntry, error)
	FlushGateEntries(entries []GateEntry) error
}

// Config holds tunable parameters for the Bayesian gate.
type Config struct {
	// AgreementThreshold is the minimum P(agreement) to skip LLM (default: 0.75).
	AgreementThreshold float64 `json:"agreement_threshold" yaml:"agreement_threshold"`
	// UncertaintyMax is the maximum uncertainty to skip LLM (default: 0.10).
	UncertaintyMax float64 `json:"uncertainty_max" yaml:"uncertainty_max"`
	// HeuristicConfidence is the minimum heuristic confidence to consider gating (default: 0.8).
	HeuristicConfidence float64 `json:"heuristic_confidence" yaml:"heuristic_confidence"`
	// HierarchicalMinObs is the minimum observations before trusting specific prior (default: 20).
	HierarchicalMinObs int `json:"hierarchical_min_obs" yaml:"hierarchical_min_obs"`
	// MaxEntries is the maximum number of gate entries in memory (default: 10000).
	MaxEntries int `json:"max_entries" yaml:"max_entries"`
	// ShadowMode when true evaluates but never skips (default: false).
	ShadowMode bool `json:"shadow_mode" yaml:"shadow_mode"`
}

// DefaultConfig returns a Config with sensible defaults.
func DefaultConfig() Config {
	return Config{
		AgreementThreshold:  0.75,
		UncertaintyMax:      0.10,
		HeuristicConfidence: 0.8,
		HierarchicalMinObs:  20,
		MaxEntries:          10000,
		ShadowMode:          false,
	}
}

// lruEntry wraps a BetaPrior with access tracking for LRU eviction.
type lruEntry struct {
	prior      BetaPrior
	lastAccess time.Time
	dirty      bool
}

// BayesianGate manages Beta-Binomial classification gates.
type BayesianGate struct {
	mu       sync.RWMutex
	entries  map[GateKey]*lruEntry
	config   Config
	store    GateStore // optional persistence

	// Hierarchical prior: pooled per (tenant_id, category)
	pooled   map[string]*BetaPrior // key: "tenant_id:category"
	pooledMu sync.RWMutex

	// Metrics callbacks (optional)
	OnSkip       func()
	OnEvaluation func()
	OnShadowSkip func()
}

// New creates a new BayesianGate with the given config and optional store.
func New(cfg Config, store GateStore) *BayesianGate {
	if cfg.MaxEntries == 0 {
		cfg.MaxEntries = 10000
	}
	if cfg.AgreementThreshold == 0 {
		cfg.AgreementThreshold = 0.75
	}
	if cfg.UncertaintyMax == 0 {
		cfg.UncertaintyMax = 0.10
	}
	if cfg.HeuristicConfidence == 0 {
		cfg.HeuristicConfidence = 0.8
	}
	if cfg.HierarchicalMinObs == 0 {
		cfg.HierarchicalMinObs = 20
	}

	return &BayesianGate{
		entries: make(map[GateKey]*lruEntry),
		config:  cfg,
		store:   store,
		pooled:  make(map[string]*BetaPrior),
	}
}

// ShouldSkipLLM returns true if the gate recommends skipping the LLM call.
func (g *BayesianGate) ShouldSkipLLM(tenantID, category, sourceType string, heuristicConfidence float64) bool {
	if g.OnEvaluation != nil {
		g.OnEvaluation()
	}

	if heuristicConfidence < g.config.HeuristicConfidence {
		return false
	}

	prior := g.getPrior(tenantID, category, sourceType)
	agreement := prior.Agreement()
	uncertainty := prior.Uncertainty()

	shouldSkip := agreement >= g.config.AgreementThreshold && uncertainty < g.config.UncertaintyMax

	if shouldSkip && g.config.ShadowMode {
		if g.OnShadowSkip != nil {
			g.OnShadowSkip()
		}
		return false
	}

	if shouldSkip {
		if g.OnSkip != nil {
			g.OnSkip()
		}
	}

	return shouldSkip
}

// getPrior retrieves the Beta prior for a (tenant, category, sourceType) triple.
// Falls back to the hierarchical pooled prior if observations are insufficient.
func (g *BayesianGate) getPrior(tenantID, category, sourceType string) BetaPrior {
	key := GateKey{TenantID: tenantID, Category: category, SourceType: sourceType}

	g.mu.Lock()
	entry, exists := g.entries[key]
	if exists && entry.prior.Observations >= g.config.HierarchicalMinObs {
		entry.lastAccess = time.Now()
		priorCopy := entry.prior // copy under lock before releasing
		g.mu.Unlock()
		return priorCopy
	}
	// Copy prior fields under lock so reads below are race-free.
	var priorCopy BetaPrior
	if exists {
		priorCopy = entry.prior
	}
	g.mu.Unlock()

	// Hierarchical fallback: blend specific and pooled priors
	if exists && priorCopy.Observations > 0 {
		pooled := g.getPooledPrior(tenantID, category)
		weight := float64(priorCopy.Observations) / float64(g.config.HierarchicalMinObs)
		return BetaPrior{
			Alpha:        weight*priorCopy.Alpha + (1-weight)*pooled.Alpha,
			Beta:         weight*priorCopy.Beta + (1-weight)*pooled.Beta,
			Observations: priorCopy.Observations,
		}
	}

	// No observations: return pooled or uninformative prior
	pooled := g.getPooledPrior(tenantID, category)
	return *pooled
}

func (g *BayesianGate) getPooledPrior(tenantID, category string) *BetaPrior {
	poolKey := tenantID + ":" + category

	g.pooledMu.RLock()
	p, exists := g.pooled[poolKey]
	g.pooledMu.RUnlock()

	if exists {
		return p
	}

	return &BetaPrior{Alpha: 2.0, Beta: 2.0}
}

// Update records an agreement/disagreement observation.
func (g *BayesianGate) Update(tenantID, category, sourceType string, agreed bool) {
	key := GateKey{TenantID: tenantID, Category: category, SourceType: sourceType}

	g.mu.Lock()
	defer g.mu.Unlock()

	entry, exists := g.entries[key]
	if !exists {
		entry = &lruEntry{
			prior: BetaPrior{Alpha: 2.0, Beta: 2.0},
		}
		g.entries[key] = entry
		g.evictIfNeeded()
	}

	entry.prior.Update(agreed)
	entry.lastAccess = time.Now()
	entry.dirty = true

	g.updatePooledPrior(tenantID, category, agreed)
}

func (g *BayesianGate) updatePooledPrior(tenantID, category string, agreed bool) {
	poolKey := tenantID + ":" + category

	g.pooledMu.Lock()
	defer g.pooledMu.Unlock()

	p, exists := g.pooled[poolKey]
	if !exists {
		p = &BetaPrior{Alpha: 2.0, Beta: 2.0}
		g.pooled[poolKey] = p
		// Evict if over capacity
		if len(g.pooled) > g.config.MaxEntries {
			g.evictOldestPooled()
		}
	}
	p.Update(agreed)
}

// evictOldestPooled removes the pooled entry with the fewest observations. Must hold g.pooledMu.
func (g *BayesianGate) evictOldestPooled() {
	var oldestKey string
	oldestObs := int(^uint(0) >> 1) // max int
	for k, v := range g.pooled {
		if v.Observations < oldestObs {
			oldestKey = k
			oldestObs = v.Observations
		}
	}
	if oldestKey != "" {
		delete(g.pooled, oldestKey)
	}
}

// evictIfNeeded removes the LRU entry. Must hold g.mu.
func (g *BayesianGate) evictIfNeeded() {
	if len(g.entries) <= g.config.MaxEntries {
		return
	}

	var oldestKey GateKey
	var oldestTime time.Time
	first := true

	for k, v := range g.entries {
		if first || v.lastAccess.Before(oldestTime) {
			oldestKey = k
			oldestTime = v.lastAccess
			first = false
		}
	}

	if entry, ok := g.entries[oldestKey]; ok && entry.dirty && g.store != nil {
		go g.flushSingle(oldestKey, entry.prior)
	}

	delete(g.entries, oldestKey)
}

// Load reads gate entries from the store into memory.
func (g *BayesianGate) Load() error {
	if g.store == nil {
		return nil
	}

	entries, err := g.store.LoadGateEntries(g.config.MaxEntries)
	if err != nil {
		return err
	}

	g.mu.Lock()
	defer g.mu.Unlock()

	for _, e := range entries {
		g.entries[e.Key] = &lruEntry{
			prior:      e.Prior,
			lastAccess: time.Now(),
		}

		poolKey := e.Key.TenantID + ":" + e.Key.Category
		g.pooledMu.Lock()
		p, exists := g.pooled[poolKey]
		if !exists {
			p = &BetaPrior{Alpha: 2.0, Beta: 2.0}
			g.pooled[poolKey] = p
		}
		p.Alpha += (e.Prior.Alpha - 2.0) * float64(e.Prior.Observations) / 100.0
		p.Beta += (e.Prior.Beta - 2.0) * float64(e.Prior.Observations) / 100.0
		if p.Alpha < 1 {
			p.Alpha = 1
		}
		if p.Beta < 1 {
			p.Beta = 1
		}
		g.pooledMu.Unlock()
	}

	return nil
}

// Flush writes all dirty entries to the store.
func (g *BayesianGate) Flush() error {
	if g.store == nil {
		return nil
	}

	g.mu.Lock()
	var toFlush []GateEntry
	for k, v := range g.entries {
		if v.dirty {
			toFlush = append(toFlush, GateEntry{
				Key:   k,
				Prior: v.prior,
			})
			v.dirty = false
		}
	}
	g.mu.Unlock()

	if len(toFlush) == 0 {
		return nil
	}

	return g.store.FlushGateEntries(toFlush)
}

func (g *BayesianGate) flushSingle(key GateKey, prior BetaPrior) {
	if g.store == nil {
		return
	}
	_ = g.store.FlushGateEntries([]GateEntry{{Key: key, Prior: prior}})
}

// EntryCount returns the number of entries in the gate.
func (g *BayesianGate) EntryCount() int {
	g.mu.RLock()
	defer g.mu.RUnlock()
	return len(g.entries)
}

// GetAgreement returns the current agreement probability for a triple.
func (g *BayesianGate) GetAgreement(tenantID, category, sourceType string) float64 {
	prior := g.getPrior(tenantID, category, sourceType)
	return prior.Agreement()
}
