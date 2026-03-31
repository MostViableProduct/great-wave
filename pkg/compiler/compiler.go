package compiler

import (
	"context"
	"encoding/json"
	"log"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/MostViableProduct/great-wave/pkg/classifier"
	"github.com/MostViableProduct/great-wave/pkg/gate"
	"github.com/MostViableProduct/great-wave/pkg/health"
	"github.com/MostViableProduct/great-wave/pkg/keywords"
)

// Config holds all compiler configuration.
type Config struct {
	// Classifier configures the heuristic cascade classifier.
	Classifier classifier.Config `json:"classifier" yaml:"classifier"`
	// Gate configures the Bayesian gate for LLM skip decisions.
	Gate gate.Config `json:"gate" yaml:"gate"`
	// Health configures the per-entity health model.
	Health health.Config `json:"health" yaml:"health"`
	// Keywords configures the self-improving keyword learning loop.
	Keywords keywords.Config `json:"keywords" yaml:"keywords"`
	// MaxVectorResults is the maximum entity matches from vector search (default: 5).
	MaxVectorResults int `json:"max_vector_results" yaml:"max_vector_results"`
	// LLMMinConfidence is the minimum LLM confidence to accept (default: 0.3).
	LLMMinConfidence float64 `json:"llm_min_confidence" yaml:"llm_min_confidence"`
}

// DefaultConfig returns a Config with sensible defaults.
func DefaultConfig() Config {
	return Config{
		Gate:             gate.DefaultConfig(),
		Health:           health.DefaultConfig(),
		Keywords:         keywords.DefaultConfig(),
		MaxVectorResults: 5,
		LLMMinConfidence: 0.3,
	}
}

// Callbacks provides optional hooks for observability. All fields are optional.
type Callbacks struct {
	// OnClassify is called after every classification with source and category.
	OnClassify func(source, category string)
	// OnGateSkip is called when the Bayesian gate skips an LLM call.
	OnGateSkip func()
	// OnLLMFallback is called when LLM fails and heuristic is used as fallback.
	OnLLMFallback func(err error)
	// OnAgreement is called after comparing heuristic and LLM results.
	OnAgreement func(agreed bool)
	// OnKeywordsPromoted is called after keyword promotion with the new count.
	OnKeywordsPromoted func(count int)
}

// Compiler is the top-level orchestrator for the contextual compilation pipeline.
type Compiler struct {
	config     Config
	deps       Deps
	callbacks  Callbacks
	classifier *classifier.Classifier
	gate       *gate.BayesianGate
	health     *health.Model
	learner    *keywords.Learner
	runtime    *classifier.LearnedKeywordStore

	mu sync.RWMutex
}

// New creates a Compiler from configuration and dependencies.
func New(cfg Config, deps Deps, cbs Callbacks) *Compiler {
	if cfg.MaxVectorResults == 0 {
		cfg.MaxVectorResults = 5
	}
	if cfg.LLMMinConfidence == 0 {
		cfg.LLMMinConfidence = 0.3
	}

	cls := classifier.New(cfg.Classifier)
	runtime := classifier.NewLearnedKeywordStore()

	// Wire gate callbacks into compiler callbacks
	gateCfg := cfg.Gate
	g := gate.New(gateCfg, deps.GateStore)
	if cbs.OnGateSkip != nil {
		g.OnSkip = cbs.OnGateSkip
	}

	h := health.NewModel(cfg.Health, deps.HealthStore)
	l := keywords.NewLearner(cfg.Keywords, deps.KeywordStore, cls, runtime)

	return &Compiler{
		config:     cfg,
		deps:       deps,
		callbacks:  cbs,
		classifier: cls,
		gate:       g,
		health:     h,
		learner:    l,
		runtime:    runtime,
	}
}

// Classify runs the full cascade classification pipeline on a signal:
//  1. Heuristic classification (always runs — fast baseline)
//  2. Bayesian gate check (skip LLM if heuristic is reliable)
//  3. LLM classification (if gate says heuristic is unreliable)
//  4. Async keyword learning + gate update on LLM result
//  5. Optional vector search for entity resolution
func (c *Compiler) Classify(ctx context.Context, signal Signal) (*ClassifyResult, error) {
	if strings.TrimSpace(signal.Content) == "" {
		return nil, ErrEmptyContent
	}

	// Step 1: Always run heuristic (fast baseline)
	heuristicResult := c.classifier.ClassifyWithLearned(signal.Content, signal.Type, c.runtime)

	result := &ClassifyResult{
		Category:             heuristicResult.Category,
		RelevanceScore:       heuristicResult.RelevanceScore,
		ClassificationSource: SourceHeuristic,
		Confidence:           heuristicResult.Confidence,
	}

	// Infer signal class from payload
	if len(signal.Payload) > 0 {
		result.SignalClass = c.classifier.InferSignalClass(signal.Type, signal.Payload)
	} else {
		result.SignalClass = classifier.SignalClassSemantic
	}

	// Step 2: Check Bayesian gate
	if c.gate.ShouldSkipLLM(signal.TenantID, heuristicResult.Category, signal.Type, heuristicResult.Confidence) {
		result.ClassificationSource = SourceHeuristicGated
		c.notifyClassify(result.ClassificationSource, result.Category)
		c.resolveEntities(ctx, signal, result)
		return result, nil
	}

	// Step 3: Try LLM classification (if adapter is available)
	if c.deps.LLM != nil {
		categories := c.categoryNames()
		llmResult, err := c.deps.LLM.Classify(ctx, signal.Content, signal.Type, categories)
		if err == nil && llmResult.Confidence >= c.config.LLMMinConfidence {
			// Validate category
			if c.classifier.ValidCategories()[llmResult.Category] {
				result.Category = llmResult.Category
				result.Confidence = llmResult.Confidence
				result.ClassificationSource = SourceLLM

				// Track agreement
				agreed := llmResult.Category == heuristicResult.Category
				if c.callbacks.OnAgreement != nil {
					c.callbacks.OnAgreement(agreed)
				}

				// Async: keyword learning + gate update.
				// Intentionally detached from request context — background
				// work must complete even after the HTTP response is sent.
				go c.recordComparison(signal, heuristicResult, llmResult, agreed) //#nosec G118
			}
		} else if err != nil {
			if c.callbacks.OnLLMFallback != nil {
				c.callbacks.OnLLMFallback(err)
			}
		}
	}

	c.notifyClassify(result.ClassificationSource, result.Category)
	c.resolveEntities(ctx, signal, result)
	return result, nil
}

// ClassifySignal is a convenience method that flattens a JSON payload into
// searchable content before classifying.
func (c *Compiler) ClassifySignal(ctx context.Context, tenantID, source, signalType string, payload json.RawMessage) (*ClassifyResult, error) {
	content := classifier.FlattenPayload(payload)
	return c.Classify(ctx, Signal{
		TenantID: tenantID,
		Source:   source,
		Type:     signalType,
		Content:  content,
		Payload:  payload,
	})
}

// ScoreHealth computes the Bayesian health score for an entity.
func (c *Compiler) ScoreHealth(tenantID, entityID string) *HealthResult {
	score := c.health.Score(tenantID, entityID)
	return &HealthResult{
		EntityID:                entityID,
		Score:                   score.Score,
		ConfidenceIntervalLower: score.ConfidenceIntervalLower,
		ConfidenceIntervalUpper: score.ConfidenceIntervalUpper,
	}
}

// RecordHealthEvent records an event that affects an entity's health score.
func (c *Compiler) RecordHealthEvent(tenantID, entityID, severity, category string, confidence float64) {
	c.health.UpdateFromEvent(tenantID, entityID, severity, category, confidence)
}

// PromoteKeywords runs the keyword promotion cycle: high-confidence learned
// keywords are promoted to the heuristic classifier's runtime store.
func (c *Compiler) PromoteKeywords() (int, error) {
	count, err := c.learner.Promote()
	if err != nil {
		return 0, err
	}
	if c.callbacks.OnKeywordsPromoted != nil {
		c.callbacks.OnKeywordsPromoted(count)
	}
	return count, nil
}

// RecordFalsePositives weakens the given keywords by decrementing their
// observation counts. This records a negative signal from dismissed review
// findings, making the keywords more likely to be demoted in the next
// promotion cycle.
func (c *Compiler) RecordFalsePositives(keywords []string) error {
	for _, kw := range keywords {
		if err := c.learner.WeakenKeyword(kw); err != nil {
			return err
		}
	}
	return nil
}

// LoadState loads persisted state (gate entries, health priors, promoted keywords)
// from the storage adapters. Call this once at startup.
func (c *Compiler) LoadState() error {
	if err := c.gate.Load(); err != nil {
		return err
	}
	if err := c.health.Load(); err != nil {
		return err
	}
	if _, err := c.learner.LoadPromoted(); err != nil {
		return err
	}
	return nil
}

// FlushState persists current state (gate entries, health priors) to the
// storage adapters. Call this periodically or before shutdown.
func (c *Compiler) FlushState() error {
	if err := c.gate.Flush(); err != nil {
		return err
	}
	return c.health.Flush()
}

// Gate returns the underlying Bayesian gate for direct access if needed.
func (c *Compiler) Gate() *gate.BayesianGate {
	return c.gate
}

// HealthModel returns the underlying health model for direct access.
func (c *Compiler) HealthModel() *health.Model {
	return c.health
}

// Runtime returns the learned keyword store for inspection.
func (c *Compiler) Runtime() *classifier.LearnedKeywordStore {
	return c.runtime
}

// HasLLM reports whether an LLM classifier adapter is configured.
func (c *Compiler) HasLLM() bool { return c.deps.LLM != nil }

// HasVector reports whether a vector search adapter is configured.
func (c *Compiler) HasVector() bool { return c.deps.Vector != nil }

// HasEvents reports whether an event sink adapter is configured.
func (c *Compiler) HasEvents() bool { return c.deps.Events != nil }

// HasStorage reports whether any storage adapter is configured.
func (c *Compiler) HasStorage() bool {
	return c.deps.GateStore != nil || c.deps.HealthStore != nil || c.deps.KeywordStore != nil
}

// ValidSeverities returns the set of configured health severity names.
func (c *Compiler) ValidSeverities() map[string]bool {
	return c.health.ValidSeverities()
}

// --- internal helpers ---

func (c *Compiler) categoryNames() []string {
	cats := c.classifier.ValidCategories()
	names := make([]string, 0, len(cats))
	for name := range cats {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

func (c *Compiler) notifyClassify(source, category string) {
	if c.callbacks.OnClassify != nil {
		c.callbacks.OnClassify(source, category)
	}
}

func (c *Compiler) resolveEntities(ctx context.Context, signal Signal, result *ClassifyResult) {
	if c.deps.Vector == nil {
		return
	}
	matches, err := c.deps.Vector.Search(ctx, signal.Content, c.config.MaxVectorResults)
	if err != nil {
		return
	}
	for _, m := range matches {
		result.RelatedEntities = append(result.RelatedEntities, m.ID)
	}
}

func (c *Compiler) recordComparison(signal Signal, heuristic classifier.Result, llm *LLMResult, agreed bool) {
	// Update Bayesian gate with agreement observation
	c.gate.Update(signal.TenantID, heuristic.Category, signal.Type, agreed)

	// If they disagreed, extract and record novel keywords
	if !agreed {
		c.learner.RecordDisagreement(signal.Content, llm.Category, llm.Confidence)
	}

	// Emit event if sink is configured
	if c.deps.Events != nil {
		evt := map[string]interface{}{
			"tenant_id":  signal.TenantID,
			"source":     signal.Source,
			"type":       signal.Type,
			"heuristic":  heuristic.Category,
			"llm":        llm.Category,
			"agreed":     agreed,
			"confidence": llm.Confidence,
			"timestamp":  time.Now().UTC().Format(time.RFC3339),
		}
		payload, err := json.Marshal(evt)
		if err != nil {
			log.Printf("compiler: failed to marshal classification event: %v", err)
			return
		}
		if err := c.deps.Events.Emit(context.Background(), "classification.compared", payload); err != nil {
			log.Printf("compiler: failed to emit classification event: %v", err)
		}
	}
}
