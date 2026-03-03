// Package classifier provides configurable heuristic signal classification.
//
// It classifies signals into categories using keyword matching, source priors,
// direct type-to-category mapping, and learned keywords. All domain knowledge
// (categories, keywords, priors) is configuration-driven rather than hard-coded.
package classifier

import (
	"encoding/json"
	"fmt"
	"strings"
)

// Result holds the output of a classification operation.
type Result struct {
	Category             string  `json:"category"`
	RelevanceScore       float64 `json:"relevance_score"`
	ClassificationSource string  `json:"classification_source,omitempty"`
	Confidence           float64 `json:"confidence,omitempty"`
}

// CategoryConfig defines a single classification category.
type CategoryConfig struct {
	Name     string             `json:"name" yaml:"name"`
	Keywords []string           `json:"keywords" yaml:"keywords"`
	Weights  map[string]float64 `json:"weights,omitempty" yaml:"weights,omitempty"`
}

// Config holds all the classification configuration.
type Config struct {
	// Categories maps category names to their keyword configuration.
	Categories []CategoryConfig `json:"categories" yaml:"categories"`
	// SourcePriors provides additive score boosts based on signal source.
	SourcePriors map[string]map[string]float64 `json:"source_priors,omitempty" yaml:"source_priors,omitempty"`
	// TypeToCategory provides direct mapping from event types to categories.
	TypeToCategory map[string]string `json:"type_to_category,omitempty" yaml:"type_to_category,omitempty"`
	// SignalClassConfig configures signal class inference.
	SignalClassConfig *SignalClassConfig `json:"signal_class_config,omitempty" yaml:"signal_class_config,omitempty"`
}

// SignalClassConfig configures how signals are classified into evaluation paths.
type SignalClassConfig struct {
	MetricFields   []string `json:"metric_fields,omitempty" yaml:"metric_fields,omitempty"`
	TrendFields    []string `json:"trend_fields,omitempty" yaml:"trend_fields,omitempty"`
	AnomalyTypes   []string `json:"anomaly_types,omitempty" yaml:"anomaly_types,omitempty"`
	MetricTypes    []string `json:"metric_types,omitempty" yaml:"metric_types,omitempty"`
}

// Classifier is the configured heuristic classifier.
type Classifier struct {
	categoryKeywords map[string][]string
	keywordWeights   map[string]float64
	sourcePriors     map[string]map[string]float64
	typeToCategory   map[string]string
	validCategories  map[string]bool

	// Signal class fields
	numericFields  map[string]bool
	trendFields    map[string]bool
	anomalyTypes   map[string]bool
	metricTypes    map[string]bool
}

// New creates a Classifier from configuration.
func New(cfg Config) *Classifier {
	c := &Classifier{
		categoryKeywords: make(map[string][]string),
		keywordWeights:   make(map[string]float64),
		sourcePriors:     cfg.SourcePriors,
		typeToCategory:   cfg.TypeToCategory,
		validCategories:  make(map[string]bool),
	}

	if c.sourcePriors == nil {
		c.sourcePriors = make(map[string]map[string]float64)
	}
	if c.typeToCategory == nil {
		c.typeToCategory = make(map[string]string)
	}

	for _, cat := range cfg.Categories {
		c.categoryKeywords[cat.Name] = cat.Keywords
		c.validCategories[cat.Name] = true
		for kw, w := range cat.Weights {
			c.keywordWeights[kw] = w
		}
	}

	// Signal class config
	c.numericFields = defaultNumericFields()
	c.trendFields = defaultTrendFields()
	c.anomalyTypes = defaultAnomalyTypes()
	c.metricTypes = defaultMetricTypes()

	if cfg.SignalClassConfig != nil {
		if len(cfg.SignalClassConfig.MetricFields) > 0 {
			c.numericFields = toSet(cfg.SignalClassConfig.MetricFields)
		}
		if len(cfg.SignalClassConfig.TrendFields) > 0 {
			c.trendFields = toSet(cfg.SignalClassConfig.TrendFields)
		}
		if len(cfg.SignalClassConfig.AnomalyTypes) > 0 {
			c.anomalyTypes = toSet(cfg.SignalClassConfig.AnomalyTypes)
		}
		if len(cfg.SignalClassConfig.MetricTypes) > 0 {
			c.metricTypes = toSet(cfg.SignalClassConfig.MetricTypes)
		}
	}

	return c
}

func toSet(items []string) map[string]bool {
	s := make(map[string]bool, len(items))
	for _, item := range items {
		s[item] = true
	}
	return s
}

// keywordWeight returns the weight for a keyword. Defaults to 1.0.
func (c *Classifier) keywordWeight(kw string) float64 {
	if w, ok := c.keywordWeights[kw]; ok {
		return w
	}
	return 1.0
}

// Classify performs heuristic classification on the given content and signal type.
func (c *Classifier) Classify(searchableContent, signalType string) Result {
	contentLower := strings.ToLower(searchableContent)
	typeLower := strings.ToLower(signalType)

	categoryScores := map[string]float64{}
	totalWeight := 0.0

	for category, keywords := range c.categoryKeywords {
		for _, kw := range keywords {
			if strings.Contains(contentLower, kw) || strings.Contains(typeLower, kw) {
				w := c.keywordWeight(kw)
				categoryScores[category] += w
				totalWeight += w
			}
		}
	}

	bestCategory := ""
	bestScore := 0.0

	if mapped, ok := c.typeToCategory[typeLower]; ok {
		bestCategory = mapped
		bestScore = categoryScores[mapped] + 2.0
		categoryScores[mapped] = bestScore
	}

	for category, score := range categoryScores {
		if score > bestScore || (score == bestScore && bestCategory != "" && category < bestCategory) {
			bestCategory = category
			bestScore = score
		}
	}

	if bestCategory == "" {
		bestCategory = "uncategorized"
	}

	relevanceScore := 0.0
	if totalWeight > 0 {
		relevanceScore = bestScore / (totalWeight + 2.0)
		if relevanceScore > 1.0 {
			relevanceScore = 1.0
		}
		if relevanceScore < 0.3 {
			relevanceScore = 0.3
		}
	}
	if _, ok := c.typeToCategory[typeLower]; ok && relevanceScore < 0.5 {
		relevanceScore = 0.5
	}

	return Result{
		Category:       bestCategory,
		RelevanceScore: relevanceScore,
	}
}

// ClassifySignal flattens a JSON payload, includes the source in searchable content,
// and applies source-aware priors before classifying.
func (c *Classifier) ClassifySignal(source, signalType string, payload json.RawMessage) Result {
	contentLower := strings.ToLower(FlattenPayload(payload))
	sourceLower := strings.ToLower(source)
	typeLower := strings.ToLower(signalType)

	searchable := contentLower + " " + sourceLower + " " + typeLower
	result := c.Classify(searchable, signalType)

	if priors, ok := c.sourcePriors[sourceLower]; ok {
		result = c.classifyWithPriors(searchable, signalType, priors)
	}

	return result
}

func (c *Classifier) classifyWithPriors(searchableContent, signalType string, priors map[string]float64) Result {
	contentLower := strings.ToLower(searchableContent)
	typeLower := strings.ToLower(signalType)

	categoryScores := map[string]float64{}
	totalWeight := 0.0

	for category, keywords := range c.categoryKeywords {
		for _, kw := range keywords {
			if strings.Contains(contentLower, kw) || strings.Contains(typeLower, kw) {
				w := c.keywordWeight(kw)
				categoryScores[category] += w
				totalWeight += w
			}
		}
	}

	for category, prior := range priors {
		categoryScores[category] += prior
	}

	bestCategory := ""
	bestScore := 0.0

	if mapped, ok := c.typeToCategory[typeLower]; ok {
		bestCategory = mapped
		bestScore = categoryScores[mapped] + 2.0
		categoryScores[mapped] = bestScore
	}

	for category, score := range categoryScores {
		if score > bestScore || (score == bestScore && bestCategory != "" && category < bestCategory) {
			bestCategory = category
			bestScore = score
		}
	}

	if bestCategory == "" {
		bestCategory = "uncategorized"
	}

	relevanceScore := 0.0
	if totalWeight > 0 || bestScore > 0 {
		relevanceScore = bestScore / (totalWeight + 2.0)
		if relevanceScore > 1.0 {
			relevanceScore = 1.0
		}
		if relevanceScore < 0.3 {
			relevanceScore = 0.3
		}
	}
	if _, ok := c.typeToCategory[typeLower]; ok && relevanceScore < 0.5 {
		relevanceScore = 0.5
	}

	return Result{
		Category:       bestCategory,
		RelevanceScore: relevanceScore,
	}
}

// ClassifyWithLearned performs classification using both static keywords and
// promoted learned keywords from a LearnedKeywordStore.
func (c *Classifier) ClassifyWithLearned(searchableContent, signalType string, store *LearnedKeywordStore) Result {
	contentLower := strings.ToLower(searchableContent)
	typeLower := strings.ToLower(signalType)

	categoryScores := map[string]float64{}
	totalWeight := 0.0

	for category, keywords := range c.categoryKeywords {
		for _, kw := range keywords {
			if strings.Contains(contentLower, kw) || strings.Contains(typeLower, kw) {
				w := c.keywordWeight(kw)
				categoryScores[category] += w
				totalWeight += w
			}
		}
	}

	if store != nil {
		promotedKeywords := store.GetPromotedKeywords()
		promotedWeights := store.GetPromotedWeights()
		for category, keywords := range promotedKeywords {
			for _, kw := range keywords {
				if strings.Contains(contentLower, kw) || strings.Contains(typeLower, kw) {
					w := 1.0
					if pw, ok := promotedWeights[kw]; ok {
						w = pw
					}
					categoryScores[category] += w
					totalWeight += w
				}
			}
		}
	}

	bestCategory := ""
	bestScore := 0.0

	if mapped, ok := c.typeToCategory[typeLower]; ok {
		bestCategory = mapped
		bestScore = categoryScores[mapped] + 2.0
		categoryScores[mapped] = bestScore
	}

	for category, score := range categoryScores {
		if score > bestScore || (score == bestScore && bestCategory != "" && category < bestCategory) {
			bestCategory = category
			bestScore = score
		}
	}

	if bestCategory == "" {
		bestCategory = "uncategorized"
	}

	relevanceScore := 0.0
	if totalWeight > 0 {
		relevanceScore = bestScore / (totalWeight + 2.0)
		if relevanceScore > 1.0 {
			relevanceScore = 1.0
		}
		if relevanceScore < 0.3 {
			relevanceScore = 0.3
		}
	}
	if _, ok := c.typeToCategory[typeLower]; ok && relevanceScore < 0.5 {
		relevanceScore = 0.5
	}

	return Result{
		Category:             bestCategory,
		RelevanceScore:       relevanceScore,
		ClassificationSource: "heuristic",
	}
}

// ValidCategories returns the set of valid category names.
func (c *Classifier) ValidCategories() map[string]bool {
	return c.validCategories
}

// CategoryKeywords returns the keyword map (for keyword extraction).
func (c *Classifier) CategoryKeywords() map[string][]string {
	return c.categoryKeywords
}

// SignalClass determines the evaluation path in the cascade classifier.
type SignalClass string

const (
	SignalClassMetric   SignalClass = "METRIC"
	SignalClassTrend    SignalClass = "TREND"
	SignalClassAnomaly  SignalClass = "ANOMALY"
	SignalClassSemantic SignalClass = "SEMANTIC"
)

// InferSignalClass inspects the signal type and payload structure to determine
// how the signal should be evaluated.
func (c *Classifier) InferSignalClass(signalType string, payload json.RawMessage) SignalClass {
	typeLower := strings.ToLower(signalType)

	if c.anomalyTypes[typeLower] {
		return SignalClassAnomaly
	}

	var payloadMap map[string]interface{}
	if err := json.Unmarshal(payload, &payloadMap); err != nil {
		return SignalClassSemantic
	}

	for key := range payloadMap {
		if c.trendFields[strings.ToLower(key)] {
			return SignalClassTrend
		}
	}

	for key, val := range payloadMap {
		if c.numericFields[strings.ToLower(key)] {
			switch val.(type) {
			case float64, int, int64:
				return SignalClassMetric
			}
		}
	}

	if c.metricTypes[typeLower] {
		return SignalClassMetric
	}

	return SignalClassSemantic
}

// FlattenPayload converts a JSON payload to a searchable string.
func FlattenPayload(payload json.RawMessage) string {
	if len(payload) == 0 {
		return ""
	}

	var raw interface{}
	if err := json.Unmarshal(payload, &raw); err != nil {
		return string(payload)
	}
	return FlattenToString(raw)
}

const maxFlattenDepth = 20

// FlattenToString converts arbitrary JSON content into a searchable string.
func FlattenToString(v interface{}) string {
	return flattenToString(v, 0)
}

func flattenToString(v interface{}, depth int) string {
	if depth > maxFlattenDepth || v == nil {
		return ""
	}
	switch val := v.(type) {
	case string:
		return val
	case map[string]interface{}:
		parts := make([]string, 0, len(val))
		for k, child := range val {
			parts = append(parts, k+" "+flattenToString(child, depth+1))
		}
		return strings.Join(parts, " ")
	case []interface{}:
		parts := make([]string, 0, len(val))
		for _, child := range val {
			parts = append(parts, flattenToString(child, depth+1))
		}
		return strings.Join(parts, " ")
	default:
		return fmt.Sprintf("%v", val)
	}
}

func defaultNumericFields() map[string]bool {
	return map[string]bool{
		"value": true, "latency": true, "p99": true, "p95": true, "p50": true,
		"duration": true, "response_time": true, "throughput": true, "rps": true,
		"error_rate": true, "uptime": true, "cpu": true, "memory": true,
		"count": true, "rate": true, "avg": true, "max": true, "min": true,
	}
}

func defaultTrendFields() map[string]bool {
	return map[string]bool{
		"previous": true, "current": true, "delta": true, "change": true,
		"before": true, "after": true, "diff": true, "baseline": true,
		"previous_value": true, "current_value": true,
	}
}

func defaultAnomalyTypes() map[string]bool {
	return map[string]bool{
		"alert": true, "incident": true, "error": true, "exception": true,
		"outage": true, "health_check": true, "pagerduty": true,
	}
}

func defaultMetricTypes() map[string]bool {
	return map[string]bool{
		"metric": true, "trace": true, "span": true,
	}
}
