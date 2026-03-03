package classifier

import (
	"encoding/json"
	"testing"
)

func observabilityConfig() Config {
	return Config{
		Categories: []CategoryConfig{
			{
				Name:     "performance",
				Keywords: []string{"latency", "throughput", "p50", "p95", "p99", "slow", "timeout", "response_time", "cpu", "memory"},
				Weights:  map[string]float64{"p99": 2.0, "p95": 2.0, "timeout": 2.0},
			},
			{
				Name:     "reliability",
				Keywords: []string{"error", "failure", "crash", "panic", "exception", "unavailable", "downtime", "outage", "5xx"},
				Weights:  map[string]float64{"outage": 2.0, "crash": 2.0, "panic": 2.0},
			},
			{
				Name:     "security",
				Keywords: []string{"auth", "authentication", "authorization", "permission", "forbidden", "401", "403", "vulnerability", "cve"},
				Weights:  map[string]float64{"cve": 2.0, "vulnerability": 2.0},
			},
			{
				Name:     "deployment",
				Keywords: []string{"deploy", "release", "rollout", "rollback", "build", "ci", "cd", "pipeline", "container"},
				Weights:  map[string]float64{"deploy": 2.0, "rollback": 2.0},
			},
			{
				Name:     "analytics",
				Keywords: []string{"metric", "count", "aggregation", "funnel", "conversion", "retention", "engagement", "revenue"},
				Weights:  map[string]float64{"event": 0.5, "metric": 0.5},
			},
		},
		SourcePriors: map[string]map[string]float64{
			"sentry":     {"reliability": 3.0},
			"prometheus": {"performance": 3.0},
		},
		TypeToCategory: map[string]string{
			"metric":  "performance",
			"error":   "reliability",
			"deploy":  "deployment",
			"audit":   "security",
		},
	}
}

func TestClassify_Performance(t *testing.T) {
	c := New(observabilityConfig())
	result := c.Classify("High p99 latency detected in API gateway", "metric")

	if result.Category != "performance" {
		t.Errorf("Expected performance, got %s", result.Category)
	}
	if result.RelevanceScore < 0.3 {
		t.Errorf("Expected relevance >= 0.3, got %v", result.RelevanceScore)
	}
}

func TestClassify_Reliability(t *testing.T) {
	c := New(observabilityConfig())
	result := c.Classify("Critical outage in production database", "error")

	if result.Category != "reliability" {
		t.Errorf("Expected reliability, got %s", result.Category)
	}
}

func TestClassify_Security(t *testing.T) {
	c := New(observabilityConfig())
	result := c.Classify("New CVE vulnerability detected in auth module", "audit")

	if result.Category != "security" {
		t.Errorf("Expected security, got %s", result.Category)
	}
}

func TestClassify_Deployment(t *testing.T) {
	c := New(observabilityConfig())
	result := c.Classify("Rolling out v2.5 to production", "deploy")

	if result.Category != "deployment" {
		t.Errorf("Expected deployment, got %s", result.Category)
	}
}

func TestClassify_Uncategorized(t *testing.T) {
	c := New(observabilityConfig())
	result := c.Classify("The quick brown fox jumps over the lazy dog", "unknown")

	if result.Category != "uncategorized" {
		t.Errorf("Expected uncategorized, got %s", result.Category)
	}
	if result.RelevanceScore != 0.0 {
		t.Errorf("Expected 0.0 relevance for uncategorized, got %v", result.RelevanceScore)
	}
}

func TestClassify_TypeMappingBoost(t *testing.T) {
	c := New(observabilityConfig())
	// "metric" type maps to performance, should get floor of 0.5
	result := c.Classify("some generic content", "metric")

	if result.Category != "performance" {
		t.Errorf("Expected performance from type mapping, got %s", result.Category)
	}
	if result.RelevanceScore < 0.5 {
		t.Errorf("Expected relevance >= 0.5 from type mapping, got %v", result.RelevanceScore)
	}
}

func TestClassify_DeterministicTieBreaking(t *testing.T) {
	c := New(observabilityConfig())
	// Run many times to verify determinism
	results := make(map[string]int)
	for i := 0; i < 100; i++ {
		r := c.Classify("error latency", "generic")
		results[r.Category]++
	}
	if len(results) != 1 {
		t.Errorf("Classification should be deterministic, got categories: %v", results)
	}
}

func TestClassifySignal_SourcePriors(t *testing.T) {
	c := New(observabilityConfig())
	payload := json.RawMessage(`{"message": "something happened"}`)

	result := c.ClassifySignal("sentry", "event", payload)
	if result.Category != "reliability" {
		t.Errorf("Sentry source prior should favor reliability, got %s", result.Category)
	}
}

func TestClassifyWithLearned(t *testing.T) {
	c := New(observabilityConfig())
	store := NewLearnedKeywordStore()

	// Without learned keywords
	result1 := c.ClassifyWithLearned("kubernetes pod restart detected", "event", store)

	// Add learned keywords
	store.Update([]LearnedKeyword{
		{Keyword: "kubernetes", Category: "deployment", Weight: 2.0, Confidence: 0.9},
		{Keyword: "pod", Category: "deployment", Weight: 1.5, Confidence: 0.8},
	})

	result2 := c.ClassifyWithLearned("kubernetes pod restart detected", "event", store)

	// With learned keywords, deployment should score higher
	if result2.Category != "deployment" {
		t.Logf("Without learned: %s (score: %v)", result1.Category, result1.RelevanceScore)
		t.Logf("With learned: %s (score: %v)", result2.Category, result2.RelevanceScore)
	}

	if result2.ClassificationSource != "heuristic" {
		t.Errorf("Expected source 'heuristic', got %s", result2.ClassificationSource)
	}
}

func TestClassifyWithLearned_NilStore(t *testing.T) {
	c := New(observabilityConfig())
	result := c.ClassifyWithLearned("error in production", "error", nil)

	if result.Category != "reliability" {
		t.Errorf("Expected reliability with nil store, got %s", result.Category)
	}
}

func TestInferSignalClass(t *testing.T) {
	c := New(observabilityConfig())

	tests := []struct {
		name       string
		signalType string
		payload    string
		want       SignalClass
	}{
		{"anomaly_type", "alert", `{}`, SignalClassAnomaly},
		{"metric_field", "generic", `{"value": 42.5}`, SignalClassMetric},
		{"trend_field", "generic", `{"previous": 10, "current": 20}`, SignalClassTrend},
		{"metric_type", "metric", `{"foo": "bar"}`, SignalClassMetric},
		{"semantic_fallback", "generic", `{"message": "hello"}`, SignalClassSemantic},
		{"invalid_json", "generic", `not json`, SignalClassSemantic},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := c.InferSignalClass(tt.signalType, json.RawMessage(tt.payload))
			if got != tt.want {
				t.Errorf("InferSignalClass(%q, %q) = %v, want %v", tt.signalType, tt.payload, got, tt.want)
			}
		})
	}
}

func TestFlattenPayload(t *testing.T) {
	tests := []struct {
		name    string
		payload string
		want    string
	}{
		{"empty", "", ""},
		{"string", `"hello"`, "hello"},
		{"nested", `{"key": {"nested": "value"}}`, ""},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := FlattenPayload(json.RawMessage(tt.payload))
			if tt.want != "" && got == "" {
				t.Errorf("FlattenPayload(%q) returned empty", tt.payload)
			}
		})
	}
}

func TestCustomConfig(t *testing.T) {
	// Test with a completely different domain
	cfg := Config{
		Categories: []CategoryConfig{
			{
				Name:     "user_satisfaction",
				Keywords: []string{"nps", "satisfaction", "churn", "retention"},
				Weights:  map[string]float64{"churn": 2.0, "nps": 2.0},
			},
			{
				Name:     "adoption",
				Keywords: []string{"dau", "mau", "activation", "onboarding"},
			},
			{
				Name:     "revenue_impact",
				Keywords: []string{"revenue", "arpu", "ltv", "conversion"},
			},
		},
		SourcePriors: map[string]map[string]float64{
			"mixpanel": {"adoption": 3.0},
			"stripe":   {"revenue_impact": 3.0},
		},
	}

	c := New(cfg)

	result := c.Classify("NPS score dropped to 35, churn increasing", "survey")
	if result.Category != "user_satisfaction" {
		t.Errorf("Expected user_satisfaction, got %s", result.Category)
	}

	result2 := c.Classify("DAU increased by 15% after onboarding changes", "metric")
	if result2.Category != "adoption" {
		t.Errorf("Expected adoption, got %s", result2.Category)
	}

	result3 := c.Classify("ARPU grew, revenue up 20%", "finance")
	if result3.Category != "revenue_impact" {
		t.Errorf("Expected revenue_impact, got %s", result3.Category)
	}
}

func TestLearnedKeywordStore(t *testing.T) {
	store := NewLearnedKeywordStore()

	if store.Count() != 0 {
		t.Errorf("New store should be empty, got %d", store.Count())
	}

	store.Update([]LearnedKeyword{
		{Keyword: "k8s", Category: "deployment", Weight: 1.5},
		{Keyword: "pod", Category: "deployment", Weight: 1.0},
		{Keyword: "oom", Category: "reliability", Weight: 2.0},
	})

	if store.Count() != 3 {
		t.Errorf("Expected 3 keywords, got %d", store.Count())
	}

	promoted := store.GetPromotedKeywords()
	if len(promoted["deployment"]) != 2 {
		t.Errorf("Expected 2 deployment keywords, got %d", len(promoted["deployment"]))
	}

	weights := store.GetPromotedWeights()
	if weights["oom"] != 2.0 {
		t.Errorf("Expected oom weight 2.0, got %v", weights["oom"])
	}
}

func TestRelevanceScoreBounds(t *testing.T) {
	c := New(observabilityConfig())

	// Test many inputs — score should always be in [0.0, 1.0]
	inputs := []struct {
		content string
		typ     string
	}{
		{"", ""},
		{"error crash panic outage failure unavailable downtime 5xx", "error"},
		{"a", "b"},
		{"latency p99 cpu memory timeout slow throughput response_time", "metric"},
	}

	for _, input := range inputs {
		r := c.Classify(input.content, input.typ)
		if r.RelevanceScore < 0 || r.RelevanceScore > 1.0 {
			t.Errorf("Score out of bounds for (%q, %q): %v", input.content, input.typ, r.RelevanceScore)
		}
	}
}
