package classifier

import "testing"

func benchConfig() Config {
	return Config{
		Categories: []CategoryConfig{
			{Name: "performance", Keywords: []string{"latency", "throughput", "p50", "p95", "p99", "slow", "timeout", "response_time", "cpu", "memory"}, Weights: map[string]float64{"p99": 2.0}},
			{Name: "reliability", Keywords: []string{"error", "failure", "crash", "panic", "exception", "unavailable", "downtime", "outage", "5xx"}, Weights: map[string]float64{"outage": 2.0}},
			{Name: "security", Keywords: []string{"auth", "authentication", "authorization", "permission", "forbidden", "vulnerability", "cve"}, Weights: map[string]float64{"cve": 2.0}},
			{Name: "deployment", Keywords: []string{"deploy", "release", "rollout", "rollback", "build", "ci", "cd", "pipeline", "container"}, Weights: map[string]float64{"deploy": 2.0}},
		},
		TypeToCategory: map[string]string{"metric": "performance", "error": "reliability"},
	}
}

func BenchmarkClassify(b *testing.B) {
	c := New(benchConfig())
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		c.Classify("High p99 latency detected in API gateway causing timeouts", "metric")
	}
}

func BenchmarkClassifyWithLearned(b *testing.B) {
	c := New(benchConfig())
	store := NewLearnedKeywordStore()
	store.Update([]LearnedKeyword{
		{Keyword: "api_gateway", Category: "performance", Weight: 0.9, Confidence: 0.9},
		{Keyword: "database_pool", Category: "reliability", Weight: 0.85, Confidence: 0.85},
	})
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		c.ClassifyWithLearned("High p99 latency detected in API gateway causing timeouts", "metric", store)
	}
}
