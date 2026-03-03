package api

import (
	"fmt"
	"net/http"
	"sync/atomic"
)

// Metrics tracks operational counters in a lock-free manner.
// Counters are exported in Prometheus text exposition format via GET /metrics.
type Metrics struct {
	ClassifyTotal     atomic.Int64
	ClassifyHeuristic atomic.Int64
	ClassifyLLM       atomic.Int64
	ClassifyGated     atomic.Int64
	GateSkips         atomic.Int64
	LLMFallbacks      atomic.Int64
	HealthQueries     atomic.Int64
	KeywordsPromoted  atomic.Int64
}

func (m *Metrics) handleMetrics(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
	fmt.Fprintf(w, "# HELP compiler_classify_total Total classification requests.\n")
	fmt.Fprintf(w, "# TYPE compiler_classify_total counter\n")
	fmt.Fprintf(w, "compiler_classify_total %d\n", m.ClassifyTotal.Load())
	fmt.Fprintf(w, "# HELP compiler_classify_heuristic Classifications resolved by heuristic.\n")
	fmt.Fprintf(w, "# TYPE compiler_classify_heuristic counter\n")
	fmt.Fprintf(w, "compiler_classify_heuristic %d\n", m.ClassifyHeuristic.Load())
	fmt.Fprintf(w, "# HELP compiler_classify_llm Classifications resolved by LLM.\n")
	fmt.Fprintf(w, "# TYPE compiler_classify_llm counter\n")
	fmt.Fprintf(w, "compiler_classify_llm %d\n", m.ClassifyLLM.Load())
	fmt.Fprintf(w, "# HELP compiler_classify_gated Classifications gated (LLM skipped).\n")
	fmt.Fprintf(w, "# TYPE compiler_classify_gated counter\n")
	fmt.Fprintf(w, "compiler_classify_gated %d\n", m.ClassifyGated.Load())
	fmt.Fprintf(w, "# HELP compiler_gate_skips Total Bayesian gate skip decisions.\n")
	fmt.Fprintf(w, "# TYPE compiler_gate_skips counter\n")
	fmt.Fprintf(w, "compiler_gate_skips %d\n", m.GateSkips.Load())
	fmt.Fprintf(w, "# HELP compiler_llm_fallbacks Total LLM fallbacks to heuristic.\n")
	fmt.Fprintf(w, "# TYPE compiler_llm_fallbacks counter\n")
	fmt.Fprintf(w, "compiler_llm_fallbacks %d\n", m.LLMFallbacks.Load())
	fmt.Fprintf(w, "# HELP compiler_health_queries Total health score queries.\n")
	fmt.Fprintf(w, "# TYPE compiler_health_queries counter\n")
	fmt.Fprintf(w, "compiler_health_queries %d\n", m.HealthQueries.Load())
	fmt.Fprintf(w, "# HELP compiler_keywords_promoted Total keywords promoted.\n")
	fmt.Fprintf(w, "# TYPE compiler_keywords_promoted counter\n")
	fmt.Fprintf(w, "compiler_keywords_promoted %d\n", m.KeywordsPromoted.Load())
}
