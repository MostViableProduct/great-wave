package api

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestMetrics_PrometheusFormat(t *testing.T) {
	m := &Metrics{}
	m.ClassifyTotal.Store(42)
	m.ClassifyHeuristic.Store(30)
	m.ClassifyLLM.Store(10)
	m.ClassifyGated.Store(2)
	m.GateSkips.Store(5)
	m.LLMFallbacks.Store(3)
	m.HealthQueries.Store(100)
	m.KeywordsPromoted.Store(7)

	req := httptest.NewRequest("GET", "/metrics", nil)
	w := httptest.NewRecorder()
	m.handleMetrics(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", w.Code)
	}

	ct := w.Header().Get("Content-Type")
	if !strings.Contains(ct, "text/plain") {
		t.Errorf("expected text/plain content type, got %s", ct)
	}

	body := w.Body.String()
	checks := []string{
		"compiler_classify_total 42",
		"compiler_classify_heuristic 30",
		"compiler_classify_llm 10",
		"compiler_classify_gated 2",
		"compiler_gate_skips 5",
		"compiler_llm_fallbacks 3",
		"compiler_health_queries 100",
		"compiler_keywords_promoted 7",
		"# TYPE compiler_classify_total counter",
		"# HELP compiler_classify_total",
	}
	for _, check := range checks {
		if !strings.Contains(body, check) {
			t.Errorf("expected %q in metrics output", check)
		}
	}
}

func TestMetrics_InitiallyZero(t *testing.T) {
	m := &Metrics{}

	req := httptest.NewRequest("GET", "/metrics", nil)
	w := httptest.NewRecorder()
	m.handleMetrics(w, req)

	body := w.Body.String()
	if !strings.Contains(body, "compiler_classify_total 0") {
		t.Errorf("expected initial counter 0, got:\n%s", body)
	}
}
