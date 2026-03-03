package api

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/MostViableProduct/great-wave/pkg/classifier"
	"github.com/MostViableProduct/great-wave/pkg/compiler"
)

func testCompiler() *compiler.Compiler {
	cfg := compiler.DefaultConfig()
	cfg.Classifier = classifier.Config{
		Categories: []classifier.CategoryConfig{
			{
				Name:     "performance",
				Keywords: []string{"latency", "p99", "slow", "timeout"},
				Weights:  map[string]float64{"p99": 2.0},
			},
			{
				Name:     "reliability",
				Keywords: []string{"error", "failure", "crash", "outage"},
				Weights:  map[string]float64{"outage": 2.0},
			},
		},
		TypeToCategory: map[string]string{
			"metric": "performance",
			"error":  "reliability",
		},
	}
	return compiler.New(cfg, compiler.Deps{}, compiler.Callbacks{})
}

func TestHealthEndpoint(t *testing.T) {
	h := NewHandler(testCompiler())
	mux := http.NewServeMux()
	h.RegisterRoutes(mux)

	req := httptest.NewRequest("GET", "/health", nil)
	w := httptest.NewRecorder()
	mux.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("Expected 200, got %d", w.Code)
	}

	var resp struct {
		Status       string            `json:"status"`
		Dependencies map[string]string `json:"dependencies"`
	}
	json.NewDecoder(w.Body).Decode(&resp)
	if resp.Status != "ok" {
		t.Errorf("Expected status ok, got %s", resp.Status)
	}
	if resp.Dependencies["llm"] != "disabled" {
		t.Errorf("Expected llm=disabled, got %s", resp.Dependencies["llm"])
	}
	if resp.Dependencies["vector"] != "disabled" {
		t.Errorf("Expected vector=disabled, got %s", resp.Dependencies["vector"])
	}
	if resp.Dependencies["events"] != "disabled" {
		t.Errorf("Expected events=disabled, got %s", resp.Dependencies["events"])
	}
	if resp.Dependencies["storage"] != "disabled" {
		t.Errorf("Expected storage=disabled, got %s", resp.Dependencies["storage"])
	}
}

func TestClassifyEndpoint(t *testing.T) {
	h := NewHandler(testCompiler())
	mux := http.NewServeMux()
	h.RegisterRoutes(mux)

	body := ClassifyRequest{
		Source:  "prometheus",
		Type:    "metric",
		Content: "High p99 latency detected in API gateway",
	}
	b, _ := json.Marshal(body)

	req := httptest.NewRequest("POST", "/v1/classify", bytes.NewReader(b))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	mux.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("Expected 200, got %d: %s", w.Code, w.Body.String())
	}

	var resp compiler.ClassifyResult
	json.NewDecoder(w.Body).Decode(&resp)
	if resp.Category != "performance" {
		t.Errorf("Expected performance, got %s", resp.Category)
	}
}

func TestClassifyEndpoint_MissingType(t *testing.T) {
	h := NewHandler(testCompiler())
	mux := http.NewServeMux()
	h.RegisterRoutes(mux)

	body := ClassifyRequest{Content: "some content"}
	b, _ := json.Marshal(body)

	req := httptest.NewRequest("POST", "/v1/classify", bytes.NewReader(b))
	w := httptest.NewRecorder()
	mux.ServeHTTP(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("Expected 400, got %d", w.Code)
	}
}

func TestClassifyEndpoint_EmptyBody(t *testing.T) {
	h := NewHandler(testCompiler())
	mux := http.NewServeMux()
	h.RegisterRoutes(mux)

	req := httptest.NewRequest("POST", "/v1/classify", bytes.NewReader([]byte("")))
	w := httptest.NewRecorder()
	mux.ServeHTTP(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("Expected 400, got %d", w.Code)
	}
}

func TestClassifyEndpoint_MissingContent(t *testing.T) {
	h := NewHandler(testCompiler())
	mux := http.NewServeMux()
	h.RegisterRoutes(mux)

	body := ClassifyRequest{Type: "metric"}
	b, _ := json.Marshal(body)

	req := httptest.NewRequest("POST", "/v1/classify", bytes.NewReader(b))
	w := httptest.NewRecorder()
	mux.ServeHTTP(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("Expected 400, got %d", w.Code)
	}
}

func TestClassifySignalEndpoint(t *testing.T) {
	h := NewHandler(testCompiler())
	mux := http.NewServeMux()
	h.RegisterRoutes(mux)

	body := ClassifySignalRequest{
		TenantID: "t1",
		Source:   "prometheus",
		Type:     "metric",
		Payload:  json.RawMessage(`{"message": "p99 latency spike"}`),
	}
	b, _ := json.Marshal(body)

	req := httptest.NewRequest("POST", "/v1/classify/signal", bytes.NewReader(b))
	w := httptest.NewRecorder()
	mux.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("Expected 200, got %d: %s", w.Code, w.Body.String())
	}
}

func TestClassifySignalEndpoint_MissingType(t *testing.T) {
	h := NewHandler(testCompiler())
	mux := http.NewServeMux()
	h.RegisterRoutes(mux)

	body := ClassifySignalRequest{
		Source:  "sentry",
		Payload: json.RawMessage(`{"message": "crash"}`),
	}
	b, _ := json.Marshal(body)

	req := httptest.NewRequest("POST", "/v1/classify/signal", bytes.NewReader(b))
	w := httptest.NewRecorder()
	mux.ServeHTTP(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("Expected 400, got %d", w.Code)
	}
}

func TestClassifySignalEndpoint_MissingPayload(t *testing.T) {
	h := NewHandler(testCompiler())
	mux := http.NewServeMux()
	h.RegisterRoutes(mux)

	// Send JSON without the payload field so it decodes as nil (len 0)
	b := []byte(`{"source":"sentry","type":"error"}`)

	req := httptest.NewRequest("POST", "/v1/classify/signal", bytes.NewReader(b))
	w := httptest.NewRecorder()
	mux.ServeHTTP(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("Expected 400, got %d", w.Code)
	}
}

func TestEntityHealthEndpoint(t *testing.T) {
	h := NewHandler(testCompiler())
	mux := http.NewServeMux()
	h.RegisterRoutes(mux)

	req := httptest.NewRequest("GET", "/v1/health/tenant-1/entity-1", nil)
	w := httptest.NewRecorder()
	mux.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("Expected 200, got %d: %s", w.Code, w.Body.String())
	}

	var resp compiler.HealthResult
	json.NewDecoder(w.Body).Decode(&resp)
	if resp.EntityID != "entity-1" {
		t.Errorf("Expected entity-1, got %s", resp.EntityID)
	}
	if resp.Score < 0 || resp.Score > 100 {
		t.Errorf("Score out of bounds: %v", resp.Score)
	}
}

func TestRecordHealthEventEndpoint(t *testing.T) {
	h := NewHandler(testCompiler())
	mux := http.NewServeMux()
	h.RegisterRoutes(mux)

	body := RecordHealthEventRequest{
		Severity:   "critical",
		Category:   "reliability",
		Confidence: 0.9,
	}
	b, _ := json.Marshal(body)

	req := httptest.NewRequest("POST", "/v1/health/tenant-1/entity-1/events", bytes.NewReader(b))
	w := httptest.NewRecorder()
	mux.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("Expected 200, got %d: %s", w.Code, w.Body.String())
	}
}

func TestRecordHealthEventEndpoint_MissingSeverity(t *testing.T) {
	h := NewHandler(testCompiler())
	mux := http.NewServeMux()
	h.RegisterRoutes(mux)

	body := RecordHealthEventRequest{
		Category:   "reliability",
		Confidence: 0.9,
	}
	b, _ := json.Marshal(body)

	req := httptest.NewRequest("POST", "/v1/health/tenant-1/entity-1/events", bytes.NewReader(b))
	w := httptest.NewRecorder()
	mux.ServeHTTP(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("Expected 400, got %d", w.Code)
	}
}

func TestPromoteKeywordsEndpoint(t *testing.T) {
	h := NewHandler(testCompiler())
	mux := http.NewServeMux()
	h.RegisterRoutes(mux)

	req := httptest.NewRequest("POST", "/v1/keywords/promote", nil)
	w := httptest.NewRecorder()
	mux.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("Expected 200, got %d: %s", w.Code, w.Body.String())
	}
}

func TestFlushStateEndpoint(t *testing.T) {
	h := NewHandler(testCompiler())
	mux := http.NewServeMux()
	h.RegisterRoutes(mux)

	req := httptest.NewRequest("POST", "/v1/state/flush", nil)
	w := httptest.NewRecorder()
	mux.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("Expected 200, got %d: %s", w.Code, w.Body.String())
	}
}

func TestMetricsEndpoint(t *testing.T) {
	m := &Metrics{}
	h := NewHandler(testCompiler(), WithMetrics(m))
	mux := http.NewServeMux()
	h.RegisterRoutes(mux)

	// Do a classify to increment counters
	body := ClassifyRequest{
		Source:  "prometheus",
		Type:    "metric",
		Content: "High p99 latency detected",
	}
	b, _ := json.Marshal(body)
	req := httptest.NewRequest("POST", "/v1/classify", bytes.NewReader(b))
	w := httptest.NewRecorder()
	mux.ServeHTTP(w, req)
	if w.Code != http.StatusOK {
		t.Fatalf("Classify failed: %d %s", w.Code, w.Body.String())
	}

	// Get metrics
	req = httptest.NewRequest("GET", "/metrics", nil)
	w = httptest.NewRecorder()
	mux.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("Expected 200, got %d", w.Code)
	}

	metricsBody := w.Body.String()
	if !strings.Contains(metricsBody, "compiler_classify_total 1") {
		t.Errorf("Expected classify_total=1 in metrics output:\n%s", metricsBody)
	}
	if !strings.Contains(metricsBody, "text/plain") {
		ct := w.Header().Get("Content-Type")
		if !strings.Contains(ct, "text/plain") {
			t.Errorf("Expected text/plain content type, got %s", ct)
		}
	}
}

func TestMetricsEndpoint_NotRegisteredWithoutMetrics(t *testing.T) {
	h := NewHandler(testCompiler())
	mux := http.NewServeMux()
	h.RegisterRoutes(mux)

	req := httptest.NewRequest("GET", "/metrics", nil)
	w := httptest.NewRecorder()
	mux.ServeHTTP(w, req)

	// Without metrics, /metrics should 404
	if w.Code != http.StatusNotFound {
		t.Errorf("Expected 404 without metrics, got %d", w.Code)
	}
}
