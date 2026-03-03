// Package api provides HTTP handlers for the contextual compiler REST API.
//
// All handlers operate on a *compiler.Compiler instance and expose the
// cascade classification, health scoring, and keyword promotion endpoints.
package api

import (
	"encoding/json"
	"log"
	"math"
	"net/http"

	"github.com/MostViableProduct/great-wave/pkg/compiler"
)

const (
	maxContentLen = 100_000 // 100 KB
	maxFieldLen   = 1_000
	maxPayloadLen = 512_000 // 512 KB
	maxBodySize   = 1 << 20 // 1 MiB
)

// Handler wraps a compiler and exposes HTTP endpoints.
type Handler struct {
	compiler *compiler.Compiler
	metrics  *Metrics
}

// HandlerOption configures a Handler.
type HandlerOption func(*Handler)

// WithMetrics attaches a Metrics instance for counter tracking.
func WithMetrics(m *Metrics) HandlerOption {
	return func(h *Handler) { h.metrics = m }
}

// NewHandler creates a Handler from a configured compiler.
func NewHandler(c *compiler.Compiler, opts ...HandlerOption) *Handler {
	h := &Handler{compiler: c}
	for _, opt := range opts {
		opt(h)
	}
	return h
}

// HealthHandler returns the HTTP handler for GET /health.
// It is intentionally separate from RegisterProtectedRoutes so callers can
// mount it outside the auth middleware chain (Docker healthchecks send no token).
func (h *Handler) HealthHandler() http.HandlerFunc {
	return h.handleHealth
}

// RegisterProtectedRoutes registers all authenticated API routes on a ServeMux.
// The caller is responsible for auth, rate-limiting, and other middleware.
// GET /health is not included here — use HealthHandler instead.
func (h *Handler) RegisterProtectedRoutes(mux *http.ServeMux) {
	mux.HandleFunc("POST /v1/classify", h.handleClassify)
	mux.HandleFunc("POST /v1/classify/signal", h.handleClassifySignal)
	mux.HandleFunc("GET /v1/health/{tenant_id}/{entity_id}", h.handleEntityHealth)
	mux.HandleFunc("POST /v1/health/{tenant_id}/{entity_id}/events", h.handleRecordHealthEvent)
	mux.HandleFunc("POST /v1/keywords/promote", h.handlePromoteKeywords)
	mux.HandleFunc("POST /v1/state/flush", h.handleFlushState)
	if h.metrics != nil {
		mux.HandleFunc("GET /metrics", h.metrics.handleMetrics)
	}
}

// RegisterRoutes registers all API routes on a ServeMux, including /health.
// Use this when the caller manages auth externally for all routes.
func (h *Handler) RegisterRoutes(mux *http.ServeMux) {
	mux.HandleFunc("GET /health", h.handleHealth)
	h.RegisterProtectedRoutes(mux)
}

func (h *Handler) handleHealth(w http.ResponseWriter, _ *http.Request) {
	depStatus := func(has bool) string {
		if has {
			return "connected"
		}
		return "disabled"
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"status": "ok",
		"dependencies": map[string]string{
			"llm":     depStatus(h.compiler.HasLLM()),
			"vector":  depStatus(h.compiler.HasVector()),
			"events":  depStatus(h.compiler.HasEvents()),
			"storage": depStatus(h.compiler.HasStorage()),
		},
	})
}

// ClassifyRequest is the request body for POST /v1/classify.
type ClassifyRequest struct {
	TenantID string `json:"tenant_id,omitempty"`
	Source   string `json:"source"`
	Type     string `json:"type"`
	Content  string `json:"content"`
	Payload  json.RawMessage `json:"payload,omitempty"`
}

func (h *Handler) handleClassify(w http.ResponseWriter, r *http.Request) {
	r.Body = http.MaxBytesReader(w, r.Body, maxBodySize)
	var req ClassifyRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid request body")
		return
	}
	if req.Type == "" {
		writeError(w, http.StatusBadRequest, "missing required field: type")
		return
	}
	if req.Content == "" {
		writeError(w, http.StatusBadRequest, "missing required field: content")
		return
	}
	if len(req.Content) > maxContentLen {
		writeError(w, http.StatusBadRequest, "content exceeds maximum length")
		return
	}
	if len(req.Type) > maxFieldLen || len(req.Source) > maxFieldLen || len(req.TenantID) > maxFieldLen {
		writeError(w, http.StatusBadRequest, "field exceeds maximum length")
		return
	}

	result, err := h.compiler.Classify(r.Context(), compiler.Signal{
		TenantID: req.TenantID,
		Source:   req.Source,
		Type:     req.Type,
		Content:  req.Content,
		Payload:  req.Payload,
	})
	if err != nil {
		log.Printf("api: classify error: %v", err)
		writeError(w, http.StatusUnprocessableEntity, "classification failed")
		return
	}

	h.recordClassifyMetrics(result)
	writeJSON(w, http.StatusOK, result)
}

// ClassifySignalRequest is the request body for POST /v1/classify/signal.
type ClassifySignalRequest struct {
	TenantID string          `json:"tenant_id,omitempty"`
	Source   string          `json:"source"`
	Type     string          `json:"type"`
	Payload  json.RawMessage `json:"payload"`
}

func (h *Handler) handleClassifySignal(w http.ResponseWriter, r *http.Request) {
	r.Body = http.MaxBytesReader(w, r.Body, maxBodySize)
	var req ClassifySignalRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid request body")
		return
	}
	if req.Type == "" {
		writeError(w, http.StatusBadRequest, "missing required field: type")
		return
	}
	if len(req.Payload) == 0 {
		writeError(w, http.StatusBadRequest, "missing required field: payload")
		return
	}
	if len(req.Payload) > maxPayloadLen {
		writeError(w, http.StatusBadRequest, "payload exceeds maximum length")
		return
	}
	if len(req.Type) > maxFieldLen || len(req.Source) > maxFieldLen || len(req.TenantID) > maxFieldLen {
		writeError(w, http.StatusBadRequest, "field exceeds maximum length")
		return
	}

	result, err := h.compiler.ClassifySignal(r.Context(), req.TenantID, req.Source, req.Type, req.Payload)
	if err != nil {
		log.Printf("api: classify signal error: %v", err)
		writeError(w, http.StatusUnprocessableEntity, "signal classification failed")
		return
	}

	h.recordClassifyMetrics(result)
	writeJSON(w, http.StatusOK, result)
}

func (h *Handler) handleEntityHealth(w http.ResponseWriter, r *http.Request) {
	tenantID := r.PathValue("tenant_id")
	entityID := r.PathValue("entity_id")
	if tenantID == "" || entityID == "" {
		writeError(w, http.StatusBadRequest, "tenant_id and entity_id are required")
		return
	}
	if len(tenantID) > maxFieldLen || len(entityID) > maxFieldLen {
		writeError(w, http.StatusBadRequest, "field exceeds maximum length")
		return
	}

	if h.metrics != nil {
		h.metrics.HealthQueries.Add(1)
	}
	result := h.compiler.ScoreHealth(tenantID, entityID)
	writeJSON(w, http.StatusOK, result)
}

// RecordHealthEventRequest is the request body for POST /v1/health/{tenant_id}/{entity_id}/events.
type RecordHealthEventRequest struct {
	Severity   string  `json:"severity"`
	Category   string  `json:"category"`
	Confidence float64 `json:"confidence"`
}

func (h *Handler) handleRecordHealthEvent(w http.ResponseWriter, r *http.Request) {
	tenantID := r.PathValue("tenant_id")
	entityID := r.PathValue("entity_id")
	if tenantID == "" || entityID == "" {
		writeError(w, http.StatusBadRequest, "tenant_id and entity_id are required")
		return
	}
	if len(tenantID) > maxFieldLen || len(entityID) > maxFieldLen {
		writeError(w, http.StatusBadRequest, "field exceeds maximum length")
		return
	}

	r.Body = http.MaxBytesReader(w, r.Body, maxBodySize)
	var req RecordHealthEventRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid request body")
		return
	}
	if req.Severity == "" {
		writeError(w, http.StatusBadRequest, "missing required field: severity")
		return
	}
	if len(req.Severity) > maxFieldLen || len(req.Category) > maxFieldLen {
		writeError(w, http.StatusBadRequest, "field exceeds maximum length")
		return
	}
	if math.IsNaN(req.Confidence) || math.IsInf(req.Confidence, 0) || req.Confidence < 0 || req.Confidence > 1 {
		writeError(w, http.StatusBadRequest, "confidence must be between 0.0 and 1.0")
		return
	}
	if severities := h.compiler.ValidSeverities(); len(severities) > 0 {
		if !severities[req.Severity] {
			writeError(w, http.StatusBadRequest, "unknown severity level")
			return
		}
	}

	h.compiler.RecordHealthEvent(tenantID, entityID, req.Severity, req.Category, req.Confidence)
	writeJSON(w, http.StatusOK, map[string]string{"status": "recorded"})
}

func (h *Handler) handlePromoteKeywords(w http.ResponseWriter, r *http.Request) {
	count, err := h.compiler.PromoteKeywords()
	if err != nil {
		log.Printf("api: promote keywords error: %v", err)
		writeError(w, http.StatusInternalServerError, "keyword promotion failed")
		return
	}
	if h.metrics != nil && count > 0 {
		h.metrics.KeywordsPromoted.Add(int64(count))
	}
	writeJSON(w, http.StatusOK, map[string]interface{}{"promoted": count})
}

func (h *Handler) handleFlushState(w http.ResponseWriter, r *http.Request) {
	if err := h.compiler.FlushState(); err != nil {
		log.Printf("api: flush state error: %v", err)
		writeError(w, http.StatusInternalServerError, "state flush failed")
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"status": "flushed"})
}

func (h *Handler) recordClassifyMetrics(result *compiler.ClassifyResult) {
	if h.metrics == nil {
		return
	}
	h.metrics.ClassifyTotal.Add(1)
	switch result.ClassificationSource {
	case compiler.SourceLLM:
		h.metrics.ClassifyLLM.Add(1)
	case compiler.SourceHeuristicGated:
		h.metrics.ClassifyGated.Add(1)
	default:
		h.metrics.ClassifyHeuristic.Add(1)
	}
}

func writeJSON(w http.ResponseWriter, status int, v interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	if err := json.NewEncoder(w).Encode(v); err != nil {
		log.Printf("api: failed to encode response: %v", err)
	}
}

func writeError(w http.ResponseWriter, status int, message string) {
	writeJSON(w, status, map[string]string{"error": message})
}
