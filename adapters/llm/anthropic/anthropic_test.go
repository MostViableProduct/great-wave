package anthropic

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestClassify_Success(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("x-api-key") != "test-key" {
			t.Errorf("expected x-api-key test-key, got %s", r.Header.Get("x-api-key"))
		}
		if r.Header.Get("anthropic-version") != apiVersion {
			t.Errorf("expected anthropic-version %s, got %s", apiVersion, r.Header.Get("anthropic-version"))
		}

		json.NewEncoder(w).Encode(map[string]any{
			"content": []map[string]string{
				{"type": "text", "text": `{"category":"performance","confidence":0.92,"keywords":["latency","p99"]}`},
			},
		})
	}))
	defer srv.Close()

	client := New("test-key", WithBaseURL(srv.URL))
	result, err := client.Classify(context.Background(), "high p99 latency", "metric", []string{"performance", "reliability"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Category != "performance" {
		t.Errorf("expected performance, got %s", result.Category)
	}
	if result.Confidence != 0.92 {
		t.Errorf("expected confidence 0.92, got %f", result.Confidence)
	}
	if len(result.Keywords) != 2 {
		t.Errorf("expected 2 keywords, got %d", len(result.Keywords))
	}
}

func TestClassify_PromptContainsCategories(t *testing.T) {
	var capturedBody map[string]any

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&capturedBody)
		json.NewEncoder(w).Encode(map[string]any{
			"content": []map[string]string{
				{"type": "text", "text": `{"category":"security","confidence":0.8,"keywords":["cve"]}`},
			},
		})
	}))
	defer srv.Close()

	client := New("k", WithBaseURL(srv.URL))
	_, err := client.Classify(context.Background(), "CVE found", "audit", []string{"security", "performance"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Verify categories appear in user message
	messages := capturedBody["messages"].([]any)
	userMsg := messages[0].(map[string]any)["content"].(string)
	if !strings.Contains(userMsg, "security") || !strings.Contains(userMsg, "performance") {
		t.Errorf("user prompt missing categories: %s", userMsg)
	}

	// Verify system prompt is set
	if capturedBody["system"] == nil || capturedBody["system"].(string) == "" {
		t.Error("expected system prompt to be set")
	}

	// Verify model
	if capturedBody["model"] != defaultModel {
		t.Errorf("expected model %s, got %s", defaultModel, capturedBody["model"])
	}
}

func TestClassify_UnknownCategory(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		json.NewEncoder(w).Encode(map[string]any{
			"content": []map[string]string{
				{"type": "text", "text": `{"category":"unknown_cat","confidence":0.9,"keywords":[]}`},
			},
		})
	}))
	defer srv.Close()

	client := New("k", WithBaseURL(srv.URL))
	_, err := client.Classify(context.Background(), "test", "metric", []string{"performance", "reliability"})
	if err == nil {
		t.Fatal("expected error for unknown category")
	}
	if !strings.Contains(err.Error(), "unknown category") {
		t.Errorf("expected unknown category error, got: %v", err)
	}
}

func TestClassify_InvalidJSON(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		json.NewEncoder(w).Encode(map[string]any{
			"content": []map[string]string{
				{"type": "text", "text": `not valid json`},
			},
		})
	}))
	defer srv.Close()

	client := New("k", WithBaseURL(srv.URL))
	_, err := client.Classify(context.Background(), "test", "metric", []string{"performance"})
	if err == nil {
		t.Fatal("expected error for invalid JSON")
	}
}

func TestClassify_HTTPError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusTooManyRequests)
		w.Write([]byte(`{"error":"rate_limited"}`))
	}))
	defer srv.Close()

	client := New("k", WithBaseURL(srv.URL))
	_, err := client.Classify(context.Background(), "test", "metric", []string{"performance"})
	if err == nil {
		t.Fatal("expected error for HTTP 429")
	}
	if !strings.Contains(err.Error(), "429") {
		t.Errorf("expected 429 in error, got: %v", err)
	}
}

func TestClassify_EmptyContent(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		json.NewEncoder(w).Encode(map[string]any{
			"content": []map[string]string{},
		})
	}))
	defer srv.Close()

	client := New("k", WithBaseURL(srv.URL))
	_, err := client.Classify(context.Background(), "test", "metric", []string{"performance"})
	if err == nil {
		t.Fatal("expected error for empty content")
	}
}

func TestWithModel(t *testing.T) {
	var capturedModel string

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		json.NewDecoder(r.Body).Decode(&body)
		capturedModel = body["model"].(string)
		json.NewEncoder(w).Encode(map[string]any{
			"content": []map[string]string{
				{"type": "text", "text": `{"category":"perf","confidence":0.5,"keywords":[]}`},
			},
		})
	}))
	defer srv.Close()

	client := New("k", WithBaseURL(srv.URL), WithModel("claude-opus-4-6"))
	client.Classify(context.Background(), "test", "metric", []string{"perf"})

	if capturedModel != "claude-opus-4-6" {
		t.Errorf("expected claude-opus-4-6, got %s", capturedModel)
	}
}
