package openai

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
		if !strings.HasPrefix(r.Header.Get("Authorization"), "Bearer ") {
			t.Errorf("expected Bearer auth, got %s", r.Header.Get("Authorization"))
		}

		json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{
				{"message": map[string]string{
					"content": `{"category":"reliability","confidence":0.85,"keywords":["crash","error"]}`,
				}},
			},
		})
	}))
	defer srv.Close()

	client := New("test-key", WithBaseURL(srv.URL))
	result, err := client.Classify(context.Background(), "service crash detected", "error", []string{"performance", "reliability"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Category != "reliability" {
		t.Errorf("expected reliability, got %s", result.Category)
	}
	if result.Confidence != 0.85 {
		t.Errorf("expected confidence 0.85, got %f", result.Confidence)
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
			"choices": []map[string]any{
				{"message": map[string]string{
					"content": `{"category":"security","confidence":0.8,"keywords":["cve"]}`,
				}},
			},
		})
	}))
	defer srv.Close()

	client := New("k", WithBaseURL(srv.URL))
	_, err := client.Classify(context.Background(), "CVE found", "audit", []string{"security", "performance"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Verify messages format (system + user)
	messages := capturedBody["messages"].([]any)
	if len(messages) != 2 {
		t.Fatalf("expected 2 messages (system+user), got %d", len(messages))
	}

	systemMsg := messages[0].(map[string]any)["content"].(string)
	if systemMsg == "" {
		t.Error("expected system prompt to be set")
	}

	userMsg := messages[1].(map[string]any)["content"].(string)
	if !strings.Contains(userMsg, "security") || !strings.Contains(userMsg, "performance") {
		t.Errorf("user prompt missing categories: %s", userMsg)
	}

	// Verify response_format
	rf := capturedBody["response_format"].(map[string]any)
	if rf["type"] != "json_object" {
		t.Errorf("expected response_format json_object, got %v", rf["type"])
	}

	// Verify model
	if capturedBody["model"] != defaultModel {
		t.Errorf("expected model %s, got %s", defaultModel, capturedBody["model"])
	}
}

func TestClassify_UnknownCategory(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{
				{"message": map[string]string{
					"content": `{"category":"unknown_cat","confidence":0.9,"keywords":[]}`,
				}},
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
			"choices": []map[string]any{
				{"message": map[string]string{
					"content": `not valid json at all`,
				}},
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
		w.WriteHeader(http.StatusInternalServerError)
		w.Write([]byte(`{"error":"internal error"}`))
	}))
	defer srv.Close()

	client := New("k", WithBaseURL(srv.URL))
	_, err := client.Classify(context.Background(), "test", "metric", []string{"performance"})
	if err == nil {
		t.Fatal("expected error for HTTP 500")
	}
	if !strings.Contains(err.Error(), "500") {
		t.Errorf("expected 500 in error, got: %v", err)
	}
}

func TestClassify_NoChoices(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{},
		})
	}))
	defer srv.Close()

	client := New("k", WithBaseURL(srv.URL))
	_, err := client.Classify(context.Background(), "test", "metric", []string{"performance"})
	if err == nil {
		t.Fatal("expected error for no choices")
	}
}

func TestWithModel(t *testing.T) {
	var capturedModel string

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		json.NewDecoder(r.Body).Decode(&body)
		capturedModel = body["model"].(string)
		json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{
				{"message": map[string]string{
					"content": `{"category":"perf","confidence":0.5,"keywords":[]}`,
				}},
			},
		})
	}))
	defer srv.Close()

	client := New("k", WithBaseURL(srv.URL), WithModel("gpt-5.3"))
	client.Classify(context.Background(), "test", "metric", []string{"perf"})

	if capturedModel != "gpt-5.3" {
		t.Errorf("expected gpt-5.3, got %s", capturedModel)
	}
}
