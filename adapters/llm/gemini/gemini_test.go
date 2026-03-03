package gemini

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func geminiResponse(category string, confidence float64, keywords []string) map[string]any {
	classJSON, _ := json.Marshal(map[string]any{
		"category":   category,
		"confidence": confidence,
		"keywords":   keywords,
	})
	return map[string]any{
		"candidates": []map[string]any{
			{
				"content": map[string]any{
					"parts": []map[string]string{
						{"text": string(classJSON)},
					},
				},
			},
		},
	}
}

func TestClassify_Success(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(geminiResponse("reliability", 0.85, []string{"crash", "error"}))
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

func TestClassify_PromptVerification(t *testing.T) {
	var capturedBody map[string]any

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&capturedBody)
		json.NewEncoder(w).Encode(geminiResponse("security", 0.8, []string{"cve"}))
	}))
	defer srv.Close()

	client := New("k", WithBaseURL(srv.URL))
	_, err := client.Classify(context.Background(), "CVE found", "audit", []string{"security", "performance"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Verify systemInstruction is set
	sysInstr := capturedBody["systemInstruction"].(map[string]any)
	parts := sysInstr["parts"].([]any)
	if len(parts) == 0 {
		t.Fatal("expected system instruction parts")
	}
	sysText := parts[0].(map[string]any)["text"].(string)
	if sysText == "" {
		t.Error("expected system prompt text")
	}

	// Verify contents include categories
	contents := capturedBody["contents"].([]any)
	contentParts := contents[0].(map[string]any)["parts"].([]any)
	userText := contentParts[0].(map[string]any)["text"].(string)
	if !strings.Contains(userText, "security") || !strings.Contains(userText, "performance") {
		t.Errorf("user prompt missing categories: %s", userText)
	}

	// Verify generationConfig
	genConfig := capturedBody["generationConfig"].(map[string]any)
	if genConfig["responseMimeType"] != "application/json" {
		t.Errorf("expected responseMimeType=application/json, got %v", genConfig["responseMimeType"])
	}
}

func TestClassify_APIKeyInURL(t *testing.T) {
	var capturedURL string

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		capturedURL = r.URL.String()
		json.NewEncoder(w).Encode(geminiResponse("performance", 0.7, nil))
	}))
	defer srv.Close()

	client := New("my-secret-key", WithBaseURL(srv.URL))
	client.Classify(context.Background(), "test", "metric", []string{"performance"})

	if !strings.Contains(capturedURL, "key=my-secret-key") {
		t.Errorf("expected API key in URL query param, got %s", capturedURL)
	}
}

func TestClassify_UnknownCategory(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		json.NewEncoder(w).Encode(geminiResponse("unknown_cat", 0.9, nil))
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
			"candidates": []map[string]any{
				{
					"content": map[string]any{
						"parts": []map[string]string{
							{"text": "not valid json at all"},
						},
					},
				},
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

func TestClassify_EmptyCandidates(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		json.NewEncoder(w).Encode(map[string]any{
			"candidates": []map[string]any{},
		})
	}))
	defer srv.Close()

	client := New("k", WithBaseURL(srv.URL))
	_, err := client.Classify(context.Background(), "test", "metric", []string{"performance"})
	if err == nil {
		t.Fatal("expected error for empty candidates")
	}
	if !strings.Contains(err.Error(), "no candidates") {
		t.Errorf("expected 'no candidates' error, got: %v", err)
	}
}

func TestWithModel_Gemini(t *testing.T) {
	var capturedURL string

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		capturedURL = r.URL.String()
		json.NewEncoder(w).Encode(geminiResponse("performance", 0.5, nil))
	}))
	defer srv.Close()

	client := New("k", WithBaseURL(srv.URL), WithModel("gemini-1.5-pro"))
	client.Classify(context.Background(), "test", "metric", []string{"performance"})

	if !strings.Contains(capturedURL, "gemini-1.5-pro") {
		t.Errorf("expected gemini-1.5-pro in URL, got %s", capturedURL)
	}
}
