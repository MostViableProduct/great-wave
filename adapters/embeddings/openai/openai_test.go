package openaiembed

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestEmbed_Success(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !strings.HasPrefix(r.Header.Get("Authorization"), "Bearer ") {
			t.Errorf("expected Bearer auth, got %s", r.Header.Get("Authorization"))
		}
		if r.URL.Path != "/v1/embeddings" {
			t.Errorf("expected /v1/embeddings, got %s", r.URL.Path)
		}

		json.NewEncoder(w).Encode(map[string]any{
			"data": []map[string]any{
				{"embedding": []float32{0.1, 0.2, 0.3}, "index": 0},
			},
		})
	}))
	defer srv.Close()

	client := New("test-key", WithBaseURL(srv.URL))
	result, err := client.Embed(context.Background(), []string{"hello world"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(result) != 1 {
		t.Fatalf("expected 1 embedding, got %d", len(result))
	}
	if len(result[0]) != 3 {
		t.Errorf("expected 3 dimensions, got %d", len(result[0]))
	}
}

func TestEmbed_BatchMultipleTexts(t *testing.T) {
	var capturedBody map[string]any

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&capturedBody)
		json.NewEncoder(w).Encode(map[string]any{
			"data": []map[string]any{
				{"embedding": []float32{0.1, 0.2}, "index": 0},
				{"embedding": []float32{0.3, 0.4}, "index": 1},
				{"embedding": []float32{0.5, 0.6}, "index": 2},
			},
		})
	}))
	defer srv.Close()

	client := New("k", WithBaseURL(srv.URL))
	texts := []string{"first", "second", "third"}
	result, err := client.Embed(context.Background(), texts)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(result) != 3 {
		t.Fatalf("expected 3 embeddings, got %d", len(result))
	}

	// Verify all texts sent in one batch
	input := capturedBody["input"].([]any)
	if len(input) != 3 {
		t.Errorf("expected 3 inputs in batch, got %d", len(input))
	}
}

func TestEmbed_HTTPError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusTooManyRequests)
		w.Write([]byte(`{"error":"rate limited"}`))
	}))
	defer srv.Close()

	client := New("k", WithBaseURL(srv.URL))
	_, err := client.Embed(context.Background(), []string{"test"})
	if err == nil {
		t.Fatal("expected error for HTTP 429")
	}
	if !strings.Contains(err.Error(), "429") {
		t.Errorf("expected 429 in error, got: %v", err)
	}
}

func TestEmbed_EmptyInput(t *testing.T) {
	client := New("k")
	result, err := client.Embed(context.Background(), []string{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result != nil {
		t.Errorf("expected nil for empty input, got %v", result)
	}
}

func TestEmbed_NilInput(t *testing.T) {
	client := New("k")
	result, err := client.Embed(context.Background(), nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result != nil {
		t.Errorf("expected nil for nil input, got %v", result)
	}
}

func TestWithModel_Embedding(t *testing.T) {
	var capturedModel string

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		json.NewDecoder(r.Body).Decode(&body)
		capturedModel = body["model"].(string)
		json.NewEncoder(w).Encode(map[string]any{
			"data": []map[string]any{
				{"embedding": []float32{0.1}, "index": 0},
			},
		})
	}))
	defer srv.Close()

	client := New("k", WithBaseURL(srv.URL), WithModel("text-embedding-3-large"))
	client.Embed(context.Background(), []string{"test"})

	if capturedModel != "text-embedding-3-large" {
		t.Errorf("expected text-embedding-3-large, got %s", capturedModel)
	}
}

func TestWithDimensions(t *testing.T) {
	var capturedDims float64

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		json.NewDecoder(r.Body).Decode(&body)
		capturedDims = body["dimensions"].(float64)
		json.NewEncoder(w).Encode(map[string]any{
			"data": []map[string]any{
				{"embedding": []float32{0.1}, "index": 0},
			},
		})
	}))
	defer srv.Close()

	client := New("k", WithBaseURL(srv.URL), WithDimensions(512))
	client.Embed(context.Background(), []string{"test"})

	if capturedDims != 512 {
		t.Errorf("expected dimensions=512, got %v", capturedDims)
	}
	if client.Dimensions() != 512 {
		t.Errorf("expected Dimensions()=512, got %d", client.Dimensions())
	}
}

func TestDimensions_Default(t *testing.T) {
	client := New("k")
	if client.Dimensions() != defaultDimensions {
		t.Errorf("expected default dimensions %d, got %d", defaultDimensions, client.Dimensions())
	}
}
