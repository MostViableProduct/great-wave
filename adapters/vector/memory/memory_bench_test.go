package memoryvec

import (
	"context"
	"fmt"
	"math"
	"testing"

	"github.com/MostViableProduct/great-wave/pkg/compiler"
)

// deterministicEmbedder returns vectors based on a simple hash for benchmarks.
type deterministicEmbedder struct {
	dims int
}

var _ compiler.Embedder = (*deterministicEmbedder)(nil)

func (d *deterministicEmbedder) Embed(_ context.Context, texts []string) ([][]float32, error) {
	result := make([][]float32, len(texts))
	for i, t := range texts {
		vec := make([]float32, d.dims)
		// Simple deterministic vector from text
		for j, ch := range t {
			vec[j%d.dims] += float32(ch) / 1000.0
		}
		// Normalize
		var norm float64
		for _, v := range vec {
			norm += float64(v) * float64(v)
		}
		norm = math.Sqrt(norm)
		if norm > 0 {
			for j := range vec {
				vec[j] /= float32(norm)
			}
		}
		result[i] = vec
	}
	return result, nil
}

func (d *deterministicEmbedder) Dimensions() int { return d.dims }

func benchStore(n int) *Store {
	emb := &deterministicEmbedder{dims: 128}
	s := New(emb)
	for i := 0; i < n; i++ {
		vec := make([]float32, 128)
		// Spread across dimensions
		idx := i % 128
		vec[idx] = 1.0
		s.Add(Entry{ID: fmt.Sprintf("entry-%d", i), Vector: vec})
	}
	return s
}

func BenchmarkSearch_100(b *testing.B) {
	s := benchStore(100)
	ctx := context.Background()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		s.Search(ctx, "test query", 10)
	}
}

func BenchmarkSearch_1000(b *testing.B) {
	s := benchStore(1000)
	ctx := context.Background()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		s.Search(ctx, "test query", 10)
	}
}

func BenchmarkSearch_10000(b *testing.B) {
	s := benchStore(10000)
	ctx := context.Background()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		s.Search(ctx, "test query", 10)
	}
}
