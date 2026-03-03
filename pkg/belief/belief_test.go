package belief

import (
	"math"
	"testing"
)

const epsilon = 1e-9

func approxEqual(a, b float64) bool {
	return math.Abs(a-b) < epsilon
}

func TestFromConfidence(t *testing.T) {
	tests := []struct {
		name       string
		confidence float64
		wantB      float64
		wantP      float64
		wantU      float64
	}{
		{"zero", 0.0, 0.0, 0.5, 0.5},
		{"half", 0.5, 0.5, 0.75, 0.25},
		{"high", 0.8, 0.8, 0.9, 0.1},
		{"full", 1.0, 1.0, 1.0, 0.0},
		{"clamp_negative", -0.5, 0.0, 0.5, 0.5},
		{"clamp_above_one", 1.5, 1.0, 1.0, 0.0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s := FromConfidence(tt.confidence)
			if !approxEqual(s.Belief, tt.wantB) {
				t.Errorf("Belief = %v, want %v", s.Belief, tt.wantB)
			}
			if !approxEqual(s.Plausibility, tt.wantP) {
				t.Errorf("Plausibility = %v, want %v", s.Plausibility, tt.wantP)
			}
			if !approxEqual(s.Uncertainty, tt.wantU) {
				t.Errorf("Uncertainty = %v, want %v", s.Uncertainty, tt.wantU)
			}
		})
	}
}

func TestVacuous(t *testing.T) {
	v := Vacuous()
	if v.Belief != 0 || v.Plausibility != 1 || v.Uncertainty != 1 {
		t.Errorf("Vacuous() = %+v, want {0, 1, 1}", v)
	}
}

func TestCombineAgreement(t *testing.T) {
	// Two agreeing evidence sources should increase belief
	a := FromConfidence(0.7)
	b := FromConfidence(0.8)
	combined := a.Combine(b)

	if combined.Belief <= a.Belief || combined.Belief <= b.Belief {
		t.Errorf("Combined belief %v should be higher than individual beliefs %v, %v",
			combined.Belief, a.Belief, b.Belief)
	}
	if combined.Uncertainty >= a.Uncertainty || combined.Uncertainty >= b.Uncertainty {
		t.Errorf("Combined uncertainty %v should be lower than individual uncertainties %v, %v",
			combined.Uncertainty, a.Uncertainty, b.Uncertainty)
	}
}

func TestCombineTotalConflict(t *testing.T) {
	// Total conflict returns vacuous
	a := New(1.0, 1.0, 0.0) // completely believes H
	b := New(0.0, 0.0, 0.0) // completely believes ¬H
	combined := a.Combine(b)

	if combined.Belief != 0 || combined.Plausibility != 1 || combined.Uncertainty != 1 {
		t.Errorf("Total conflict should return Vacuous, got %+v", combined)
	}
}

func TestCombineWithVacuous(t *testing.T) {
	// Combining with vacuous should return roughly the original
	a := FromConfidence(0.8)
	v := Vacuous()
	combined := a.Combine(v)

	if !approxEqual(combined.Belief, a.Belief) {
		t.Errorf("Combine with Vacuous: Belief = %v, want %v", combined.Belief, a.Belief)
	}
}

func TestConflict(t *testing.T) {
	tests := []struct {
		name string
		a, b State
		want float64
	}{
		{"agreement", FromConfidence(0.8), FromConfidence(0.7), 0.0},
		{"no_evidence", Vacuous(), Vacuous(), 0.0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			k := tt.a.Conflict(tt.b)
			if k < 0 || k > 1 {
				t.Errorf("Conflict = %v, should be in [0, 1]", k)
			}
		})
	}
}

func TestToConfidence(t *testing.T) {
	s := FromConfidence(0.8)
	c := s.ToConfidence()

	// Pignistic: belief + uncertainty/2 = 0.8 + 0.1/2 = 0.85
	if !approxEqual(c, 0.85) {
		t.Errorf("ToConfidence() = %v, want 0.85", c)
	}
}

func TestIsHighConflict(t *testing.T) {
	a := FromConfidence(0.9)
	b := FromConfidence(0.9)
	if a.IsHighConflict(b, 0.5) {
		t.Error("Two agreeing states should not be high conflict")
	}
}

func TestCombineMultiple(t *testing.T) {
	// Empty returns vacuous
	v := CombineMultiple(nil)
	if v.Belief != 0 {
		t.Errorf("CombineMultiple(nil) Belief = %v, want 0", v.Belief)
	}

	// Single element returns itself
	single := FromConfidence(0.7)
	result := CombineMultiple([]State{single})
	if !approxEqual(result.Belief, single.Belief) {
		t.Errorf("CombineMultiple single: Belief = %v, want %v", result.Belief, single.Belief)
	}

	// Multiple agreeing should increase belief
	states := []State{
		FromConfidence(0.6),
		FromConfidence(0.7),
		FromConfidence(0.5),
	}
	multi := CombineMultiple(states)
	if multi.Belief <= states[0].Belief {
		t.Errorf("Multiple agreeing states should increase belief, got %v", multi.Belief)
	}
}

func TestTemporalDecay(t *testing.T) {
	s := FromConfidence(0.8)

	// No decay at age 0
	nodecay := s.TemporalDecay(0, 24)
	if !approxEqual(nodecay.Belief, s.Belief) {
		t.Errorf("TemporalDecay(0) should not change belief, got %v", nodecay.Belief)
	}

	// Decay should reduce belief
	decayed := s.TemporalDecay(48, 24) // 2 half-lives
	if decayed.Belief >= s.Belief {
		t.Errorf("TemporalDecay should reduce belief: got %v, original %v", decayed.Belief, s.Belief)
	}

	// After 1 half-life, belief should be ~half
	oneHalf := s.TemporalDecay(24, 24)
	expectedBelief := s.Belief * 0.5
	if !approxEqual(oneHalf.Belief, expectedBelief) {
		t.Errorf("After 1 half-life, Belief = %v, want ~%v", oneHalf.Belief, expectedBelief)
	}

	// Invalid half-life should return unchanged
	invalid := s.TemporalDecay(24, 0)
	if !approxEqual(invalid.Belief, s.Belief) {
		t.Errorf("Invalid half-life should return unchanged, got %v", invalid.Belief)
	}
}

// Property-based tests

func TestBeliefInvariants(t *testing.T) {
	// Property: for any valid state, belief ≤ plausibility
	// Property: uncertainty = plausibility - belief (approximately)
	confidences := []float64{0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0}

	for _, c := range confidences {
		s := FromConfidence(c)
		if s.Belief > s.Plausibility+epsilon {
			t.Errorf("Invariant violated: belief (%v) > plausibility (%v) for confidence %v",
				s.Belief, s.Plausibility, c)
		}
		expectedU := s.Plausibility - s.Belief
		if !approxEqual(s.Uncertainty, expectedU) {
			t.Errorf("Invariant violated: uncertainty (%v) != plausibility-belief (%v) for confidence %v",
				s.Uncertainty, expectedU, c)
		}
	}
}

func TestCombineCommutativity(t *testing.T) {
	// Dempster's rule is commutative: A⊕B = B⊕A
	a := FromConfidence(0.6)
	b := FromConfidence(0.8)

	ab := a.Combine(b)
	ba := b.Combine(a)

	if !approxEqual(ab.Belief, ba.Belief) {
		t.Errorf("Combine not commutative: A⊕B belief=%v, B⊕A belief=%v", ab.Belief, ba.Belief)
	}
	if !approxEqual(ab.Plausibility, ba.Plausibility) {
		t.Errorf("Combine not commutative: A⊕B plaus=%v, B⊕A plaus=%v", ab.Plausibility, ba.Plausibility)
	}
}

func TestCombineAssociativity(t *testing.T) {
	// Dempster's rule is associative: (A⊕B)⊕C = A⊕(B⊕C)
	a := FromConfidence(0.5)
	b := FromConfidence(0.6)
	c := FromConfidence(0.7)

	left := a.Combine(b).Combine(c)  // (A⊕B)⊕C
	right := a.Combine(b.Combine(c)) // A⊕(B⊕C)

	if !approxEqual(left.Belief, right.Belief) {
		t.Errorf("Combine not associative: (A⊕B)⊕C belief=%v, A⊕(B⊕C) belief=%v", left.Belief, right.Belief)
	}
}

func TestCombineVacuousIdentity(t *testing.T) {
	// Vacuous is the identity element: A⊕Vacuous = A
	confidences := []float64{0.1, 0.3, 0.5, 0.7, 0.9}
	v := Vacuous()

	for _, c := range confidences {
		a := FromConfidence(c)
		combined := a.Combine(v)

		if !approxEqual(combined.Belief, a.Belief) {
			t.Errorf("Vacuous not identity for c=%v: belief %v != %v", c, combined.Belief, a.Belief)
		}
	}
}

func TestTemporalDecayMonotonicity(t *testing.T) {
	// Property: older evidence always has lower belief
	s := FromConfidence(0.9)
	halfLife := 24.0

	prev := s
	for age := 0.0; age <= 120; age += 12 {
		current := s.TemporalDecay(age, halfLife)
		if current.Belief > prev.Belief+epsilon {
			t.Errorf("Decay not monotonic at age %v: belief %v > previous %v", age, current.Belief, prev.Belief)
		}
		prev = current
	}
}
