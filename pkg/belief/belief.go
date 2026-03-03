// Package belief implements Dempster-Shafer theory for evidence combination.
//
// Instead of a single confidence float, each measurement carries a triplet:
// {belief, plausibility, uncertainty} where:
//   - belief: minimum probability that the hypothesis is true
//   - plausibility: maximum probability (1 - belief in negation)
//   - uncertainty: plausibility - belief (epistemic gap)
//
// This enables principled fusion of independent evidence sources with explicit
// uncertainty tracking and temporal decay.
package belief

import "math"

// State implements a Dempster-Shafer belief state.
type State struct {
	Belief       float64 `json:"belief"`
	Plausibility float64 `json:"plausibility"`
	Uncertainty  float64 `json:"uncertainty"`
}

// FromConfidence converts a simple confidence float to a State.
// A confidence of 0.8 becomes {belief: 0.8, plausibility: 0.9, uncertainty: 0.1}.
func FromConfidence(confidence float64) State {
	confidence = clamp(confidence, 0, 1)
	remaining := 1 - confidence
	uncertainty := remaining * 0.5
	return State{
		Belief:       confidence,
		Plausibility: confidence + uncertainty,
		Uncertainty:  uncertainty,
	}
}

// New creates a State with explicit values.
func New(belief, plausibility, uncertainty float64) State {
	return State{
		Belief:       clamp(belief, 0, 1),
		Plausibility: clamp(plausibility, 0, 1),
		Uncertainty:  clamp(uncertainty, 0, 1),
	}
}

// Vacuous returns the maximally uncertain (ignorant) belief state.
func Vacuous() State {
	return State{
		Belief:       0,
		Plausibility: 1,
		Uncertainty:  1,
	}
}

// Combine merges two independent belief states using Dempster's rule of
// combination. When evidence agrees, belief increases. When evidence
// conflicts, the conflict factor K normalizes the result. High K
// indicates contradictory evidence.
func (b State) Combine(other State) State {
	// Mass assignments for b: m1(H)=belief, m1(¬H)=1-plausibility, m1(Θ)=uncertainty
	m1H := b.Belief
	m1NotH := 1 - b.Plausibility
	m1Theta := b.Uncertainty

	// Mass assignments for other
	m2H := other.Belief
	m2NotH := 1 - other.Plausibility
	m2Theta := other.Uncertainty

	// Conflict factor K
	K := m1H*m2NotH + m1NotH*m2H
	if K >= 1.0 {
		return Vacuous()
	}
	norm := 1.0 / (1.0 - K)

	// Combined masses
	combinedH := norm * (m1H*m2H + m1H*m2Theta + m1Theta*m2H)
	combinedNotH := norm * (m1NotH*m2NotH + m1NotH*m2Theta + m1Theta*m2NotH)
	combinedTheta := norm * (m1Theta * m2Theta)

	// Normalize to ensure they sum to 1
	total := combinedH + combinedNotH + combinedTheta
	if total > 0 {
		combinedH /= total
		combinedNotH /= total
		combinedTheta /= total
	}

	return State{
		Belief:       clamp(combinedH, 0, 1),
		Plausibility: clamp(combinedH+combinedTheta, 0, 1),
		Uncertainty:  clamp(combinedTheta, 0, 1),
	}
}

// Conflict returns the Dempster-Shafer conflict factor K between two belief states.
// K=0 means perfect agreement, K=1 means total contradiction.
func (b State) Conflict(other State) float64 {
	m1H := b.Belief
	m1NotH := 1 - b.Plausibility
	m2H := other.Belief
	m2NotH := 1 - other.Plausibility

	K := m1H*m2NotH + m1NotH*m2H
	return clamp(K, 0, 1)
}

// ToConfidence collapses a State back to a single confidence value
// using the pignistic probability: belief + uncertainty/2.
func (b State) ToConfidence() float64 {
	return clamp(b.Belief+b.Uncertainty/2, 0, 1)
}

// IsHighConflict returns true when conflict with another state exceeds threshold.
func (b State) IsHighConflict(other State, threshold float64) bool {
	return b.Conflict(other) > threshold
}

// CombineMultiple fuses an ordered slice of belief states left to right.
func CombineMultiple(states []State) State {
	if len(states) == 0 {
		return Vacuous()
	}
	result := states[0]
	for _, s := range states[1:] {
		result = result.Combine(s)
	}
	return result
}

// TemporalDecay applies exponential decay to a belief state based on age
// and a half-life duration (in hours). Older evidence contributes less,
// moving the state toward vacuous (ignorance increases).
func (b State) TemporalDecay(ageHours, halfLifeHours float64) State {
	if halfLifeHours <= 0 || ageHours <= 0 {
		return b
	}
	decayFactor := math.Pow(0.5, ageHours/halfLifeHours)
	return State{
		Belief:       b.Belief * decayFactor,
		Plausibility: 1 - (1-b.Plausibility)*decayFactor,
		Uncertainty:  1 - b.Belief*decayFactor - (1-b.Plausibility)*decayFactor,
	}
}

func clamp(v, lo, hi float64) float64 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}
