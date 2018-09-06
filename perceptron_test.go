package ann

import (
	"testing"
)

func TestSetWeights(t *testing.T) {

	nn := NewPerceptron(3, 5, 3)

	// The length of weights is set to an incorrect value.
	weights := make([]float64, nn.GetWeightsLen()-1)

	if err := nn.SetWeights(weights); err == nil {
		t.Fatalf("Expected error but got no error")
	}
}

// Compute function
func TestSinus(t *testing.T) {
	//TODO
}

// Compute function
func TestCosinus(t *testing.T) {
	//TODO
}

func BenchmarkCompute(b *testing.B) {

	nn := NewPerceptron(3, 5, 5, 3)

	for i := 0; i < b.N; i++ {
		nn.Compute([]float64{0.5, 0.5, 0.5})
	}
}
