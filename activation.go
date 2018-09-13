package ann

import "math"

// Activation functions for the neural networks

func sigmoid(input float64) float64 {
	return float64(1.0) / (1.0 + math.Exp(-input))
}
