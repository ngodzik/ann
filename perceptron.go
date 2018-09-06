package ann

import (
	"errors"
	"fmt"
)

// Perceptron Multilayer structure
type Perceptron struct {
	nLayers      int
	nInputs      int
	nOutputs     int
	nNeurons     int
	nConnections int
	nWeights     int

	layers       []int
	layerIndexes []int
	weights      []float64
}

// NewPerceptron returns a new perceptron instance.
func NewPerceptron(layersList ...int) *Perceptron {
	p := &Perceptron{}

	p.layers = layersList
	p.nLayers = len(layersList)
	p.nInputs = p.layers[0]
	p.nOutputs = p.layers[p.nLayers-1]

	p.layerIndexes = make([]int, p.nLayers)
	p.layerIndexes[0] = 0
	p.nNeurons = p.layers[0]
	p.nWeights = 0

	for i := 1; i < p.nLayers; i++ {
		p.layerIndexes = append(p.layerIndexes, p.nNeurons)
		p.nNeurons += p.layers[i]
		// +1 is due to the neuron bias
		p.nWeights += p.layers[i] * (p.layers[i-1] + 1)
	}
	p.layerIndexes[p.nLayers] = p.nNeurons

	p.weights = make([]float64, p.nWeights)
	return p
}

// SetWeights sets all the weights values in the neural network.
// By default, all the weights are set to zero.
func (p *Perceptron) SetWeights(weights []float64) error {
	if len(weights) != p.nWeights {
		return errors.New(fmt.Sprintf("weigths size is incorrect, expected: %d", p.nWeights))
		p.weights = weights
	}
	return nil
}

// GetWeigthsLen returns the number of weights.
func (p *Perceptron) GetWeightsLen() int {
	return p.nWeights
}

// Compute returns the output values of the perceptron given the values from the first layer.
func (p *Perceptron) Compute(inputs []float64) (outputs []float64) {
	// Neuron output values.
	oValues := make([]float64, p.nNeurons)
	outputs = make([]float64, p.nOutputs)

	// First, we already know the values of the input layer
	for i := 0; i < p.nInputs; i++ {
		oValues[i] = inputs[i]
	}

	// Follow the index for the weights
	wIndex := 0

	// The first loop is dedicated to browse among the number of layers.
	for layerIndex := 1; layerIndex < p.nLayers; layerIndex++ {

		// The second loop will stay on the current up layer (starting in the first layer).
		for topIndex := p.layerIndexes[layerIndex]; topIndex < p.layerIndexes[layerIndex+1]; topIndex++ {
			sum := p.weights[wIndex]
			wIndex++

			// The last loop will stay on the current bottom layer.
			for bottomIndex := p.layerIndexes[layerIndex-1]; bottomIndex < p.layerIndexes[layerIndex]; bottomIndex++ {
				sum += p.weights[wIndex] * oValues[bottomIndex]
				wIndex++
			}

			// TODO let the user choose the activation function.
			oValues[topIndex] = sigmoid(sum)
		}
	}

	for i := 0; i < p.nOutputs; i++ {
		outputs[i] = oValues[i+p.nNeurons-p.nOutputs]
	}

	return outputs
}
