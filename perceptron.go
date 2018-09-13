package ann

import (
	"fmt"
	"math"
	"math/rand"
)

// Perceptron Multi-layer structure
type Perceptron struct {
	nLayers      int
	nInputs      int
	nOutputs     int
	nNeurons     int
	nConnections int
	nWeights     int

	layers       []int
	layerIndexes []int

	// Weights[i][j][k]
	// i: perceptron layer index, the weight is from :
	// j: neuron bottom layer position index (perceptron layer index) current indexes to
	// k: neuron upper layer position index (perceptron layer index + 1)
	weights [][][]float64

	weightIndexes   []int
	isWRangeDefined bool
	minWeight       float64
	maxWeight       float64

	// For back-propagation, save neuron outputs during the calculation
	// outputs[i][j] and gradients[i][j]
	// i: perceptron layer index
	// j: neuron position index in that layer.
	outputs   [][]float64
	gradients [][]float64

	// wVariations array has same indexes as weights
	wVariations [][][]float64
}

// NewPerceptron returns a new perceptron instance.
func NewPerceptron(layersList ...int) *Perceptron {
	p := &Perceptron{}

	// Include internal bias neuron for computation purposes
	for _, l := range layersList {
		p.layers = append(p.layers, l+1)
	}

	p.nLayers = len(layersList)
	p.nInputs = p.layers[0]
	p.nOutputs = p.layers[p.nLayers-1]

	p.layerIndexes = make([]int, p.nLayers+1)

	p.weightIndexes = make([]int, p.nLayers)

	p.nNeurons = p.layers[0]
	p.nWeights = 0

	nWeights := 0

	for i := 1; i < p.nLayers; i++ {
		// To make the computation easier, a "virtual" bias neuron is added
		// it will be always equal to 1.0
		p.layerIndexes[i] = p.nNeurons
		p.nNeurons += p.layers[i]
		p.nWeights += (p.layers[i] - 1) * (p.layers[i-1])
		p.weightIndexes[i] += nWeights
	}
	p.layerIndexes[p.nLayers] = p.nNeurons

	// Initialization of weights and variation
	// Initialization of the layers
	p.weights = make([][][]float64, p.nLayers)
	p.wVariations = make([][][]float64, p.nLayers)
	// Initialization of the bottom layers
	for i := 0; i < p.nLayers-1; i++ {
		p.weights[i] = make([][]float64, p.layers[i])
		p.wVariations[i] = make([][]float64, p.layers[i])

		// Initialization of the top layer
		for j := 0; j < p.layers[i]; j++ {
			p.weights[i][j] = make([]float64, p.layers[i+1])
			p.wVariations[i][j] = make([]float64, p.layers[i+1])
		}
	}

	p.outputs = make([][]float64, p.nLayers)
	p.gradients = make([][]float64, p.nLayers)

	for i := 0; i < p.nLayers; i++ {
		p.outputs[i] = make([]float64, p.layers[i])
		p.gradients[i] = make([]float64, p.layers[i])
		// Add neuron bias value
		// That value should never change
		// TODO add a test to check that
		p.outputs[i][0] = 1.0
	}

	return p
}

// SetWeights sets all the weights values in the neural network.
// By default, all the weights are set to zero.
func (p *Perceptron) MustSetWeights(weights []float64) {
	if len(weights) != p.nWeights {
		panic(fmt.Sprintf("weights size is incorrect, expected: %d, got: %d", p.nWeights, len(weights)))
	} else {
		wIndex := 0

		for i := 0; i < p.nLayers-1; i++ {
			for k := 1; k < p.layers[i+1]; k++ {
				for j := 0; j < p.layers[i]; j++ {
					p.weights[i][j][k] = weights[wIndex]
					wIndex++
				}
			}
		}
	}
}

func (p *Perceptron) MustSetRangeWeights(min, max float64) {
	if min > max {
		panic("min > max")
	}

	p.isWRangeDefined = true
	p.minWeight = min
	p.maxWeight = max
}

// MustInitRandomWeights sets all the weights values with random values
// within the provided range.
func (p *Perceptron) MustInitRandomWeights(min, max float64) {
	if min > max {
		panic("min > max")
	}

	amplitude := max - min

	// The first loop is dedicated to browse among the number of layers.
	for layerIndex := 1; layerIndex < p.nLayers; layerIndex++ {
		// The second loop will stay on the current up layer (starting in the first layer).
		for topIndex := 1; topIndex < p.layers[layerIndex]; topIndex++ {
			// The last loop will stay on the current bottom layer.
			for bottomIndex := 0; bottomIndex < p.layers[layerIndex-1]; bottomIndex++ {
				p.weights[layerIndex-1][bottomIndex][topIndex] = min + amplitude*rand.Float64()
			}
		}
	}
}

// PrintWeights will display on stdout all the weights values.
func (p *Perceptron) PrintWeights() {
	// The first loop is dedicated to browse among the number of layers.
	for layerIndex := 1; layerIndex < p.nLayers; layerIndex++ {
		// The second loop will stay on the current up layer (starting in the first layer).
		for topIndex := 1; topIndex < p.layers[layerIndex]; topIndex++ {
			// The last loop will stay on the current bottom layer.
			for bottomIndex := 0; bottomIndex < p.layers[layerIndex-1]; bottomIndex++ {
				fmt.Printf("(Layer, Neuron index) from: (%d, %d) to (%d, %d), weight: %f\n",
					layerIndex-1, bottomIndex, layerIndex, topIndex, p.weights[layerIndex-1][bottomIndex][topIndex])
			}
		}
	}
}

// PrintNeurons will display on stdout all the neuron values (included internal bias)
func (p *Perceptron) PrintNeurons() {
	// The first loop is dedicated to browse among the number of layers.
	for layerIndex := 0; layerIndex < p.nLayers; layerIndex++ {
		for neuronIndex := 0; neuronIndex < p.layers[layerIndex]; neuronIndex++ {
			fmt.Printf("(Layer, Neuron index), (%d, %d) value: %f\n",
				layerIndex, neuronIndex, p.outputs[layerIndex][neuronIndex])
		}
	}
}

// GetWeightsLen returns the number of weights.
func (p *Perceptron) GetWeightsLen() int {
	return p.nWeights
}

// GetWeights returns the weights using an 1D array.
func (p *Perceptron) GetWeights() []float64 {
	weights := make([]float64, p.nWeights)
	wIndex := 0
	// The first loop is dedicated to browse among the number of layers.
	for layerIndex := 1; layerIndex < p.nLayers; layerIndex++ {
		// The second loop will stay on the current up layer (starting in the first layer).
		for topIndex := 1; topIndex < p.layers[layerIndex]; topIndex++ {
			// The last loop will stay on the current bottom layer.
			for bottomIndex := 0; bottomIndex < p.layers[layerIndex-1]; bottomIndex++ {
				weights[wIndex] = p.weights[layerIndex-1][bottomIndex][topIndex]
				wIndex++
			}
		}
	}

	return weights
}

// Compute returns the output values of the perceptron given the values from the first layer.
// It returns the total error
func (p *Perceptron) ComputeError(inputs, target []float64) (outputs []float64, totalError float64) {
	outputs = p.Compute(inputs)

	for i := range p.outputs[p.nLayers-1][1:] {
		totalError += math.Pow((target[i] - p.outputs[p.nLayers-1][i+1]), 2)
	}

	return outputs, 0.5 * totalError
}

// Compute returns the output values of the perceptron given the values from the first layer.
func (p *Perceptron) Compute(inputs []float64) (outputs []float64) {
	// First, we already know the values of the input layer
	// Let's note that bias neuron is already set during the initialization
	for i := 1; i < p.nInputs; i++ {
		p.outputs[0][i] = inputs[i-1]
	}

	// The first loop is dedicated to browse among the number of layers.
	for layerIndex := 1; layerIndex < p.nLayers; layerIndex++ {

		// The second loop will stay on the current up layer (starting in the first layer).
		for topIndex := 1; topIndex < p.layers[layerIndex]; topIndex++ {
			sum := 0.0

			// The last loop will stay on the current bottom layer.
			for bottomIndex := 0; bottomIndex < p.layers[layerIndex-1]; bottomIndex++ {
				sum += p.weights[layerIndex-1][bottomIndex][topIndex] * p.outputs[layerIndex-1][bottomIndex]
			}

			// TODO let the user choose the activation function.
			p.outputs[layerIndex][topIndex] = sigmoid(sum)
		}
	}
	return p.outputs[p.nLayers-1][1:]
}

// BackPropagation function uses the following convention:
// The target outputs and the current outputs are taken in parameters
// The weights are then modified.
//     .----.       .----.       .----.
//    | bias |     |      |     |      |   Layer k
//    |  1   |     |      |     |      |
//     '----'      /'----'\     /'----'\
//                /    |   \        |
//               /     |    \
//              /      |     \
//             /       |      \
//            /        |       \
//     .----./      .----.      \.----.
//    | bias |     |      |     |      |   Layer j
//    |  1   |     |      |     |      |
//     '----'      /'----'\     /'----'\
//                /    |   \        |
//               /     |    \
//              /      |     \
//             /       |      \
//            /        |       \
//     .----./      .----.      \.----.
//    | bias |     |      |     |      |   Layer i
//    |  1   |     |      |     |      |
//     '----'       '----'       '----'
//
func (p *Perceptron) BackPropagation(targetOutputs []float64) {

	if len(targetOutputs) != len(p.outputs[p.nLayers-1][1:]) {
		panic("Back-propagation, length of the targetOutputs and outputs are different")
	}

	// We are using these notations:
	// i index is used for the input layer
	// j index is used for the hidden layer
	// k index is used for the output layer

	//TODO set alpha for inertial rate
	//TODO set learning-rate
	learningRate := 0.1
	alphaRate := 0.5

	// First step in the back-propagation algorithm, we need to compute the delta on the output layer (k)
	// Bias "neuron" index 0
	for k := 1; k < p.nOutputs; k++ {
		// We don't take the bias neuron
		outputk := p.outputs[p.nLayers-1][k]

		p.gradients[p.nLayers-1][k] = -(targetOutputs[k-1] - outputk) * (outputk * (1.0 - outputk))
	}

	// We now back-propagate the gradient all of neurons in reverse order: from the upper layer to the bottom layers.
	for iLayer := p.nLayers - 2; iLayer >= 1; iLayer-- {
		// Bias neuron 0: computing the gradient of the middle layer, index j
		for j := 1; j < p.layers[iLayer]; j++ {

			p.gradients[iLayer][j] = 0
			// gradient neuron i: gradient upper layer k * weights linked to the middle layer i * value_neuron_i * (1 - value_neuron_i)
			for k := 1; k < p.layers[iLayer+1]; k++ {

				p.gradients[iLayer][j] += p.gradients[iLayer+1][k] * p.weights[iLayer][j][k]
			}

			output := p.outputs[iLayer][j]
			p.gradients[iLayer][j] *= output * (1.0 - output)
		}
	}

	// Compute the weights variation in reverse order: from the upper layer to the bottom layers.
	for iLayer := p.nLayers - 1; iLayer >= 1; iLayer-- {
		// Bias neuron 0: computing the gradient of the middle layer, index j
		for j := 1; j < p.layers[iLayer]; j++ {

			// The i layer gradients are computed
			// Now, computation of the weights delta and modification to the weight
			for i := 0; i < p.layers[iLayer-1]; i++ {

				var gradient float64 = p.gradients[iLayer][j] * p.outputs[iLayer-1][i]
				var variation float64 = -(1-alphaRate)*learningRate*gradient + (alphaRate * p.wVariations[iLayer-1][i][j])

				w := p.weights[iLayer-1][i][j]

				w += variation

				if p.isWRangeDefined {
					if w < p.minWeight {
						w = p.minWeight
					} else if w > p.maxWeight {
						w = p.maxWeight
					}
				}
				p.wVariations[iLayer-1][i][j] = variation
				p.weights[iLayer-1][i][j] = w
			}
		}
	}
}
