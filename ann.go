package ann

type feedforward interface {
	SetWeights(weights []float64)
	Compute(inputs []float64) []float64
	GetWeigthsLen()
}
