package ann

type feedforward interface {
	MustSetWeights(weights []float64)
	Compute(inputs []float64) []float64
	GetWeigthsLen() int
}
