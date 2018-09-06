package ann

type feedforward interface {
	Compute(weights []float64, inputs []float64) []float64
	GetWeigthsLen()
}
