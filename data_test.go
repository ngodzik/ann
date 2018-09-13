package ann

import (
	"math"
)

const max = 2 * math.Pi
const step = max / 20

// Sine function, normalization of the data for x inputs
var sinTable = []struct {
	X float64
	Y float64
}{
	{0, 0},
	{step / max, (math.Sin(step) + 1) / 2},
	{2 * step / max, (math.Sin(2*step) + 1) / 2},
	{3 * step / max, (math.Sin(3*step) + 1) / 2},
	{4 * step / max, (math.Sin(4*step) + 1) / 2},
	{5 * step / max, (math.Sin(5*step) + 1) / 2},
	{6 * step / max, (math.Sin(6*step) + 1) / 2},
	{7 * step / max, (math.Sin(7*step) + 1) / 2},
	{8 * step / max, (math.Sin(8*step) + 1) / 2},
	{9 * step / max, (math.Sin(9*step) + 1) / 2},
	{10 * step / max, (math.Sin(10*step) + 1) / 2},
	{11 * step / max, (math.Sin(11*step) + 1) / 2},
	{12 * step / max, (math.Sin(12*step) + 1) / 2},
	{13 * step / max, (math.Sin(13*step) + 1) / 2},
	{14 * step / max, (math.Sin(14*step) + 1) / 2},
	{15 * step / max, (math.Sin(15*step) + 1) / 2},
	{16 * step / max, (math.Sin(16*step) + 1) / 2},
	{17 * step / max, (math.Sin(17*step) + 1) / 2},
	{18 * step / max, (math.Sin(18*step) + 1) / 2},
	{19 * step / max, (math.Sin(19*step) + 1) / 2},
	{1, (math.Sin(max) + 1) / 2},
}

// Cosine function, normalization of the data for x inputs
var cosTable = []struct {
	x float64
	y float64
}{
	{0, 0},
	{step / max, (math.Cos(step) + 1)},
	{2 * step / max, (math.Cos(2*step) + 1) / 2},
	{3 * step / max, (math.Cos(3*step) + 1) / 2},
	{4 * step / max, (math.Cos(4*step) + 1) / 2},
	{5 * step / max, (math.Cos(5*step) + 1) / 2},
	{6 * step / max, (math.Cos(6*step) + 1) / 2},
	{7 * step / max, (math.Cos(7*step) + 1) / 2},
	{8 * step / max, (math.Cos(8*step) + 1) / 2},
	{9 * step / max, (math.Cos(9*step) + 1) / 2},
	{10 * step / max, (math.Cos(10*step) + 1) / 2},
	{11 * step / max, (math.Cos(11*step) + 1) / 2},
	{12 * step / max, (math.Cos(12*step) + 1) / 2},
	{13 * step / max, (math.Cos(13*step) + 1) / 2},
	{14 * step / max, (math.Cos(14*step) + 1) / 2},
	{15 * step / max, (math.Cos(15*step) + 1) / 2},
	{16 * step / max, (math.Cos(16*step) + 1) / 2},
	{17 * step / max, (math.Cos(17*step) + 1) / 2},
	{18 * step / max, (math.Cos(18*step) + 1) / 2},
	{19 * step / max, (math.Cos(19*step) + 1) / 2},
	{1, (math.Cos(max) + 1) / 2},
}
