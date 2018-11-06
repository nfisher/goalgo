package mat_test

import (
	"math/rand"
	"reflect"
	"testing"

	"github.com/nfisher/goalgo/mat"
)

var eightByEight = mat.NewDense(8, 8, []float64{1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8})
var eightByEightTwo = mat.NewDense(8, 8, []float64{1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8})
var eightByEightProduct = []float64{36, 72, 108, 144, 180, 216, 252, 288, 36, 72, 108, 144, 180, 216, 252, 288, 36, 72, 108, 144, 180, 216, 252, 288, 36, 72, 108, 144, 180, 216, 252, 288, 36, 72, 108, 144, 180, 216, 252, 288, 36, 72, 108, 144, 180, 216, 252, 288, 36, 72, 108, 144, 180, 216, 252, 288, 36, 72, 108, 144, 180, 216, 252, 288}
var vecTen = []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

func Test_Dot(t *testing.T) {
	td := []struct {
		name     string
		a        *mat.Dense
		b        *mat.Dense
		expected []float64
		err      error
	}{
		{"[3x2] x [2x1]", mat.NewDense(3, 2, []float64{1, 2, 3, 4, 5, 6}), mat.NewDense(2, 1, []float64{2, 4}), []float64{10, 22, 34}, nil},
		{"[4x3] x [3x2]", mat.NewDense(4, 3, []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}), mat.NewDense(3, 2, []float64{2, 7, 1, 2, 3, 6}), []float64{13, 29, 31, 74, 49, 119, 67, 164}, nil},
		{"[8x8] x [8x8]", eightByEight, eightByEight, eightByEightProduct, nil},
		{"[1x10] x [10x1]", mat.NewDense(1, 10, vecTen), mat.NewDense(10, 1, vecTen), []float64{385}, nil},
		{"[1x9] x [9x1]", mat.NewDense(1, 9, vecTen[0:9]), mat.NewDense(9, 1, vecTen[0:9]), []float64{285}, nil},
	}

	for _, tc := range td {
		t.Run(tc.name, func(t *testing.T) {
			c, err := mat.Dot(tc.a, tc.b)
			if err != tc.err {
				t.Fatalf("err = %v, want %v", err, tc.err)
			}

			if !reflect.DeepEqual(c.Raw(), tc.expected) {
				t.Errorf("raw = %v, want %v", c.Raw(), tc.expected)
			}
		})
	}
}

func Benchmark_DotSmall(b *testing.B) {
	var c *mat.Dense
	for n := 0; n < b.N; n++ {
		c, _ = mat.Dot(eightByEight, eightByEightTwo)
	}
	result = c
}

func Benchmark_DotLarge(b *testing.B) {
	var aArr [1000000]float64
	var bArr [1000000]float64

	rand.Seed(173)
	for i := range aArr {
		aArr[i] = rand.NormFloat64()
		bArr[i] = rand.NormFloat64()
	}
	aMat := mat.NewDense(1000, 1000, aArr[:])
	bMat := mat.NewDense(1000, 1000, bArr[:])

	b.Run("million dollar matrix", func(b *testing.B) {
		var c *mat.Dense
		for n := 0; n < b.N; n++ {
			c, _ = mat.Dot(aMat, bMat)
		}
		result = c
	})
}

var result *mat.Dense
