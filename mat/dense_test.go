package mat_test

import (
	"fmt"
	"math/rand"
	"reflect"
	"testing"

	"github.com/nfisher/goalgo/mat"
	"gonum.org/v1/gonum/blas/blas64"
	gnmat "gonum.org/v1/gonum/mat"
)

var ff = []struct {
	name    string
	product func(c, a, b *mat.Dense) error
}{
	// WIP {"daxpy", mat.DotDaxpy},
	{"gonum mulprefetch", mat.MulMultiplePrefetch2},
	{"gonum stride", mat.MulGonumStride},
	{"stride", mat.MulStride},
	{"naive", mat.MulNaive},
	{"gonum interface", mat.MulGonumNaive},
	{"gonum prefetch", mat.MulGonumNaivePrefetch},
}

func Test_Product(t *testing.T) {
	td := []struct {
		name     string
		a        *mat.Dense
		b        *mat.Dense
		expected []float64
		err      error
	}{
		{"[3,2] x [2,1]", mat.NewDense(3, 2, []float64{1, 2, 3, 4, 5, 6}), mat.NewDense(2, 1, []float64{2, 4}), []float64{10, 22, 34}, nil},
		{"[4,3] x [3,2]", mat.NewDense(4, 3, []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}), mat.NewDense(3, 2, []float64{2, 7, 1, 2, 3, 6}), []float64{13, 29, 31, 74, 49, 119, 67, 164}, nil},
		{"[8,8] x [8,8]", eightByEight, eightByEight, eightByEightProduct, nil},
		{"[1,10] x [10,1]", mat.NewDense(1, 10, vecTen), mat.NewDense(10, 1, vecTen), []float64{385}, nil},
		{"[1,9] x [9,1]", mat.NewDense(1, 9, vecTen[0:9]), mat.NewDense(9, 1, vecTen[0:9]), []float64{285}, nil},
		{"[1,16] x [16,1]", mat.NewDense(1, 16, vecSixteen), mat.NewDense(16, 1, vecSixteen), []float64{1496}, nil},
	}

	for _, f := range ff {
		for _, tc := range td {
			t.Run(f.name+" "+tc.name, func(t *testing.T) {
				c := mat.NewDense(tc.a.Rows(), tc.b.Columns(), make([]float64, tc.a.Rows()*tc.b.Columns()))
				err := f.product(c, tc.a, tc.b)
				if err != tc.err {
					t.Fatalf("err = %v, want %v", err, tc.err)
				}

				if !reflect.DeepEqual(c.Raw(), tc.expected) {
					t.Errorf("raw = %v, want %v", c.Raw(), tc.expected)
				}
			})
		}
	}
}

func Benchmark_Dot(b *testing.B) {
	aMat := mat.NewDense(1, 1024, aArr[:1024])
	bMat := mat.NewDense(1024, 1, bArr[:1024])
	cMat := mat.NewDense(1, 1, make([]float64, 1))

	for _, f := range ff {
		b.Run(f.name, func(b *testing.B) {
			for n := 0; n < b.N; n++ {
				_ = f.product(cMat, aMat, bMat)
			}
			DotResult = cMat
		})
	}
}

func Benchmark_Product(b *testing.B) {
	var sizes = sizes
	if testing.Short() {
		sizes = []int{1024}
	}
	for _, sz := range sizes {
		dim := sz * sz
		aMat := mat.NewDense(sz, sz, aArr[:dim])
		bMat := mat.NewDense(sz, sz, bArr[:dim])
		cMat := mat.NewDense(sz, sz, make([]float64, dim))

		for _, f := range ff {
			b.Run(fmt.Sprintf("%s @n=%v", f.name, sz), func(b *testing.B) {
				for n := 0; n < b.N; n++ {
					_ = f.product(cMat, aMat, bMat)
				}
				DotResult = cMat
			})
		}
	}
}

// 16214216899
// 7259547830 naive - ouch have lots of optimisation to do...
// 126482249 gonum
// 146958870
func Benchmark_ProductGonum(b *testing.B) {
	var sizes = sizes
	if testing.Short() {
		sizes = []int{1024}
	}
	for _, sz := range sizes {
		dim := sz * sz
		aMat := gnmat.NewDense(sz, sz, aArr[:dim])
		bMat := gnmat.NewDense(sz, sz, bArr[:dim])
		cMat := gnmat.NewDense(sz, sz, nil)

		b.Run(fmt.Sprintf("@n=%v", sz), func(b *testing.B) {
			for n := 0; n < b.N; n++ {
				cMat.Product(aMat, bMat)
			}
		})
		Result = cMat.RawMatrix()
	}
}

func Test_GonumProduct(t *testing.T) {
	a := gnmat.NewDense(8, 8, eightByEight.Raw())
	b := gnmat.NewDense(8, 8, eightByEightTwo.Raw())
	c := gnmat.NewDense(8, 8, make([]float64, 64))

	c.Product(a, b)
	actual := c.RawMatrix().Data
	if !reflect.DeepEqual(actual, eightByEightProduct) {
		t.Errorf("got %v, want %v", actual, eightByEightProduct)
	}
}

var (
	DotResult *mat.Dense
	Result    blas64.General
	aArr      [2048 * 2048]float64
	bArr      [2048 * 2048]float64
)

var sizes = []int{
	64,
	128,
	256,
	512,
	1024,
	2048,
}

var eightByEight = mat.NewDense(8, 8, []float64{1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8})
var eightByEightTwo = mat.NewDense(8, 8, []float64{1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8})
var eightByEightProduct = []float64{36, 72, 108, 144, 180, 216, 252, 288, 36, 72, 108, 144, 180, 216, 252, 288, 36, 72, 108, 144, 180, 216, 252, 288, 36, 72, 108, 144, 180, 216, 252, 288, 36, 72, 108, 144, 180, 216, 252, 288, 36, 72, 108, 144, 180, 216, 252, 288, 36, 72, 108, 144, 180, 216, 252, 288, 36, 72, 108, 144, 180, 216, 252, 288}
var vecTen = []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
var vecSixteen = []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}

func init() {
	rand.Seed(173)

	for i := range aArr {
		aArr[i] = rand.NormFloat64()
		bArr[i] = rand.NormFloat64()
	}
}
