package mat_test

import (
	"fmt"
	"math/rand"
	"reflect"
	"runtime"
	"testing"

	"github.com/nfisher/goalgo/mat"
	"gonum.org/v1/gonum/blas/blas64"
	gnmat "gonum.org/v1/gonum/mat"
)

func Test_QuadAxpy(t *testing.T) {
	c := make([]float64, 4)
	b := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
	mat.QuadAxpy(c, b[:4], 2, 0, 0)
	expected := []float64{2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32}
	if !reflect.DeepEqual(c, expected[:4]) {
		t.Errorf("quadaxpy = %v, want %v", c, expected[:4])
	}
}

func Test_HexAxpy(t *testing.T) {
	c := make([]float64, 8)
	b := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
	mat.HexAxpy(c, b[:len(c)], 2, 0, 0)
	expected := []float64{2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32}
	if !reflect.DeepEqual(c, expected[:len(c)]) {
		t.Errorf("quadaxpy = %v, want %v", c, expected[:len(c)])
	}
}

func Test_HexadecAxpy(t *testing.T) {
	c := make([]float64, 16)
	b := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
	mat.HexadecAxpy(c, b[:len(c)], 2, 0, 0)
	expected := []float64{2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32}
	if !reflect.DeepEqual(c, expected[:len(c)]) {
		t.Errorf("quadaxpy = %v, want %v", c, expected[:len(c)])
	}
}

func Test_DenseWithStride(t *testing.T) {
	d := mat.NewDenseWithStride(2, 3, 5, nil)
	if got, want := d.Stride(), 5; got != want {
		t.Fatalf("stride = %v, want %v", got, want)
	}
	d.Set(1, 2, 42)
	if got, want := d.At(1, 2), float64(42); got != want {
		t.Fatalf("At(1,2) = %v, want %v", got, want)
	}
	if got, want := d.Raw()[5+2], float64(42); got != want {
		t.Fatalf("raw stride value = %v, want %v", got, want)
	}
}

func Test_DenseView(t *testing.T) {
	d := mat.NewDense(3, 4, []float64{
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
	})
	v := d.View(1, 1, 2, 2)
	if got, want := v.Stride(), d.Stride(); got != want {
		t.Fatalf("view stride = %v, want %v", got, want)
	}
	if got, want := v.At(1, 1), float64(11); got != want {
		t.Fatalf("view At(1,1) = %v, want %v", got, want)
	}
	v.Set(0, 1, 70)
	if got, want := d.At(1, 2), float64(70); got != want {
		t.Fatalf("base At(1,2) = %v, want %v", got, want)
	}
}

var ff = []struct {
	name    string
	product func(c, a, b *mat.Dense) error
}{
	{"simd ijk", mat.MulSimdIJK},
	{"blocked gemm", mat.MulBlockedGemm},
	{"dot transposed", mat.MulDotTransposed},
	{"dot transposed simd", mat.MulDotTransposedSimd},
	{"block8 ijk", mat.MulBlockIJK(8)},
	{"block16 ijk", mat.MulBlockIJK(16)},
	{"bunroll8 ijk", mat.MulBlockUnrollIJK(8)},
	{"bunroll16 ijk", mat.MulBlockUnrollIJK(16)},
	{"bunroll32 ijk", mat.MulBlockUnrollIJK(32)},
	{"bunroll64 ijk", mat.MulBlockUnrollIJK(64)},
	{"gaxpy", mat.MulGaxpy},
	/* Commented out tests that benchmark over 1.5s
	{"block2 ijk", mat.MulBlockIJK(2)},
	{"block4 ijk", mat.MulBlockIJK(4)},
	{"block32 ijk", mat.MulBlockIJK(32)},
	{"block64 ijk", mat.MulBlockIJK(64)},
	{"bunroll4 ijk", mat.MulBlockUnrollIJK(4)},
	{"blockfetch2 ijk", mat.MulBlockFetchIJK(2)},
	{"blockfetch4 ijk", mat.MulBlockFetchIJK(4)},
	{"blockfetch8 ijk", mat.MulBlockFetchIJK(8)},
	{"blockfetch16 ijk", mat.MulBlockFetchIJK(16)},
	{"blockfetch32 ijk", mat.MulBlockFetchIJK(32)},
	{"blockfetch64 ijk", mat.MulBlockFetchIJK(64)},
	{"mulprefetch", mat.MulMultiplePrefetch2},
	{"gni prefetch", mat.MulGonumNaivePrefetch},
	{"gni unroll", mat.MulGonumUnroll},
	{"gni", mat.MulGonumNaive},
	{"unroll", mat.MulUnroll},
	{"naive IKJ", mat.MulNaiveIKJ},
	{"naive IJK", mat.MulNaiveIJK},
	{"naive JIK", mat.MulNaiveJIK},
	{"naive JKI", mat.MulNaiveJKI},
	{"naive KIJ", mat.MulNaiveKIJ},
	{"naive KJI", mat.MulNaiveKJI},
	*/
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
		{"[16,16] x [16,16]", mat.NewDense(16, 16, matSixteen), mat.NewDense(16, 16, matSixteen), matSixteenProduct, nil},
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

func Test_ProductStrided(t *testing.T) {
	a := mat.NewDenseWithStride(2, 3, 5, []float64{
		1, 2, 3, 0, 0,
		4, 5, 6, 0, 0,
	})
	b := mat.NewDenseWithStride(3, 2, 4, []float64{
		7, 8, 0, 0,
		9, 10, 0, 0,
		11, 12, 0, 0,
	})
	expected := []float64{58, 64, 139, 154}

	for _, f := range ff {
		t.Run(f.name, func(t *testing.T) {
			c := mat.NewDenseWithStride(2, 2, 4, nil)
			err := f.product(c, a, b)
			if err != nil {
				t.Fatalf("err = %v, want nil", err)
			}
			for i, want := range expected {
				row := i / 2
				col := i % 2
				if got := c.At(row, col); got != want {
					t.Fatalf("At(%d,%d) = %v, want %v", row, col, got, want)
				}
			}
		})
	}
}

func Test_MulBlockedGemmColumnTails(t *testing.T) {
	td := []struct {
		name string
		rows int
		k    int
		cols int
		pad  int
	}{
		{"4x8 compact", 4, 5, 8, 0},
		{"4x9 compact", 4, 5, 9, 0},
		{"4x12 compact", 4, 5, 12, 0},
		{"5x13 compact", 5, 7, 13, 0},
		{"4x8 split k", 4, 2051, 8, 0},
		{"4x9 strided", 4, 5, 9, 3},
		{"5x13 strided", 5, 7, 13, 3},
	}

	for _, tc := range td {
		t.Run(tc.name, func(t *testing.T) {
			a := newPatternDense(tc.rows, tc.k, tc.k+tc.pad)
			b := newPatternDense(tc.k, tc.cols, tc.cols+tc.pad)
			got := mat.NewDenseWithStride(tc.rows, tc.cols, tc.cols+tc.pad, nil)
			want := mat.NewDenseWithStride(tc.rows, tc.cols, tc.cols+tc.pad, nil)

			if err := mat.MulGonumNaive(want, a, b); err != nil {
				t.Fatalf("MulGonumNaive err = %v, want nil", err)
			}
			for _, product := range []struct {
				name string
				fn   func(c, a, b *mat.Dense) error
			}{
				{"blocked", mat.MulBlockedGemm},
				{"packed", mat.MulBlockedGemmPacked},
			} {
				t.Run(product.name, func(t *testing.T) {
					if err := product.fn(got, a, b); err != nil {
						t.Fatalf("%s err = %v, want nil", product.name, err)
					}
					for r := 0; r < tc.rows; r++ {
						for c := 0; c < tc.cols; c++ {
							gotValue := got.At(r, c)
							wantValue := want.At(r, c)
							if gotValue != wantValue {
								t.Fatalf("At(%d,%d) = %v, want %v", r, c, gotValue, wantValue)
							}
						}
					}
				})
			}
		})
	}
}

func Test_MulPackedDotTransposedSimd(t *testing.T) {
	td := []struct {
		name     string
		a        *mat.Dense
		b        *mat.Dense
		expected []float64
	}{
		{"[3,2] x [2,1]", mat.NewDense(3, 2, []float64{1, 2, 3, 4, 5, 6}), mat.NewDense(2, 1, []float64{2, 4}), []float64{10, 22, 34}},
		{"[4,3] x [3,2]", mat.NewDense(4, 3, []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}), mat.NewDense(3, 2, []float64{2, 7, 1, 2, 3, 6}), []float64{13, 29, 31, 74, 49, 119, 67, 164}},
		{"[16,16] x [16,16]", mat.NewDense(16, 16, matSixteen), mat.NewDense(16, 16, matSixteen), matSixteenProduct},
	}

	for _, tc := range td {
		t.Run(tc.name, func(t *testing.T) {
			packed := mat.PackBTransposed(tc.b)
			c := mat.NewDense(tc.a.Rows(), tc.b.Columns(), make([]float64, tc.a.Rows()*tc.b.Columns()))
			err := mat.MulPackedDotTransposedSimd(c, tc.a, packed)
			if err != nil {
				t.Fatalf("err = %v, want nil", err)
			}
			if !reflect.DeepEqual(c.Raw(), tc.expected) {
				t.Errorf("raw = %v, want %v", c.Raw(), tc.expected)
			}
		})
	}
}

func Test_MulPackedGemm(t *testing.T) {
	td := []struct {
		name string
		rows int
		k    int
		cols int
		pad  int
	}{
		{"compact", 8, 17, 10, 0},
		{"strided", 8, 17, 10, 3},
		{"split k", 8, 33, 10, 0},
	}

	for _, tc := range td {
		t.Run(tc.name, func(t *testing.T) {
			a := newPatternDense(tc.rows, tc.k, tc.k+tc.pad)
			b := newPatternDense(tc.k, tc.cols, tc.cols+tc.pad)
			got := mat.NewDenseWithStride(tc.rows, tc.cols, tc.cols+tc.pad, nil)
			want := mat.NewDenseWithStride(tc.rows, tc.cols, tc.cols+tc.pad, nil)

			if err := mat.MulGonumNaive(want, a, b); err != nil {
				t.Fatalf("MulGonumNaive err = %v, want nil", err)
			}
			packed := mat.PackGemmBWithBlocks(b, 16, 6)
			if err := mat.MulPackedGemm(got, a, packed); err != nil {
				t.Fatalf("MulPackedGemm err = %v, want nil", err)
			}
			for r := 0; r < tc.rows; r++ {
				for c := 0; c < tc.cols; c++ {
					if gotValue, wantValue := got.At(r, c), want.At(r, c); gotValue != wantValue {
						t.Fatalf("At(%d,%d) = %v, want %v", r, c, gotValue, wantValue)
					}
				}
			}
		})
	}
}

func Test_MulPackedDotTransposedSimdSmallRowsWide(t *testing.T) {
	previous := runtime.GOMAXPROCS(4)
	defer runtime.GOMAXPROCS(previous)

	const (
		aRows = 1
		aCols = 512
		bCols = 512
	)

	aData := make([]float64, aRows*aCols)
	bData := make([]float64, aCols*bCols)
	for i := range aData {
		aData[i] = float64(i%7 - 3)
	}
	for i := range bData {
		bData[i] = float64(i%11 - 5)
	}

	a := mat.NewDense(aRows, aCols, aData)
	b := mat.NewDense(aCols, bCols, bData)
	want := mat.NewDense(aRows, bCols, make([]float64, aRows*bCols))
	if err := mat.MulGaxpy(want, a, b); err != nil {
		t.Fatalf("MulGaxpy err = %v, want nil", err)
	}

	got := mat.NewDense(aRows, bCols, make([]float64, aRows*bCols))
	if err := mat.MulPackedDotTransposedSimd(got, a, mat.PackBTransposed(b)); err != nil {
		t.Fatalf("MulPackedDotTransposedSimd err = %v, want nil", err)
	}
	if !reflect.DeepEqual(got.Raw(), want.Raw()) {
		t.Errorf("raw = %v, want %v", got.Raw(), want.Raw())
	}
}

func Test_MulBlockedGemmSmallRowsWide(t *testing.T) {
	previous := runtime.GOMAXPROCS(4)
	defer runtime.GOMAXPROCS(previous)

	const (
		aRows = 3
		aCols = 256
		bCols = 256
	)

	a := newPatternDense(aRows, aCols, aCols+3)
	b := newPatternDense(aCols, bCols, bCols+5)
	got := mat.NewDenseWithStride(aRows, bCols, bCols+7, nil)
	want := mat.NewDenseWithStride(aRows, bCols, bCols+7, nil)

	if err := mat.MulGonumNaive(want, a, b); err != nil {
		t.Fatalf("MulGonumNaive err = %v, want nil", err)
	}
	if err := mat.MulBlockedGemm(got, a, b); err != nil {
		t.Fatalf("MulBlockedGemm err = %v, want nil", err)
	}
	for r := 0; r < aRows; r++ {
		for c := 0; c < bCols; c++ {
			if gotValue, wantValue := got.At(r, c), want.At(r, c); gotValue != wantValue {
				t.Fatalf("At(%d,%d) = %v, want %v", r, c, gotValue, wantValue)
			}
		}
	}
}

func Benchmark_Dot(b *testing.B) {
	if testing.Short() {
		b.Skip()
	}
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
			b.Run(fmt.Sprintf("%s %v", f.name, sz), func(b *testing.B) {
				for n := 0; n < b.N; n++ {
					_ = f.product(cMat, aMat, bMat)
				}
				DotResult = cMat
			})
		}
	}
}

func Benchmark_ProductStrided(b *testing.B) {
	var sizes = sizes
	if testing.Short() {
		sizes = []int{1024}
	}
	funcs := []struct {
		name    string
		product func(c, a, b *mat.Dense) error
	}{
		{"blocked gemm", mat.MulBlockedGemm},
		{"dot transposed simd", mat.MulDotTransposedSimd},
		{"gni", mat.MulGonumNaive},
	}

	for _, sz := range sizes {
		stride := sz + 8
		aMat := newStridedBenchmarkDense(sz, sz, stride, aArr[:])
		bMat := newStridedBenchmarkDense(sz, sz, stride, bArr[:])
		cMat := mat.NewDenseWithStride(sz, sz, stride, nil)

		for _, f := range funcs {
			b.Run(fmt.Sprintf("%s %v", f.name, sz), func(b *testing.B) {
				for n := 0; n < b.N; n++ {
					_ = f.product(cMat, aMat, bMat)
				}
				DotResult = cMat
			})
		}
	}
}

func Benchmark_ProductBlockedPacked(b *testing.B) {
	var sizes = sizes
	if testing.Short() {
		sizes = []int{1024}
	}
	for _, sz := range sizes {
		dim := sz * sz
		aMat := mat.NewDense(sz, sz, aArr[:dim])
		bMat := mat.NewDense(sz, sz, bArr[:dim])
		cMat := mat.NewDense(sz, sz, make([]float64, dim))

		b.Run(fmt.Sprintf("compact %v", sz), func(b *testing.B) {
			for n := 0; n < b.N; n++ {
				_ = mat.MulBlockedGemmPacked(cMat, aMat, bMat)
			}
			DotResult = cMat
		})
	}
}

func Benchmark_ProductBlockedPackedStrided(b *testing.B) {
	var sizes = sizes
	if testing.Short() {
		sizes = []int{1024}
	}
	for _, sz := range sizes {
		stride := sz + 8
		aMat := newStridedBenchmarkDense(sz, sz, stride, aArr[:])
		bMat := newStridedBenchmarkDense(sz, sz, stride, bArr[:])
		cMat := mat.NewDenseWithStride(sz, sz, stride, nil)

		b.Run(fmt.Sprintf("strided %v", sz), func(b *testing.B) {
			for n := 0; n < b.N; n++ {
				_ = mat.MulBlockedGemmPacked(cMat, aMat, bMat)
			}
			DotResult = cMat
		})
	}
}

func Benchmark_ProductSmallRowsWide(b *testing.B) {
	const (
		rows = 3
		k    = 512
		cols = 512
	)

	aMat := mat.NewDense(rows, k, aArr[:rows*k])
	bMat := mat.NewDense(k, cols, bArr[:k*cols])
	cMat := mat.NewDense(rows, cols, make([]float64, rows*cols))

	funcs := []struct {
		name    string
		product func(c, a, b *mat.Dense) error
	}{
		{"blocked gemm", mat.MulBlockedGemm},
		{"dot transposed simd", mat.MulDotTransposedSimd},
		{"gni", mat.MulGonumNaive},
	}

	for _, f := range funcs {
		b.Run(f.name, func(b *testing.B) {
			for n := 0; n < b.N; n++ {
				_ = f.product(cMat, aMat, bMat)
			}
			DotResult = cMat
		})
	}
}

func Benchmark_ProductReusablePackedGemm(b *testing.B) {
	var sizes = sizes
	if testing.Short() {
		sizes = []int{1024}
	}
	for _, sz := range sizes {
		dim := sz * sz
		aMat := mat.NewDense(sz, sz, aArr[:dim])
		bMat := mat.NewDense(sz, sz, bArr[:dim])
		packedB := mat.PackGemmB(bMat)
		cMat := mat.NewDense(sz, sz, make([]float64, dim))

		b.Run(fmt.Sprintf("compact %v", sz), func(b *testing.B) {
			for n := 0; n < b.N; n++ {
				_ = mat.MulPackedGemm(cMat, aMat, packedB)
			}
			DotResult = cMat
		})
	}
}

func Benchmark_ProductPacked(b *testing.B) {
	var sizes = sizes
	if testing.Short() {
		sizes = []int{1024}
	}
	for _, sz := range sizes {
		dim := sz * sz
		aMat := mat.NewDense(sz, sz, aArr[:dim])
		bMat := mat.NewDense(sz, sz, bArr[:dim])
		packedB := mat.PackBTransposed(bMat)
		cMat := mat.NewDense(sz, sz, make([]float64, dim))

		b.Run(fmt.Sprintf("dot transposed simd packed %v", sz), func(b *testing.B) {
			for n := 0; n < b.N; n++ {
				_ = mat.MulPackedDotTransposedSimd(cMat, aMat, packedB)
			}
			DotResult = cMat
		})
	}
}

func Benchmark_ProductPackedStrided(b *testing.B) {
	var sizes = sizes
	if testing.Short() {
		sizes = []int{1024}
	}
	for _, sz := range sizes {
		stride := sz + 8
		aMat := newStridedBenchmarkDense(sz, sz, stride, aArr[:])
		bMat := newStridedBenchmarkDense(sz, sz, stride, bArr[:])
		packedB := mat.PackBTransposed(bMat)
		cMat := mat.NewDenseWithStride(sz, sz, stride, nil)

		b.Run(fmt.Sprintf("dot transposed simd packed %v", sz), func(b *testing.B) {
			for n := 0; n < b.N; n++ {
				_ = mat.MulPackedDotTransposedSimd(cMat, aMat, packedB)
			}
			DotResult = cMat
		})
	}
}

func newStridedBenchmarkDense(rows, cols, stride int, values []float64) *mat.Dense {
	data := make([]float64, (rows-1)*stride+cols)
	valueStart := 0
	for r := 0; r < rows; r++ {
		copy(data[r*stride:r*stride+cols], values[valueStart:valueStart+cols])
		valueStart += cols
	}
	return mat.NewDenseWithStride(rows, cols, stride, data)
}

func newPatternDense(rows, cols, stride int) *mat.Dense {
	data := make([]float64, (rows-1)*stride+cols)
	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			data[r*stride+c] = float64((r*cols+c)%11 - 5)
		}
	}
	return mat.NewDenseWithStride(rows, cols, stride, data)
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

const maxN = 2048

var (
	DotResult *mat.Dense
	Result    blas64.General
	aArr      [maxN * maxN]float64
	bArr      [maxN * maxN]float64
)

var sizes = []int{
	64,
	128,
	256,
	512,
	1024,
}

var eightByEight = mat.NewDense(8, 8, []float64{1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8})
var eightByEightTwo = mat.NewDense(8, 8, []float64{1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8})
var eightByEightProduct = []float64{36, 72, 108, 144, 180, 216, 252, 288, 36, 72, 108, 144, 180, 216, 252, 288, 36, 72, 108, 144, 180, 216, 252, 288, 36, 72, 108, 144, 180, 216, 252, 288, 36, 72, 108, 144, 180, 216, 252, 288, 36, 72, 108, 144, 180, 216, 252, 288, 36, 72, 108, 144, 180, 216, 252, 288, 36, 72, 108, 144, 180, 216, 252, 288}
var vecTen = []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
var vecSixteen = []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
var matSixteen = []float64{
	1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
	1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
	1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
	1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
	1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
	1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
	1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
	1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
	1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
	1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
	1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
	1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
	1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
	1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
	1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
	1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
}
var matSixteenProduct = []float64{
	136, 272, 408, 544, 680, 816, 952, 1088, 1224, 1360, 1496, 1632, 1768, 1904, 2040, 2176,
	136, 272, 408, 544, 680, 816, 952, 1088, 1224, 1360, 1496, 1632, 1768, 1904, 2040, 2176,
	136, 272, 408, 544, 680, 816, 952, 1088, 1224, 1360, 1496, 1632, 1768, 1904, 2040, 2176,
	136, 272, 408, 544, 680, 816, 952, 1088, 1224, 1360, 1496, 1632, 1768, 1904, 2040, 2176,
	136, 272, 408, 544, 680, 816, 952, 1088, 1224, 1360, 1496, 1632, 1768, 1904, 2040, 2176,
	136, 272, 408, 544, 680, 816, 952, 1088, 1224, 1360, 1496, 1632, 1768, 1904, 2040, 2176,
	136, 272, 408, 544, 680, 816, 952, 1088, 1224, 1360, 1496, 1632, 1768, 1904, 2040, 2176,
	136, 272, 408, 544, 680, 816, 952, 1088, 1224, 1360, 1496, 1632, 1768, 1904, 2040, 2176,
	136, 272, 408, 544, 680, 816, 952, 1088, 1224, 1360, 1496, 1632, 1768, 1904, 2040, 2176,
	136, 272, 408, 544, 680, 816, 952, 1088, 1224, 1360, 1496, 1632, 1768, 1904, 2040, 2176,
	136, 272, 408, 544, 680, 816, 952, 1088, 1224, 1360, 1496, 1632, 1768, 1904, 2040, 2176,
	136, 272, 408, 544, 680, 816, 952, 1088, 1224, 1360, 1496, 1632, 1768, 1904, 2040, 2176,
	136, 272, 408, 544, 680, 816, 952, 1088, 1224, 1360, 1496, 1632, 1768, 1904, 2040, 2176,
	136, 272, 408, 544, 680, 816, 952, 1088, 1224, 1360, 1496, 1632, 1768, 1904, 2040, 2176,
	136, 272, 408, 544, 680, 816, 952, 1088, 1224, 1360, 1496, 1632, 1768, 1904, 2040, 2176,
	136, 272, 408, 544, 680, 816, 952, 1088, 1224, 1360, 1496, 1632, 1768, 1904, 2040, 2176,
}

func init() {
	rand.Seed(173)

	for i := range aArr {
		aArr[i] = rand.NormFloat64()
		bArr[i] = rand.NormFloat64()
	}
}
