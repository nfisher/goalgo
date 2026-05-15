package mat

import (
	"errors"
	"runtime"
	"sync"
)

var errPackedDimensionMismatch = errors.New("mat: packed B dimension mismatch")

// MulGaxpy is a General AXPY algorithm to multiple 2 matrices.
// Author: @james-bowman
func MulGaxpy(c, a, b *Dense) error {
	aCols := a.Columns()
	aRows := a.Rows()
	bCols := b.Columns()
	var data = c.data

	// IKJ
	for i := 0; i < aRows; i++ {
		for k := 0; k < aCols; k++ {
			s := a.data[i*aCols+k]
			ib := i * bCols
			kb := k * bCols
			for j := 0; j < bCols; j++ {
				data[ib+j] += s * b.data[kb+j]
			}
		}
	}
	return nil
}

// MulGonumNaivePrefetch multiples dense matrices.
// Author: @james-bowman
func MulGonumNaivePrefetch(c, a, b *Dense) error {
	aCols := a.Columns()
	aRows := a.Rows()
	bCols := b.Columns()
	row := make([]float64, aCols)
	var data = c.data
	for ar := 0; ar < aRows; ar++ {
		for i := range row {
			row[i] = a.At(ar, i)
		}
		for bc := 0; bc < bCols; bc++ {
			var sum float64
			for i, e := range row {
				sum += e * b.At(i, bc)
			}
			di := ar*bCols + bc
			data[di] = sum
		}
	}

	return nil
}

// MulGonumNaive multiples matrices.
func MulGonumNaive(c, a, b *Dense) error {
	aCols := a.Columns()
	aRows := a.Rows()
	bCols := b.Columns()

	for ar := 0; ar < aRows; ar++ {
		for bc := 0; bc < bCols; bc++ {
			var sum float64
			for ac := 0; ac < aCols; ac++ {
				sum += a.At(ar, ac) * b.At(ac, bc)
			}
			c.Set(ar, bc, sum)
		}
	}

	return nil
}

// MulMultiplePrefetch2 uses simple iteration to create the product of two matrices.
// Author: @james-bowman
func MulMultiplePrefetch2(c, a, b *Dense) error {
	aCols := a.Columns()
	aRows := a.Rows()
	bCols := b.Columns()
	var data = c.data

	if aRows%4 != 0 {
		return MulNaiveIKJ(c, a, b)
	}

	var sum0 float64
	var sum1 float64
	var sum2 float64
	var sum3 float64
	buffer0 := make([]float64, aCols)
	buffer1 := make([]float64, aCols)
	buffer2 := make([]float64, aCols)
	buffer3 := make([]float64, aCols)

	for i := 0; i < aRows; i += 4 {
		for j := 0; j < aCols; j++ {
			buffer0[j] = a.data[i*aCols+j]
			buffer1[j] = a.data[(i+1)*aCols+j]
			buffer2[j] = a.data[(i+2)*aCols+j]
			buffer3[j] = a.data[(i+3)*aCols+j]
		}
		for j := 0; j < bCols; j++ {
			sum0 = 0
			sum1 = 0
			sum2 = 0
			sum3 = 0
			for k := 0; k < aCols; k++ {
				bi := k*bCols + j
				sum0 += buffer0[k] * b.data[bi]
				sum1 += buffer1[k] * b.data[bi]
				sum2 += buffer2[k] * b.data[bi]
				sum3 += buffer3[k] * b.data[bi]
			}
			data[i*bCols+j] = sum0
			data[(i+1)*bCols+j] = sum1
			data[(i+2)*bCols+j] = sum2
			data[(i+3)*bCols+j] = sum3
		}
	}
	return nil
}

// MulGonumUnroll multiplies a matrix using the gonum At interface with 8 wide unrolled loop.
func MulGonumUnroll(c, a, b *Dense) error {
	const stride = 8
	aCols := a.Columns()
	bCols := b.Columns()
	var data = c.data
	bounded := aCols
	remainder := aCols % stride
	if remainder != 0 {
		bounded = aCols / stride * stride
	}
	for ar := 0; ar < a.Rows(); ar++ {
		for bc := 0; bc < bCols; bc++ {
			var sum float64
			di := ar*bCols + bc
			for ac := 0; ac < bounded; ac += stride {
				ac0 := ac
				ac1 := ac0 + 1
				ac2 := ac0 + 2
				ac3 := ac0 + 3
				ac4 := ac0 + 4
				ac5 := ac0 + 5
				ac6 := ac0 + 6
				ac7 := ac0 + 7
				sum += a.At(ar, ac0) * b.At(ac0, bc)
				sum += a.At(ar, ac1) * b.At(ac1, bc)
				sum += a.At(ar, ac2) * b.At(ac2, bc)
				sum += a.At(ar, ac3) * b.At(ac3, bc)
				sum += a.At(ar, ac4) * b.At(ac4, bc)
				sum += a.At(ar, ac5) * b.At(ac5, bc)
				sum += a.At(ar, ac6) * b.At(ac6, bc)
				sum += a.At(ar, ac7) * b.At(ac7, bc)
			}

			for ac := bounded; ac < aCols; ac++ {
				sum += a.At(ar, ac) * b.At(ac, bc)
			}
			data[di] = sum
		}
	}

	return nil
}

// MulNaiveIKJ uses simple iteration to create the product of two matrices.
func MulNaiveIKJ(c, a, b *Dense) error {
	aCols := a.Columns()
	aRows := a.Rows()
	bCols := b.Columns()
	var data = c.data

	for ar := 0; ar < aRows; ar++ {
		for bc := 0; bc < bCols; bc++ {
			di := ar*bCols + bc
			for ac := 0; ac < aCols; ac++ {
				ai := ar*aCols + ac
				bi := ac*bCols + bc
				data[di] += a.data[ai] * b.data[bi]
			}
		}
	}

	return nil
}

// MulNaiveIJK multiplies matrices.
func MulNaiveIJK(c, a, b *Dense) error {
	aCols := a.Columns()
	aRows := a.Rows()
	bCols := b.Columns()
	var data = c.data

	for ar := 0; ar < aRows; ar++ {
		for ac := 0; ac < aCols; ac++ {
			ai := ar*aCols + ac
			for bc := 0; bc < bCols; bc++ {
				di := ar*bCols + bc
				bi := ac*bCols + bc
				data[di] += a.data[ai] * b.data[bi]
			}
		}
	}

	return nil
}

// MulSimdIJK is lazily implemented if any of the dims aren't divisible by block size it defers to naive IJK.
func MulSimdIJK(c, a, b *Dense) error {
	const blockSize = 32
	aCols := a.Columns()
	aRows := a.Rows()
	bCols := b.Columns()
	var cdata = c.data
	var bdata = b.data

	if aCols%blockSize != 0 || aRows%blockSize != 0 || bCols%blockSize != 0 {
		return MulNaiveIJK(c, a, b)
	}

	for ar := 0; ar < aRows; ar += blockSize {
		for ac := 0; ac < aCols; ac += blockSize {
			for bc := 0; bc < bCols; bc += blockSize {
				for arb := ar; arb < ar+blockSize; arb++ {
					cib := arb * bCols
					aib := arb * aCols
					for acb := ac; acb < ac+blockSize; acb++ {
						bib := acb * bCols
						ai := aib + acb
						s := a.data[ai]
						for bcb := bc; bcb < bc+blockSize; bcb += blockSize {
							ci := cib + bcb
							bi := bib + bcb
							WideAxpy(cdata, bdata, s, ci, bi)
							//AxpyLoop(cdata, bdata, s, bc, cib, bib, blockSize)
						}
					}
				}
			}
		}
	}
	return nil
}

// MulDotTransposed multiplies dense matrices by transposing B once so each
// output becomes a contiguous dot product.
func MulDotTransposed(c, a, b *Dense) error {
	const colBlock = 4
	const dotUnroll = 8

	aCols := a.Columns()
	aRows := a.Rows()
	bCols := b.Columns()
	cdata := c.data

	bt := make([]float64, len(b.data))
	transposeDense(bt, b.data, aCols, bCols)

	bounded := aCols / dotUnroll * dotUnroll
	for ar := 0; ar < aRows; ar++ {
		aStart := ar * aCols
		aRow := a.data[aStart : aStart+aCols]
		cStart := ar * bCols

		bc := 0
		for ; bc+colBlock <= bCols; bc += colBlock {
			b0Start := (bc + 0) * aCols
			b1Start := (bc + 1) * aCols
			b2Start := (bc + 2) * aCols
			b3Start := (bc + 3) * aCols

			b0 := bt[b0Start : b0Start+aCols]
			b1 := bt[b1Start : b1Start+aCols]
			b2 := bt[b2Start : b2Start+aCols]
			b3 := bt[b3Start : b3Start+aCols]

			var sum0 float64
			var sum1 float64
			var sum2 float64
			var sum3 float64

			for ac := 0; ac < bounded; ac += dotUnroll {
				av0 := aRow[ac+0]
				av1 := aRow[ac+1]
				av2 := aRow[ac+2]
				av3 := aRow[ac+3]
				av4 := aRow[ac+4]
				av5 := aRow[ac+5]
				av6 := aRow[ac+6]
				av7 := aRow[ac+7]

				sum0 += av0*b0[ac+0] + av1*b0[ac+1] + av2*b0[ac+2] + av3*b0[ac+3]
				sum0 += av4*b0[ac+4] + av5*b0[ac+5] + av6*b0[ac+6] + av7*b0[ac+7]
				sum1 += av0*b1[ac+0] + av1*b1[ac+1] + av2*b1[ac+2] + av3*b1[ac+3]
				sum1 += av4*b1[ac+4] + av5*b1[ac+5] + av6*b1[ac+6] + av7*b1[ac+7]
				sum2 += av0*b2[ac+0] + av1*b2[ac+1] + av2*b2[ac+2] + av3*b2[ac+3]
				sum2 += av4*b2[ac+4] + av5*b2[ac+5] + av6*b2[ac+6] + av7*b2[ac+7]
				sum3 += av0*b3[ac+0] + av1*b3[ac+1] + av2*b3[ac+2] + av3*b3[ac+3]
				sum3 += av4*b3[ac+4] + av5*b3[ac+5] + av6*b3[ac+6] + av7*b3[ac+7]
			}

			for ac := bounded; ac < aCols; ac++ {
				av := aRow[ac]
				sum0 += av * b0[ac]
				sum1 += av * b1[ac]
				sum2 += av * b2[ac]
				sum3 += av * b3[ac]
			}

			cdata[cStart+bc+0] = sum0
			cdata[cStart+bc+1] = sum1
			cdata[cStart+bc+2] = sum2
			cdata[cStart+bc+3] = sum3
		}

		for ; bc < bCols; bc++ {
			bStart := bc * aCols
			bRow := bt[bStart : bStart+aCols]
			var sum float64
			for ac := 0; ac < bounded; ac += dotUnroll {
				sum += aRow[ac+0] * bRow[ac+0]
				sum += aRow[ac+1] * bRow[ac+1]
				sum += aRow[ac+2] * bRow[ac+2]
				sum += aRow[ac+3] * bRow[ac+3]
				sum += aRow[ac+4] * bRow[ac+4]
				sum += aRow[ac+5] * bRow[ac+5]
				sum += aRow[ac+6] * bRow[ac+6]
				sum += aRow[ac+7] * bRow[ac+7]
			}
			for ac := bounded; ac < aCols; ac++ {
				sum += aRow[ac] * bRow[ac]
			}
			cdata[cStart+bc] = sum
		}
	}

	return nil
}

// MulDotTransposedSimd multiplies dense matrices by transposing B once and
// delegating each contiguous dot product to an architecture-specific kernel.
func MulDotTransposedSimd(c, a, b *Dense) error {
	return MulPackedDotTransposedSimd(c, a, PackBTransposed(b))
}

// PackedB stores B in transposed row-major form so matrix multiply can reuse a
// contiguous packed representation across repeated multiplies.
type PackedB struct {
	rows int
	cols int
	data []float64
}

// PackBTransposed copies B into a transposed contiguous layout.
func PackBTransposed(b *Dense) *PackedB {
	packed := &PackedB{
		rows: b.Rows(),
		cols: b.Columns(),
		data: make([]float64, len(b.data)),
	}
	transposeDense(packed.data, b.data, packed.rows, packed.cols)
	return packed
}

// MulPackedDotTransposedSimd multiplies A by a transposed packed-B buffer.
func MulPackedDotTransposedSimd(c, a *Dense, packed *PackedB) error {
	aCols := a.Columns()
	aRows := a.Rows()
	bCols := packed.cols

	if packed.rows != aCols {
		return errPackedDimensionMismatch
	}
	if aRows*bCols < 128*128 {
		mulPackedDotTransposedSimdRows(c.data, a, packed, 0, aRows)
		return nil
	}

	workers := runtime.GOMAXPROCS(0)
	if workers > aRows {
		workers = aRows
	}
	if workers < 2 {
		mulPackedDotTransposedSimdRows(c.data, a, packed, 0, aRows)
		return nil
	}

	chunk := (aRows + workers - 1) / workers
	var wg sync.WaitGroup
	for start := 0; start < aRows; start += chunk {
		end := start + chunk
		if end > aRows {
			end = aRows
		}
		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()
			mulPackedDotTransposedSimdRows(c.data, a, packed, start, end)
		}(start, end)
	}
	wg.Wait()
	return nil
}

func mulPackedDotTransposedSimdRows(cdata []float64, a *Dense, packed *PackedB, rowStart, rowEnd int) {
	aCols := a.Columns()
	bCols := packed.cols

	for ar := rowStart; ar < rowEnd; ar++ {
		aStart := ar * aCols
		aRow := a.data[aStart : aStart+aCols]
		cStart := ar * bCols

		bc := 0
		for ; bc+8 <= bCols; bc += 8 {
			b0Start := (bc + 0) * aCols
			b1Start := (bc + 1) * aCols
			b2Start := (bc + 2) * aCols
			b3Start := (bc + 3) * aCols
			b4Start := (bc + 4) * aCols
			b5Start := (bc + 5) * aCols
			b6Start := (bc + 6) * aCols
			b7Start := (bc + 7) * aCols

			b0 := packed.data[b0Start : b0Start+aCols]
			b1 := packed.data[b1Start : b1Start+aCols]
			b2 := packed.data[b2Start : b2Start+aCols]
			b3 := packed.data[b3Start : b3Start+aCols]
			b4 := packed.data[b4Start : b4Start+aCols]
			b5 := packed.data[b5Start : b5Start+aCols]
			b6 := packed.data[b6Start : b6Start+aCols]
			b7 := packed.data[b7Start : b7Start+aCols]
			s0, s1, s2, s3, s4, s5, s6, s7 := dotProductOct(aRow, b0, b1, b2, b3, b4, b5, b6, b7)
			cdata[cStart+bc+0] = s0
			cdata[cStart+bc+1] = s1
			cdata[cStart+bc+2] = s2
			cdata[cStart+bc+3] = s3
			cdata[cStart+bc+4] = s4
			cdata[cStart+bc+5] = s5
			cdata[cStart+bc+6] = s6
			cdata[cStart+bc+7] = s7
		}

		for ; bc+4 <= bCols; bc += 4 {
			b0Start := (bc + 0) * aCols
			b1Start := (bc + 1) * aCols
			b2Start := (bc + 2) * aCols
			b3Start := (bc + 3) * aCols

			b0 := packed.data[b0Start : b0Start+aCols]
			b1 := packed.data[b1Start : b1Start+aCols]
			b2 := packed.data[b2Start : b2Start+aCols]
			b3 := packed.data[b3Start : b3Start+aCols]
			s0, s1, s2, s3 := dotProductQuad(aRow, b0, b1, b2, b3)
			cdata[cStart+bc+0] = s0
			cdata[cStart+bc+1] = s1
			cdata[cStart+bc+2] = s2
			cdata[cStart+bc+3] = s3
		}

		for ; bc < bCols; bc++ {
			bStart := bc * aCols
			bRow := packed.data[bStart : bStart+aCols]
			cdata[cStart+bc] = dotProductUnitary(aRow, bRow)
		}
	}
}

func transposeDense(dst, src []float64, rows, cols int) {
	for r := 0; r < rows; r++ {
		srcStart := r * cols
		for c := 0; c < cols; c++ {
			dst[c*rows+r] = src[srcStart+c]
		}
	}
}

// MulNaiveKIJ multiplies matrices.
func MulNaiveKIJ(c, a, b *Dense) error {
	aCols := a.Columns()
	aRows := a.Rows()
	bCols := b.Columns()
	var data = c.data

	for bc := 0; bc < bCols; bc++ {
		for ar := 0; ar < aRows; ar++ {
			di := ar*bCols + bc
			for ac := 0; ac < aCols; ac++ {
				ai := ar*aCols + ac
				bi := ac*bCols + bc
				data[di] += a.data[ai] * b.data[bi]
			}
		}
	}

	return nil
}

// MulNaiveJIK multiplies matrices.
func MulNaiveJIK(c, a, b *Dense) error {
	aCols := a.Columns()
	aRows := a.Rows()
	bCols := b.Columns()
	var data = c.data

	for ac := 0; ac < aCols; ac++ {
		for ar := 0; ar < aRows; ar++ {
			ai := ar*aCols + ac
			for bc := 0; bc < bCols; bc++ {
				di := ar*bCols + bc
				bi := ac*bCols + bc
				data[di] += a.data[ai] * b.data[bi]
			}
		}
	}

	return nil
}

// MulNaiveJKI multiplies matrices.
func MulNaiveJKI(c, a, b *Dense) error {
	aCols := a.Columns()
	aRows := a.Rows()
	bCols := b.Columns()
	var data = c.data

	for ac := 0; ac < aCols; ac++ {
		for bc := 0; bc < bCols; bc++ {
			bi := ac*bCols + bc
			for ar := 0; ar < aRows; ar++ {
				ai := ar*aCols + ac
				di := ar*bCols + bc
				data[di] += a.data[ai] * b.data[bi]
			}
		}
	}

	return nil
}

// MulNaiveKJI multiplies matrices.
func MulNaiveKJI(c, a, b *Dense) error {
	aCols := a.Columns()
	aRows := a.Rows()
	bCols := b.Columns()

	var data = c.data

	for bc := 0; bc < bCols; bc++ {
		for ac := 0; ac < aCols; ac++ {
			bi := ac*bCols + bc
			for ar := 0; ar < aRows; ar++ {
				ai := ar*aCols + ac
				di := ar*bCols + bc
				data[di] += a.data[ai] * b.data[bi]
			}
		}
	}

	return nil
}

// MulBlockIJK is lazily implemented if any of the dims aren't divisible by block size it defers to naive IJK.
func MulBlockIJK(blockSize int) func(c, a, b *Dense) error {
	return func(c, a, b *Dense) error {
		aCols := a.Columns()
		aRows := a.Rows()
		bCols := b.Columns()
		var data = c.data

		if aCols%blockSize != 0 || aRows%blockSize != 0 || bCols%blockSize != 0 {
			return MulNaiveIJK(c, a, b)
		}

		for ar := 0; ar < aRows; ar += blockSize {
			for ac := 0; ac < aCols; ac += blockSize {
				for bc := 0; bc < bCols; bc += blockSize {
					for arb := ar; arb < ar+blockSize; arb++ {
						dib := arb * bCols
						aib := arb * aCols
						for acb := ac; acb < ac+blockSize; acb++ {
							bib := acb * bCols
							ai := aib + acb
							s := a.data[ai]
							for bcb := bc; bcb < bc+blockSize; bcb++ {
								di := dib + bcb
								bi := bib + bcb
								data[di] += s * b.data[bi]
							}
						}
					}
				}
			}
		}
		return nil
	}
}

// MulBlockUnrollIJK is lazily implemented if any of the dims aren't divisible by block size it defers to naive IJK.
func MulBlockUnrollIJK(blockSize int) func(c, a, b *Dense) error {
	return func(c, a, b *Dense) error {
		aCols := a.Columns()
		aRows := a.Rows()
		bCols := b.Columns()
		var data = c.data

		if aCols%blockSize != 0 || aRows%blockSize != 0 || bCols%blockSize != 0 {
			return MulNaiveIJK(c, a, b)
		}

		for ar := 0; ar < aRows; ar += blockSize {
			for ac := 0; ac < aCols; ac += blockSize {
				for bc := 0; bc < bCols; bc += blockSize {
					for arb := ar; arb < ar+blockSize; arb++ {
						dib := arb * bCols
						aib := arb * aCols
						for acb := ac; acb < ac+blockSize; acb++ {
							bib := acb * bCols
							ai := aib + acb
							s := a.data[ai]
							for bcb := bc; bcb < bc+blockSize; bcb += 4 {
								di := dib + bcb
								bi := bib + bcb
								data[di] += s * b.data[bi]
								data[di+1] += s * b.data[bi+1]
								data[di+2] += s * b.data[bi+2]
								data[di+3] += s * b.data[bi+3]
							}
						}
					}
				}
			}
		}
		return nil
	}
}

// MulBlockFetchIJK is lazily implemented if any of the dims aren't divisible by block size it defers to naive IJK.
func MulBlockFetchIJK(blockSize int) func(c, a, b *Dense) error {
	return func(c, a, b *Dense) error {
		aCols := a.Columns()
		aRows := a.Rows()
		bCols := b.Columns()
		row := make([]float64, blockSize)
		var data = c.data

		if aCols%blockSize != 0 || aRows%blockSize != 0 || bCols%blockSize != 0 {
			return MulNaiveIJK(c, a, b)
		}

		for ar := 0; ar < aRows; ar += blockSize {
			for ac := 0; ac < aCols; ac += blockSize {
				for bc := 0; bc < bCols; bc += blockSize {
					for arb := ar; arb < ar+blockSize; arb++ {
						dib := arb * bCols
						aib := ar * aCols
						for i := range row {
							row[i] = a.data[aib+ac+i]
						}
						for bcb := bc; bcb < bc+blockSize; bcb++ {
							var sum float64
							for i, e := range row {
								sum += e * b.data[(ac+i)*bCols+bcb]
							}
							di := dib + bcb
							data[di] += sum
						}
					}
				}
			}
		}
		return nil
	}
}

// MulUnroll uses unrolled loops to create the product of two matrices.
func MulUnroll(c, a, b *Dense) error {
	aCols := a.Columns()
	aRows := a.Rows()
	bCols := b.Columns()
	data := c.data
	const stride = 4
	bounded := aCols
	remainder := aCols % stride
	if remainder != 0 {
		bounded = aCols / stride * stride
	}
	var sum0 float64
	var sum1 float64
	var sum2 float64
	var sum3 float64
	for ar := 0; ar < aRows; ar++ {
		for bc := 0; bc < bCols; bc++ {
			var sum float64
			di := ar*bCols + bc
			for ac := 0; ac < bounded; ac += stride {
				b0 := ac*bCols + bc
				b1 := b0 + bCols
				b2 := b1 + bCols
				b3 := b2 + bCols

				a0 := ar*aCols + ac
				a1 := a0 + 1
				a2 := a1 + 1
				a3 := a2 + 1

				av0 := a.data[a0]
				av1 := a.data[a1]
				av2 := a.data[a2]
				av3 := a.data[a3]

				bv0 := b.data[b0]
				bv1 := b.data[b1]
				bv2 := b.data[b2]
				bv3 := b.data[b3]

				sum0 = av0 * bv0
				sum1 = av1 * bv1
				sum2 = av2 * bv2
				sum3 = av3 * bv3
				sum += sum0 + sum1 + sum2 + sum3
			}
			for ac := bounded; ac < aCols; ac++ {
				ai := ar*aCols + ac
				bi := ac*bCols + bc
				sum += a.data[ai] * b.data[bi]
			}
			data[di] = sum
		}
	}

	return nil
}
