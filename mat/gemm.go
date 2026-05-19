package mat

import "runtime"

const (
	blockedGemmRows = 4
	blockedGemmCols = 4
)

// MulBlockedGemm multiplies dense matrices using a small register-blocked
// 4x4 GEMM microkernel over row-major B.
func MulBlockedGemm(c, a, b *Dense) error {
	aRows := a.Rows()
	aCols := a.Columns()
	bCols := b.Columns()
	compact := denseAllCompact(c, a, b)

	if shouldUsePackedBlockedGemm(aRows, aCols, bCols) {
		mulBlockedGemmPacked(c, a, b)
		return nil
	}

	if aRows*bCols < minParallelCells {
		if compact {
			mulBlockedGemmRows(c.data, a.data, b.data, aCols, bCols, 0, aRows)
		} else {
			mulBlockedGemmRowsStrided(c.data, a.data, b.data, aCols, bCols, a.stride, b.stride, c.stride, 0, aRows)
		}
		return nil
	}

	workers := runtime.GOMAXPROCS(0)
	rowBlocks := (aRows + blockedGemmRows - 1) / blockedGemmRows
	if workers > rowBlocks {
		workers = rowBlocks
	}
	if workers < 2 {
		if compact {
			mulBlockedGemmRows(c.data, a.data, b.data, aCols, bCols, 0, aRows)
		} else {
			mulBlockedGemmRowsStrided(c.data, a.data, b.data, aCols, bCols, a.stride, b.stride, c.stride, 0, aRows)
		}
		return nil
	}

	rowsPerWorker := (aRows + workers - 1) / workers
	rowsPerWorker = (rowsPerWorker + blockedGemmRows - 1) / blockedGemmRows * blockedGemmRows
	if compact {
		getMatMulWorkerPool(workers).parallel(0, aRows, rowsPerWorker, func(start, end int) {
			mulBlockedGemmRows(c.data, a.data, b.data, aCols, bCols, start, end)
		})
	} else {
		getMatMulWorkerPool(workers).parallel(0, aRows, rowsPerWorker, func(start, end int) {
			mulBlockedGemmRowsStrided(c.data, a.data, b.data, aCols, bCols, a.stride, b.stride, c.stride, start, end)
		})
	}
	return nil
}

func mulBlockedGemmRows(c, a, b []float64, k, cols, rowStart, rowEnd int) {
	i := rowStart
	for ; i+blockedGemmRows <= rowEnd; i += blockedGemmRows {
		j := 0
		if hasGemmKernel4x12SIMD {
			for ; j+3*blockedGemmCols <= cols; j += 3 * blockedGemmCols {
				gemmKernel4x12SIMD(c, a, b, k, k, cols, cols, i, j)
			}
		}
		if hasGemmKernel4x8SIMD {
			for ; j+2*blockedGemmCols <= cols; j += 2 * blockedGemmCols {
				gemmKernel4x8SIMD(c, a, b, k, k, cols, cols, i, j)
			}
		}
		for ; j+blockedGemmCols <= cols; j += blockedGemmCols {
			if hasGemmKernel4x4SIMD {
				gemmKernel4x4SIMD(c, a, b, k, k, cols, cols, i, j)
			} else {
				gemmKernel4x4(c, a, b, k, cols, i, j)
			}
		}
		for ; j < cols; j++ {
			gemmKernel4x1(c, a, b, k, cols, i, j)
		}
	}

	for ; i < rowEnd; i++ {
		j := 0
		for ; j+blockedGemmCols <= cols; j += blockedGemmCols {
			gemmKernel1x4(c, a, b, k, cols, i, j)
		}
		for ; j < cols; j++ {
			gemmKernel1x1(c, a, b, k, cols, i, j)
		}
	}
}

func gemmKernel4x4(c, a, b []float64, k, cols, row, col int) {
	a0 := row * k
	a1 := a0 + k
	a2 := a1 + k
	a3 := a2 + k

	var c00, c01, c02, c03 float64
	var c10, c11, c12, c13 float64
	var c20, c21, c22, c23 float64
	var c30, c31, c32, c33 float64

	for p := 0; p < k; p++ {
		bp := p*cols + col
		b0 := b[bp+0]
		b1 := b[bp+1]
		b2 := b[bp+2]
		b3 := b[bp+3]

		av := a[a0+p]
		c00 += av * b0
		c01 += av * b1
		c02 += av * b2
		c03 += av * b3

		av = a[a1+p]
		c10 += av * b0
		c11 += av * b1
		c12 += av * b2
		c13 += av * b3

		av = a[a2+p]
		c20 += av * b0
		c21 += av * b1
		c22 += av * b2
		c23 += av * b3

		av = a[a3+p]
		c30 += av * b0
		c31 += av * b1
		c32 += av * b2
		c33 += av * b3
	}

	c0 := row*cols + col
	c1 := c0 + cols
	c2 := c1 + cols
	c3 := c2 + cols

	c[c0+0] = c00
	c[c0+1] = c01
	c[c0+2] = c02
	c[c0+3] = c03
	c[c1+0] = c10
	c[c1+1] = c11
	c[c1+2] = c12
	c[c1+3] = c13
	c[c2+0] = c20
	c[c2+1] = c21
	c[c2+2] = c22
	c[c2+3] = c23
	c[c3+0] = c30
	c[c3+1] = c31
	c[c3+2] = c32
	c[c3+3] = c33
}

func gemmKernel4x1(c, a, b []float64, k, cols, row, col int) {
	a0 := row * k
	a1 := a0 + k
	a2 := a1 + k
	a3 := a2 + k

	var c0, c1, c2, c3 float64
	for p := 0; p < k; p++ {
		bv := b[p*cols+col]
		c0 += a[a0+p] * bv
		c1 += a[a1+p] * bv
		c2 += a[a2+p] * bv
		c3 += a[a3+p] * bv
	}

	cBase := row*cols + col
	c[cBase] = c0
	c[cBase+cols] = c1
	c[cBase+2*cols] = c2
	c[cBase+3*cols] = c3
}

func gemmKernel1x4(c, a, b []float64, k, cols, row, col int) {
	a0 := row * k
	var c0, c1, c2, c3 float64
	for p := 0; p < k; p++ {
		bp := p*cols + col
		av := a[a0+p]
		c0 += av * b[bp+0]
		c1 += av * b[bp+1]
		c2 += av * b[bp+2]
		c3 += av * b[bp+3]
	}

	cBase := row*cols + col
	c[cBase+0] = c0
	c[cBase+1] = c1
	c[cBase+2] = c2
	c[cBase+3] = c3
}

func gemmKernel1x1(c, a, b []float64, k, cols, row, col int) {
	a0 := row * k
	var sum float64
	for p := 0; p < k; p++ {
		sum += a[a0+p] * b[p*cols+col]
	}
	c[row*cols+col] = sum
}

func mulBlockedGemmRowsStrided(c, a, b []float64, k, cols, aStride, bStride, cStride, rowStart, rowEnd int) {
	i := rowStart
	for ; i+blockedGemmRows <= rowEnd; i += blockedGemmRows {
		j := 0
		if hasGemmKernel4x12SIMD {
			for ; j+3*blockedGemmCols <= cols; j += 3 * blockedGemmCols {
				gemmKernel4x12SIMD(c, a, b, k, aStride, bStride, cStride, i, j)
			}
		}
		if hasGemmKernel4x8SIMD {
			for ; j+2*blockedGemmCols <= cols; j += 2 * blockedGemmCols {
				gemmKernel4x8SIMD(c, a, b, k, aStride, bStride, cStride, i, j)
			}
		}
		for ; j+blockedGemmCols <= cols; j += blockedGemmCols {
			if hasGemmKernel4x4SIMD {
				gemmKernel4x4SIMD(c, a, b, k, aStride, bStride, cStride, i, j)
			} else {
				gemmKernel4x4Strided(c, a, b, k, aStride, bStride, cStride, i, j)
			}
		}
		for ; j < cols; j++ {
			gemmKernel4x1Strided(c, a, b, k, aStride, bStride, cStride, i, j)
		}
	}

	for ; i < rowEnd; i++ {
		j := 0
		for ; j+blockedGemmCols <= cols; j += blockedGemmCols {
			gemmKernel1x4Strided(c, a, b, k, aStride, bStride, cStride, i, j)
		}
		for ; j < cols; j++ {
			gemmKernel1x1Strided(c, a, b, k, aStride, bStride, cStride, i, j)
		}
	}
}

func gemmKernel4x4Strided(c, a, b []float64, k, aStride, bStride, cStride, row, col int) {
	a0 := row * aStride
	a1 := a0 + aStride
	a2 := a1 + aStride
	a3 := a2 + aStride

	var c00, c01, c02, c03 float64
	var c10, c11, c12, c13 float64
	var c20, c21, c22, c23 float64
	var c30, c31, c32, c33 float64

	for p := 0; p < k; p++ {
		bp := p*bStride + col
		b0 := b[bp+0]
		b1 := b[bp+1]
		b2 := b[bp+2]
		b3 := b[bp+3]

		av := a[a0+p]
		c00 += av * b0
		c01 += av * b1
		c02 += av * b2
		c03 += av * b3

		av = a[a1+p]
		c10 += av * b0
		c11 += av * b1
		c12 += av * b2
		c13 += av * b3

		av = a[a2+p]
		c20 += av * b0
		c21 += av * b1
		c22 += av * b2
		c23 += av * b3

		av = a[a3+p]
		c30 += av * b0
		c31 += av * b1
		c32 += av * b2
		c33 += av * b3
	}

	c0 := row*cStride + col
	c1 := c0 + cStride
	c2 := c1 + cStride
	c3 := c2 + cStride

	c[c0+0] = c00
	c[c0+1] = c01
	c[c0+2] = c02
	c[c0+3] = c03
	c[c1+0] = c10
	c[c1+1] = c11
	c[c1+2] = c12
	c[c1+3] = c13
	c[c2+0] = c20
	c[c2+1] = c21
	c[c2+2] = c22
	c[c2+3] = c23
	c[c3+0] = c30
	c[c3+1] = c31
	c[c3+2] = c32
	c[c3+3] = c33
}

func gemmKernel4x1Strided(c, a, b []float64, k, aStride, bStride, cStride, row, col int) {
	a0 := row * aStride
	a1 := a0 + aStride
	a2 := a1 + aStride
	a3 := a2 + aStride

	var c0, c1, c2, c3 float64
	for p := 0; p < k; p++ {
		bv := b[p*bStride+col]
		c0 += a[a0+p] * bv
		c1 += a[a1+p] * bv
		c2 += a[a2+p] * bv
		c3 += a[a3+p] * bv
	}

	cBase := row*cStride + col
	c[cBase] = c0
	c[cBase+cStride] = c1
	c[cBase+2*cStride] = c2
	c[cBase+3*cStride] = c3
}

func gemmKernel1x4Strided(c, a, b []float64, k, aStride, bStride, cStride, row, col int) {
	a0 := row * aStride
	var c0, c1, c2, c3 float64
	for p := 0; p < k; p++ {
		bp := p*bStride + col
		av := a[a0+p]
		c0 += av * b[bp+0]
		c1 += av * b[bp+1]
		c2 += av * b[bp+2]
		c3 += av * b[bp+3]
	}

	cBase := row*cStride + col
	c[cBase+0] = c0
	c[cBase+1] = c1
	c[cBase+2] = c2
	c[cBase+3] = c3
}

func gemmKernel1x1Strided(c, a, b []float64, k, aStride, bStride, cStride, row, col int) {
	a0 := row * aStride
	var sum float64
	for p := 0; p < k; p++ {
		sum += a[a0+p] * b[p*bStride+col]
	}
	c[row*cStride+col] = sum
}
