package mat

import "runtime"

const (
	packedGemmKBlock   = 2048
	packedGemmColBlock = 256
	packedGemmMinDim   = 1024
	packedGemmMinRows  = 512
	packedGemmMinK     = 1024
	packedGemmMinCols  = 1024
	packedGemmMinOps   = 512 * 1024 * 1024
)

// PackedGemmB stores B in packed K x N panels for the blocked GEMM
// microkernels. It is useful when multiplying many A matrices by the same B.
type PackedGemmB struct {
	rows     int
	cols     int
	kBlock   int
	colBlock int
	panels   []packedGemmBPanel
	data     []float64
}

type packedGemmBPanel struct {
	kStart   int
	colStart int
	kLen     int
	colLen   int
	offset   int
}

// MulBlockedGemmPacked multiplies dense matrices using K-blocking and packed B
// panels. It is intended for large one-shot GEMM workloads where the packing
// cost is offset by better cache locality.
func MulBlockedGemmPacked(c, a, b *Dense) error {
	mulBlockedGemmPacked(c, a, b)
	return nil
}

// PackGemmB copies B into reusable K x N panels for MulPackedGemm.
func PackGemmB(b *Dense) *PackedGemmB {
	return PackGemmBWithBlocks(b, packedGemmKBlock, packedGemmColBlock)
}

// PackGemmBWithBlocks copies B into reusable K x N panels using explicit
// block sizes. It is primarily useful for benchmarking and tuning.
func PackGemmBWithBlocks(b *Dense, kBlock, colBlock int) *PackedGemmB {
	if kBlock <= 0 {
		panic("mat: packed GEMM k block must be positive")
	}
	if colBlock <= 0 {
		panic("mat: packed GEMM column block must be positive")
	}

	rows := b.Rows()
	cols := b.Columns()
	packed := &PackedGemmB{
		rows:     rows,
		cols:     cols,
		kBlock:   kBlock,
		colBlock: colBlock,
	}
	if rows == 0 || cols == 0 {
		return packed
	}

	for colStart := 0; colStart < cols; colStart += colBlock {
		colLen := colBlock
		if colStart+colLen > cols {
			colLen = cols - colStart
		}
		for kStart := 0; kStart < rows; kStart += kBlock {
			kLen := kBlock
			if kStart+kLen > rows {
				kLen = rows - kStart
			}
			offset := len(packed.data)
			packed.data = append(packed.data, make([]float64, kLen*colLen)...)
			packBPanel(packed.data[offset:offset+kLen*colLen], b.data, kStart, colStart, kLen, colLen, b.stride)
			packed.panels = append(packed.panels, packedGemmBPanel{
				kStart:   kStart,
				colStart: colStart,
				kLen:     kLen,
				colLen:   colLen,
				offset:   offset,
			})
		}
	}
	return packed
}

// MulPackedGemm multiplies A by a reusable packed-B buffer.
func MulPackedGemm(c, a *Dense, packed *PackedGemmB) error {
	if packed.rows != a.Columns() {
		return errPackedDimensionMismatch
	}
	mulPackedGemm(c, a, packed)
	return nil
}

func shouldUsePackedBlockedGemm(aRows, aCols, bCols int) bool {
	if !hasGemmKernel4x8SIMD {
		return false
	}
	if aRows >= packedGemmMinDim && aCols >= packedGemmMinDim && bCols >= packedGemmMinDim {
		return true
	}
	return aRows >= packedGemmMinRows &&
		aCols >= packedGemmMinK &&
		bCols >= packedGemmMinCols &&
		aRows*aCols*bCols >= packedGemmMinOps
}

func mulBlockedGemmPacked(c, a, b *Dense) {
	aRows := a.Rows()
	aCols := a.Columns()
	bCols := b.Columns()
	if aRows == 0 || bCols == 0 {
		return
	}
	if aCols == 0 {
		zeroDense(c, aRows, bCols)
		return
	}

	workers := runtimeWorkersForRows(aRows)
	rowsPerWorker := 0
	var pool *matMulWorkerPool
	if workers >= 2 {
		rowsPerWorker = (aRows + workers - 1) / workers
		rowsPerWorker = (rowsPerWorker + blockedGemmRows - 1) / blockedGemmRows * blockedGemmRows
		pool = getMatMulWorkerPool(workers)
	}

	maxPanelK := aCols
	if maxPanelK > packedGemmKBlock {
		maxPanelK = packedGemmKBlock
	}
	maxPanelCols := bCols
	if maxPanelCols > packedGemmColBlock {
		maxPanelCols = packedGemmColBlock
	}
	bPanel := make([]float64, maxPanelK*maxPanelCols)
	for colStart := 0; colStart < bCols; colStart += packedGemmColBlock {
		colEnd := colStart + packedGemmColBlock
		if colEnd > bCols {
			colEnd = bCols
		}
		colLen := colEnd - colStart
		cPanel := c.data[colStart:]

		for kStart := 0; kStart < aCols; kStart += packedGemmKBlock {
			kEnd := kStart + packedGemmKBlock
			if kEnd > aCols {
				kEnd = aCols
			}
			kLen := kEnd - kStart
			panel := bPanel[:kLen*colLen]
			packBPanel(panel, b.data, kStart, colStart, kLen, colLen, b.stride)
			aPanel := a.data[kStart:]
			add := kStart != 0

			if workers < 2 {
				mulPackedGemmPanelRowsAuto(cPanel, aPanel, panel, kLen, colLen, a.stride, c.stride, 0, aRows, add)
				continue
			}
			pool.parallel(0, aRows, rowsPerWorker, func(start, end int) {
				mulPackedGemmPanelRowsAuto(cPanel, aPanel, panel, kLen, colLen, a.stride, c.stride, start, end, add)
			})
		}
	}
}

func mulPackedGemm(c, a *Dense, packed *PackedGemmB) {
	aRows := a.Rows()
	if aRows == 0 || packed.cols == 0 {
		return
	}
	if packed.rows == 0 {
		zeroDense(c, aRows, packed.cols)
		return
	}

	workers := runtimeWorkersForRows(aRows)
	rowsPerWorker := 0
	var pool *matMulWorkerPool
	if workers >= 2 {
		rowsPerWorker = (aRows + workers - 1) / workers
		rowsPerWorker = (rowsPerWorker + blockedGemmRows - 1) / blockedGemmRows * blockedGemmRows
		pool = getMatMulWorkerPool(workers)
	}

	for _, panel := range packed.panels {
		bPanel := packed.data[panel.offset : panel.offset+panel.kLen*panel.colLen]
		cPanel := c.data[panel.colStart:]
		aPanel := a.data[panel.kStart:]
		add := panel.kStart != 0

		if workers < 2 {
			mulPackedGemmPanelRowsAuto(cPanel, aPanel, bPanel, panel.kLen, panel.colLen, a.stride, c.stride, 0, aRows, add)
			continue
		}
		pool.parallel(0, aRows, rowsPerWorker, func(start, end int) {
			mulPackedGemmPanelRowsAuto(cPanel, aPanel, bPanel, panel.kLen, panel.colLen, a.stride, c.stride, start, end, add)
		})
	}
}

func runtimeWorkersForRows(rows int) int {
	workers := runtime.GOMAXPROCS(0)
	rowBlocks := (rows + blockedGemmRows - 1) / blockedGemmRows
	if workers > rowBlocks {
		workers = rowBlocks
	}
	return workers
}

func packBPanel(dst, b []float64, rowStart, colStart, rows, cols, stride int) {
	for r := 0; r < rows; r++ {
		srcStart := (rowStart+r)*stride + colStart
		copy(dst[r*cols:r*cols+cols], b[srcStart:srcStart+cols])
	}
}

func zeroDense(c *Dense, rows, cols int) {
	for r := 0; r < rows; r++ {
		row := c.data[r*c.stride : r*c.stride+cols]
		for i := range row {
			row[i] = 0
		}
	}
}

func mulPackedGemmPanelRowsAuto(c, a, b []float64, k, cols, aStride, cStride, rowStart, rowEnd int, add bool) {
	if shouldPackGemmA(k, cols, rowEnd-rowStart) {
		mulPackedGemmPanelRowsPackedA(c, a, b, k, cols, aStride, cStride, rowStart, rowEnd, add)
		return
	}
	mulPackedGemmPanelRows(c, a, b, k, cols, aStride, cStride, rowStart, rowEnd, add)
}

func shouldPackGemmA(k, cols, rows int) bool {
	return hasGemmKernel4x12SIMD && rows >= blockedGemmRows && k >= 128 && cols >= 3*blockedGemmCols
}

func mulPackedGemmPanelRowsPackedA(c, a, b []float64, k, cols, aStride, cStride, rowStart, rowEnd int, add bool) {
	aBlock := make([]float64, blockedGemmRows*k)
	i := rowStart
	for ; i+blockedGemmRows <= rowEnd; i += blockedGemmRows {
		packGemmARows(aBlock, a, i, k, aStride)
		cBlock := c[i*cStride:]
		j := 0
		for ; j+3*blockedGemmCols <= cols; j += 3 * blockedGemmCols {
			if add {
				gemmKernel4x12AddSIMD(cBlock, aBlock, b, k, k, cols, cStride, 0, j)
			} else {
				gemmKernel4x12SIMD(cBlock, aBlock, b, k, k, cols, cStride, 0, j)
			}
		}
		for ; j+2*blockedGemmCols <= cols; j += 2 * blockedGemmCols {
			if add {
				gemmKernel4x8AddSIMD(cBlock, aBlock, b, k, k, cols, cStride, 0, j)
			} else {
				gemmKernel4x8SIMD(cBlock, aBlock, b, k, k, cols, cStride, 0, j)
			}
		}
		for ; j+blockedGemmCols <= cols; j += blockedGemmCols {
			if add {
				gemmKernel4x4AddStrided(cBlock, aBlock, b, k, k, cols, cStride, 0, j)
			} else if hasGemmKernel4x4SIMD {
				gemmKernel4x4SIMD(cBlock, aBlock, b, k, k, cols, cStride, 0, j)
			} else {
				gemmKernel4x4Strided(cBlock, aBlock, b, k, k, cols, cStride, 0, j)
			}
		}
		for ; j < cols; j++ {
			if add {
				gemmKernel4x1AddStrided(cBlock, aBlock, b, k, k, cols, cStride, 0, j)
			} else {
				gemmKernel4x1Strided(cBlock, aBlock, b, k, k, cols, cStride, 0, j)
			}
		}
	}

	if i < rowEnd {
		mulPackedGemmPanelRows(c, a, b, k, cols, aStride, cStride, i, rowEnd, add)
	}
}

func packGemmARows(dst, a []float64, row, k, stride int) {
	for r := 0; r < blockedGemmRows; r++ {
		srcStart := (row + r) * stride
		copy(dst[r*k:(r+1)*k], a[srcStart:srcStart+k])
	}
}

func mulPackedGemmPanelRows(c, a, b []float64, k, cols, aStride, cStride, rowStart, rowEnd int, add bool) {
	i := rowStart
	for ; i+blockedGemmRows <= rowEnd; i += blockedGemmRows {
		j := 0
		for ; j+3*blockedGemmCols <= cols; j += 3 * blockedGemmCols {
			if add {
				gemmKernel4x12AddSIMD(c, a, b, k, aStride, cols, cStride, i, j)
			} else {
				gemmKernel4x12SIMD(c, a, b, k, aStride, cols, cStride, i, j)
			}
		}
		for ; j+2*blockedGemmCols <= cols; j += 2 * blockedGemmCols {
			if add {
				gemmKernel4x8AddSIMD(c, a, b, k, aStride, cols, cStride, i, j)
			} else {
				gemmKernel4x8SIMD(c, a, b, k, aStride, cols, cStride, i, j)
			}
		}
		for ; j+blockedGemmCols <= cols; j += blockedGemmCols {
			if add {
				gemmKernel4x4AddStrided(c, a, b, k, aStride, cols, cStride, i, j)
			} else if hasGemmKernel4x4SIMD {
				gemmKernel4x4SIMD(c, a, b, k, aStride, cols, cStride, i, j)
			} else {
				gemmKernel4x4Strided(c, a, b, k, aStride, cols, cStride, i, j)
			}
		}
		for ; j < cols; j++ {
			if add {
				gemmKernel4x1AddStrided(c, a, b, k, aStride, cols, cStride, i, j)
			} else {
				gemmKernel4x1Strided(c, a, b, k, aStride, cols, cStride, i, j)
			}
		}
	}

	for ; i < rowEnd; i++ {
		j := 0
		for ; j+blockedGemmCols <= cols; j += blockedGemmCols {
			if add {
				gemmKernel1x4AddStrided(c, a, b, k, aStride, cols, cStride, i, j)
			} else {
				gemmKernel1x4Strided(c, a, b, k, aStride, cols, cStride, i, j)
			}
		}
		for ; j < cols; j++ {
			if add {
				gemmKernel1x1AddStrided(c, a, b, k, aStride, cols, cStride, i, j)
			} else {
				gemmKernel1x1Strided(c, a, b, k, aStride, cols, cStride, i, j)
			}
		}
	}
}

func gemmKernel4x4AddStrided(c, a, b []float64, k, aStride, bStride, cStride, row, col int) {
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

	c[c0+0] += c00
	c[c0+1] += c01
	c[c0+2] += c02
	c[c0+3] += c03
	c[c1+0] += c10
	c[c1+1] += c11
	c[c1+2] += c12
	c[c1+3] += c13
	c[c2+0] += c20
	c[c2+1] += c21
	c[c2+2] += c22
	c[c2+3] += c23
	c[c3+0] += c30
	c[c3+1] += c31
	c[c3+2] += c32
	c[c3+3] += c33
}

func gemmKernel4x1AddStrided(c, a, b []float64, k, aStride, bStride, cStride, row, col int) {
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
	c[cBase] += c0
	c[cBase+cStride] += c1
	c[cBase+2*cStride] += c2
	c[cBase+3*cStride] += c3
}

func gemmKernel1x4AddStrided(c, a, b []float64, k, aStride, bStride, cStride, row, col int) {
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
	c[cBase+0] += c0
	c[cBase+1] += c1
	c[cBase+2] += c2
	c[cBase+3] += c3
}

func gemmKernel1x1AddStrided(c, a, b []float64, k, aStride, bStride, cStride, row, col int) {
	a0 := row * aStride
	var sum float64
	for p := 0; p < k; p++ {
		sum += a[a0+p] * b[p*bStride+col]
	}
	c[row*cStride+col] += sum
}
