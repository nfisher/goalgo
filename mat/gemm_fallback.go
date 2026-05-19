//go:build !arm64

package mat

const hasGemmKernel4x4SIMD = false
const hasGemmKernel4x8SIMD = false

func gemmKernel4x4SIMD(c, a, b []float64, k, aStride, bStride, cStride, row, col int) {
	gemmKernel4x4Strided(c, a, b, k, aStride, bStride, cStride, row, col)
}

func gemmKernel4x8SIMD(c, a, b []float64, k, aStride, bStride, cStride, row, col int) {
	gemmKernel4x4Strided(c, a, b, k, aStride, bStride, cStride, row, col)
	gemmKernel4x4Strided(c, a, b, k, aStride, bStride, cStride, row, col+blockedGemmCols)
}
