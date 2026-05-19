//go:build arm64

package mat

const hasGemmKernel4x4SIMD = true

//go:noescape
func gemmKernel4x4SIMD(c, a, b []float64, k, aStride, bStride, cStride, row, col int)
