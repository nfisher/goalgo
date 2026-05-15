//go:build !amd64

package mat

// AxpyLoop loops around an axpy function.
func AxpyLoop(c, b []float64, s float64, bc, cib, bib, blockSize int) {
	for bcb := bc; bcb < bc+blockSize; bcb++ {
		c[cib+bcb] += s * b[bib+bcb]
	}
}

// QuadAxpy applies 4 axpy derivatives as vectorised operations.
func QuadAxpy(c, b []float64, s float64, ci, bi int) {
	for i := 0; i < 4; i++ {
		c[ci+i] += s * b[bi+i]
	}
}

// HexAxpy applies axpy operations.
func HexAxpy(c, b []float64, s float64, ci, bi int) {
	for i := 0; i < 8; i++ {
		c[ci+i] += s * b[bi+i]
	}
}

// HexadecAxpy applies axpy operations.
func HexadecAxpy(c, b []float64, s float64, ci, bi int) {
	for i := 0; i < 16; i++ {
		c[ci+i] += s * b[bi+i]
	}
}

// WideAxpy applies axpy operations.
func WideAxpy(c, b []float64, s float64, ci, bi int) {
	for i := 0; i < 32; i++ {
		c[ci+i] += s * b[bi+i]
	}
}
