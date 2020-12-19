package mat

// AxpyLoop loops around an axpy function.
func AxpyLoop(c, b []float64, s float64, bc, cib, bib, blockSize int)

// QuadAxpy applies 4 axpy derivatives as vectorised operations.
func QuadAxpy(c, b []float64, s float64, ci, bi int)

// HexAxpy applies axpy operations.
func HexAxpy(c, b []float64, s float64, ci, bi int)

// HexadecAxpy applies axpy operations.
func HexadecAxpy(c, b []float64, s float64, ci, bi int)

// WideAxpy applies axpy operations.
func WideAxpy(c, b []float64, s float64, ci, bi int)
