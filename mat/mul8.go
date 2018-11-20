package mat

// AxpyLoop loops around an axpy function.
func AxpyLoop(c, b []float64, s float64, bc, cib, bib, blockSize int)

// QuadAxpy applies 4 axpy derivatives as vectorised operations.
func QuadAxpy(c, b []float64, s float64, ci, bi int)

func HexAxpy(c, b []float64, s float64, ci, bi int)

func HexadecAxpy(c, b []float64, s float64, ci, bi int)

func WideAxpy(c, b []float64, s float64, ci, bi int)
