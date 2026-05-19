package mat

// NewDense creates a new dense matrix of the supplied dimensions.
func NewDense(rows, columns int, data []float64) *Dense {
	return NewDenseWithStride(rows, columns, columns, data)
}

// NewDenseWithStride creates a new dense matrix with the supplied row stride.
func NewDenseWithStride(rows, columns, stride int, data []float64) *Dense {
	validateDenseShape(rows, columns, stride)
	required := denseDataLen(rows, columns, stride)
	if data == nil {
		data = make([]float64, required)
	} else if len(data) < required {
		panic("mat: data length too short for dense matrix")
	}

	return &Dense{
		rows:    rows,
		columns: columns,
		stride:  stride,
		data:    data,
	}
}

// Dense matrix.
type Dense struct {
	columns int
	rows    int
	stride  int
	data    []float64
}

// Columns returns the number of the columns of the matrix.
func (d *Dense) Columns() int {
	return d.columns
}

// Rows returns the number of rows of the matrix.
func (d *Dense) Rows() int {
	return d.rows
}

// Stride returns the number of elements between adjacent row starts.
func (d *Dense) Stride() int {
	return d.stride
}

// Raw returns the raw backing array of the matrix.
func (d *Dense) Raw() []float64 {
	return d.data
}

// Dims returns the rows and columns of the matrix.
func (d *Dense) Dims() (rows, columns int) {
	return d.rows, d.columns
}

// At returns the data at position (i,j).
func (d *Dense) At(i, j int) float64 {
	idx := i*d.stride + j
	return d.data[idx]
}

// Set a specific value in the matrix.
func (d *Dense) Set(i, j int, v float64) {
	idx := i*d.stride + j
	d.data[idx] = v
}

// View returns a mutable submatrix view backed by the same data.
func (d *Dense) View(row, column, rows, columns int) *Dense {
	if row < 0 || column < 0 || rows < 0 || columns < 0 || row+rows > d.rows || column+columns > d.columns {
		panic("mat: dense view out of range")
	}
	if rows == 0 || columns == 0 {
		return &Dense{
			rows:    rows,
			columns: columns,
			stride:  d.stride,
			data:    nil,
		}
	}
	offset := row*d.stride + column
	return &Dense{
		rows:    rows,
		columns: columns,
		stride:  d.stride,
		data:    d.data[offset:],
	}
}

func (d *Dense) isCompact() bool {
	return d.stride == d.columns
}

func denseAllCompact(matrices ...*Dense) bool {
	for _, matrix := range matrices {
		if !matrix.isCompact() {
			return false
		}
	}
	return true
}

func validateDenseShape(rows, columns, stride int) {
	if rows < 0 || columns < 0 {
		panic("mat: negative dense dimension")
	}
	if stride < columns {
		panic("mat: dense stride smaller than columns")
	}
}

func denseDataLen(rows, columns, stride int) int {
	if rows == 0 || columns == 0 {
		return 0
	}
	return (rows-1)*stride + columns
}
