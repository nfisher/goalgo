package mat

// NewDense creates a new dense matrices of the supplied dimensions.
func NewDense(rows, columns int, data []float64) *Dense {
	if data == nil {
		data = make([]float64, columns*rows)
	}

	return &Dense{
		rows:    rows,
		columns: columns,
		data:    data,
	}
}

// Dense matrix.
type Dense struct {
	columns int
	rows    int
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
	idx := i*d.columns + j
	return d.data[idx]
}

// Set a specific value in the matrix.
func (d *Dense) Set(i, j int, v float64) {
	idx := i*d.columns + j
	d.data[idx] = v
}
