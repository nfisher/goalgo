package mat

type Dense struct {
	columns int
	rows    int
	data    []float64
}

func (d *Dense) Columns() int {
	return d.columns
}

func (d *Dense) Rows() int {
	return d.rows
}

func (d *Dense) Raw() []float64 {
	return d.data
}

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

// Dot creates the dot product of two matrices.
func Dot(a, b *Dense) (*Dense, error) {
	var data []float64
	if a.Columns() >= 8 {
		data = largeDot(a, b)
	} else {
		data = smallDot(a, b)
	}
	c := NewDense(a.Rows(), b.Columns(), data)
	return c, nil
}

// < 640ns/op
func largeDot(a, b *Dense) []float64 {
	acols := a.Columns()
	bcols := b.Columns()
	data := make([]float64, 0, a.Rows()*bcols)
	bounded := acols
	remainder := acols % 8
	if remainder != 0 {
		bounded = acols / 8 * 8
	}
	for ar := 0; ar < a.Rows(); ar++ {
		for bc := 0; bc < bcols; bc++ {
			var sum float64
			for ac := 0; ac < bounded; ac += 8 {
				ai := ar*acols + ac
				sum += a.data[ai] * b.data[bc]
				sum += a.data[ai+1] * b.data[bc+1*bcols]
				sum += a.data[ai+2] * b.data[bc+2*bcols]
				sum += a.data[ai+3] * b.data[bc+3*bcols]
				sum += a.data[ai+4] * b.data[bc+4*bcols]
				sum += a.data[ai+5] * b.data[bc+5*bcols]
				sum += a.data[ai+6] * b.data[bc+6*bcols]
				sum += a.data[ai+7] * b.data[bc+7*bcols]
			}
			for ac := bounded; ac < acols; ac++ {
				ai := ar*acols + ac
				bi := ac*bcols + bc
				sum += a.data[ai] * b.data[bi]
			}
			data = append(data, sum)
		}
	}
	return data
}

// < 977ns/op
func smallDot(a, b *Dense) []float64 {
	var data = make([]float64, 0, a.Rows()*b.Columns())
	acols := a.Columns()
	bcols := b.Columns()
	for ar := 0; ar < a.Rows(); ar++ {
		for bc := 0; bc < bcols; bc++ {
			var sum float64
			for ac := 0; ac < acols; ac++ {
				ai := ar*acols + ac
				bi := ac*bcols + bc
				sum += a.data[ai] * b.data[bi]
			}
			data = append(data, sum)
		}
	}
	return data
}
