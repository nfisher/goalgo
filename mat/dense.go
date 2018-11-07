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

func (d *Dense) Dims() (rows, columns int) {
	return d.rows, d.columns
}

func (d *Dense) At(i, j int) float64 {
	idx := i*d.columns + j
	return d.data[idx]
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

func DotGonumNaive(c, a, b *Dense) error {
	aCols := a.Columns()
	aRows := a.Rows()
	bCols := b.Columns()
	var data = c.data
	for ar := 0; ar < aRows; ar++ {
		for bc := 0; bc < bCols; bc++ {
			var sum float64
			for ac := 0; ac < aCols; ac++ {
				sum += a.At(ar, ac) * b.At(ac, bc)
			}
			di := ar*bCols + bc
			data[di] = sum
		}
	}

	return nil
}

func DotGonumStride(c, a, b *Dense) error {
	const stride = 8
	aCols := a.Columns()
	bCols := b.Columns()
	var data = c.data
	bounded := aCols
	remainder := aCols % stride
	if remainder != 0 {
		bounded = aCols / stride * stride
	}
	for ar := 0; ar < a.Rows(); ar++ {
		for bc := 0; bc < bCols; bc++ {
			var sum float64
			for ac := 0; ac < bounded; ac += stride {
				ac0 := ac
				ac1 := ac0 + 1
				ac2 := ac0 + 2
				ac3 := ac0 + 3
				ac4 := ac0 + 4
				ac5 := ac0 + 5
				ac6 := ac0 + 6
				ac7 := ac0 + 7
				sum += a.At(ar, ac0) * b.At(ac0, bc)
				sum += a.At(ar, ac1) * b.At(ac1, bc)
				sum += a.At(ar, ac2) * b.At(ac2, bc)
				sum += a.At(ar, ac3) * b.At(ac3, bc)
				sum += a.At(ar, ac4) * b.At(ac4, bc)
				sum += a.At(ar, ac5) * b.At(ac5, bc)
				sum += a.At(ar, ac6) * b.At(ac6, bc)
				sum += a.At(ar, ac7) * b.At(ac7, bc)
			}

			for ac := bounded; ac < aCols; ac++ {
				sum += a.At(ar, ac) * b.At(ac, bc)
			}
			di := ar*bCols + bc
			data[di] = sum
		}
	}

	return nil
}

// DotNaive uses simple iteration to create the dot product of two matrices.
func DotNaive(c, a, b *Dense) error {
	aCols := a.Columns()
	aRows := a.Rows()
	bCols := b.Columns()
	var data = c.data

	for ar := 0; ar < aRows; ar++ {
		for bc := 0; bc < bCols; bc++ {
			var sum float64
			for ac := 0; ac < aCols; ac++ {
				ai := ar*aCols + ac
				bi := ac*bCols + bc
				sum += a.data[ai] * b.data[bi]
			}
			di := ar*bCols + bc
			data[di] = sum
		}
	}

	return nil
}

func DotDaxpy(c, a, b *Dense) error {
	aCols := a.Columns()
	aRows := a.Rows()
	bCols := b.Columns()
	var data = c.data

	for ac := 0; ac < aCols; ac++ {
		for bc := 0; bc < bCols; bc++ {
			bi := ac*bCols + bc
			bv := b.data[bi]
			var di = bc
			var ai = ac
			for ar := 0; ar < aRows; ar++ {
				data[di] += a.data[ai] * bv
				ai += aCols
				di += bCols
			}
		}
	}

	return nil
}

func daxpy(n int, da float64, dx *Dense, incx int, dy *Dense, incy int) {
	if incx == 1 && incy == 1 {
		m := n % 4
		if m != 0 {
			for i := 0; i < m; i++ {
				dy.data[i] = dy.data[i] + da*dx.data[i]
			}
		}
		if n < 4 {
			return
		}
		mp1 := m + 1
		for i := mp1; i < n; i += 4 {
			dy.data[i] = dy.data[i] + da*dx.data[i]
			dy.data[i+1] = dy.data[i+1] + da*dx.data[i+1]
			dy.data[i+2] = dy.data[i+2] + da*dx.data[i+2]
			dy.data[i+3] = dy.data[i+3] + da*dx.data[i+3]
		}
	} else {
		ix := 0
		iy := 0
		if incx < 0 {
			ix = (-n+1)*incx + 1
		}
		if incy < 0 {
			iy = (-n+1)*incy + 1
		}
		for i := 0; i < n; i++ {
			dy.data[iy] = dy.data[iy] + da*dx.data[ix]
			ix = ix + incx
			iy = iy + incy
		}
	}
}

// DotStride uses unrolled loops to create the dot product of two matrices.
func DotStride(c, a, b *Dense) error {
	aCols := a.Columns()
	aRows := a.Rows()
	bCols := b.Columns()
	data := c.data
	const stride = 4
	bounded := aCols
	remainder := aCols % stride
	if remainder != 0 {
		bounded = aCols / stride * stride
	}
	var sum0 float64
	var sum1 float64
	var sum2 float64
	var sum3 float64
	for ar := 0; ar < aRows; ar++ {
		for bc := 0; bc < bCols; bc++ {
			var sum float64
			for ac := 0; ac < bounded; ac += stride {
				b0 := ac*bCols + bc
				b1 := b0 + bCols
				b2 := b1 + bCols
				b3 := b2 + bCols

				a0 := ar*aCols + ac
				a1 := a0 + 1
				a2 := a1 + 1
				a3 := a2 + 1

				av0 := a.data[a0]
				av1 := a.data[a1]
				av2 := a.data[a2]
				av3 := a.data[a3]

				bv0 := b.data[b0]
				bv1 := b.data[b1]
				bv2 := b.data[b2]
				bv3 := b.data[b3]

				sum0 = av0 * bv0
				sum1 = av1 * bv1
				sum2 = av2 * bv2
				sum3 = av3 * bv3
				sum += sum0 + sum1 + sum2 + sum3
			}
			for ac := bounded; ac < aCols; ac++ {
				ai := ar*aCols + ac
				bi := ac*bCols + bc
				sum += a.data[ai] * b.data[bi]
			}
			di := ar*bCols + bc
			data[di] = sum
		}
	}

	return nil
}
