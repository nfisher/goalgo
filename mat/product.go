package mat

// Author: @james-bowman
func MulGonumNaivePrefetch(c, a, b *Dense) error {
	aCols := a.Columns()
	aRows := a.Rows()
	bCols := b.Columns()
	row := make([]float64, aCols)
	var data = c.data
	for ar := 0; ar < aRows; ar++ {
		for i := range row {
			row[i] = a.At(ar, i)
		}
		for bc := 0; bc < bCols; bc++ {
			var sum float64
			for i, e := range row {
				sum += e * b.At(i, bc)
			}
			di := ar*bCols + bc
			data[di] = sum
		}
	}

	return nil
}

func MulGonumNaive(c, a, b *Dense) error {
	aCols := a.Columns()
	aRows := a.Rows()
	bCols := b.Columns()
	for ar := 0; ar < aRows; ar++ {

		for bc := 0; bc < bCols; bc++ {
			var sum float64
			for ac := 0; ac < aCols; ac++ {
				sum += a.At(ar, ac) * b.At(ac, bc)
			}
			c.Set(ar, bc, sum)
		}
	}

	return nil
}

// MulMultiplePrefetch2 uses simple iteration to create the dot product of two matrices.
// Author: @james-bowman
func MulMultiplePrefetch2(c, a, b *Dense) error {
	aCols := a.Columns()
	aRows := a.Rows()
	bCols := b.Columns()
	var data = c.data

	if aRows%4 != 0 {
		return MulNaive(c, a, b)
	}

	var sum0 float64
	var sum1 float64
	var sum2 float64
	var sum3 float64
	buffer0 := make([]float64, aCols)
	buffer1 := make([]float64, aCols)
	buffer2 := make([]float64, aCols)
	buffer3 := make([]float64, aCols)

	for i := 0; i < aRows; i += 4 {
		for j := 0; j < aCols; j++ {
			buffer0[j] = a.data[i*aCols+j]
			buffer1[j] = a.data[(i+1)*aCols+j]
			buffer2[j] = a.data[(i+2)*aCols+j]
			buffer3[j] = a.data[(i+3)*aCols+j]
		}
		for j := 0; j < bCols; j++ {
			sum0 = 0
			sum1 = 0
			sum2 = 0
			sum3 = 0
			for k := 0; k < aCols; k++ {
				bi := k*bCols + j
				sum0 += buffer0[k] * b.data[bi]
				sum1 += buffer1[k] * b.data[bi]
				sum2 += buffer2[k] * b.data[bi]
				sum3 += buffer3[k] * b.data[bi]
			}
			data[i*bCols+j] = sum0
			data[(i+1)*bCols+j] = sum1
			data[(i+2)*bCols+j] = sum2
			data[(i+3)*bCols+j] = sum3
		}
	}
	return nil
}

func MulGonumStride(c, a, b *Dense) error {
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

// MulNaive uses simple iteration to create the dot product of two matrices.
func MulNaive(c, a, b *Dense) error {
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

func MulDaxpy(c, a, b *Dense) error {
	aCols := a.Columns()
	aRows := a.Rows()
	bCols := b.Columns()

	for ac := 0; ac < aCols; ac++ {
		for ar := 0; ar < ar; ar++ {
			da := a.At(ar, ac)
			daxpy(aRows, da, c, bCols, b, aCols)
		}
	}

	return nil
}

func daxpy(n int, da float64, dx *Dense, incx int, accum *Dense, incAccum int) {
	if incx == 1 && incAccum == 1 {
		m := n % 4
		if m != 0 {
			for i := 0; i < m; i++ {
				accum.data[i] = accum.data[i] + da*dx.data[i]
			}
		}
		if n < 4 {
			return
		}
		mp1 := m + 1
		for i := mp1; i < n; i += 4 {
			accum.data[i] = accum.data[i] + da*dx.data[i]
			accum.data[i+1] = accum.data[i+1] + da*dx.data[i+1]
			accum.data[i+2] = accum.data[i+2] + da*dx.data[i+2]
			accum.data[i+3] = accum.data[i+3] + da*dx.data[i+3]
		}
	} else {
		ix := 0
		iy := 0
		if incx < 0 {
			ix = (-n+1)*incx + 1
		}
		if incAccum < 0 {
			iy = (-n+1)*incAccum + 1
		}
		for i := 0; i < n; i++ {
			accum.data[iy] = accum.data[iy] + da*dx.data[ix]
			ix = ix + incx
			iy = iy + incAccum
		}
	}
}

// MulStride uses unrolled loops to create the dot product of two matrices.
func MulStride(c, a, b *Dense) error {
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
