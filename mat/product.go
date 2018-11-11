package mat

// MulGaxpy is a General AXPY algorithm to multiple 2 matrices.
// Author: @james-bowman
func MulGaxpy(c, a, b *Dense) error {
	aCols := a.Columns()
	aRows := a.Rows()
	bCols := b.Columns()
	var data = c.data

	// IKJ
	for i := 0; i < aRows; i++ {
		for k := 0; k < aCols; k++ {
			s := a.data[i*aCols+k]
			for j := 0; j < bCols; j++ {
				data[i*bCols+j] += s * b.data[k*bCols+j]
			}
		}
	}
	return nil
}

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
		return MulNaiveIKJ(c, a, b)
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

func MulGonumUnroll(c, a, b *Dense) error {
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
			di := ar*bCols + bc
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
			data[di] = sum
		}
	}

	return nil
}

// MulNaive uses simple iteration to create the dot product of two matrices.
func MulNaiveIKJ(c, a, b *Dense) error {
	aCols := a.Columns()
	aRows := a.Rows()
	bCols := b.Columns()
	var data = c.data

	for ar := 0; ar < aRows; ar++ {
		for bc := 0; bc < bCols; bc++ {
			di := ar*bCols + bc
			for ac := 0; ac < aCols; ac++ {
				ai := ar*aCols + ac
				bi := ac*bCols + bc
				data[di] += a.data[ai] * b.data[bi]
			}
		}
	}

	return nil
}

func MulNaiveIJK(c, a, b *Dense) error {
	aCols := a.Columns()
	aRows := a.Rows()
	bCols := b.Columns()
	var data = c.data

	for ar := 0; ar < aRows; ar++ {
		for ac := 0; ac < aCols; ac++ {
			ai := ar*aCols + ac
			for bc := 0; bc < bCols; bc++ {
				di := ar*bCols + bc
				bi := ac*bCols + bc
				data[di] += a.data[ai] * b.data[bi]
			}
		}
	}

	return nil
}

func MulNaiveKIJ(c, a, b *Dense) error {
	aCols := a.Columns()
	aRows := a.Rows()
	bCols := b.Columns()
	var data = c.data

	for bc := 0; bc < bCols; bc++ {
		for ar := 0; ar < aRows; ar++ {
			di := ar*bCols + bc
			for ac := 0; ac < aCols; ac++ {
				ai := ar*aCols + ac
				bi := ac*bCols + bc
				data[di] += a.data[ai] * b.data[bi]
			}
		}
	}

	return nil
}

func MulNaiveJIK(c, a, b *Dense) error {
	aCols := a.Columns()
	aRows := a.Rows()
	bCols := b.Columns()
	var data = c.data

	for ac := 0; ac < aCols; ac++ {
		for ar := 0; ar < aRows; ar++ {
			ai := ar*aCols + ac
			for bc := 0; bc < bCols; bc++ {
				di := ar*bCols + bc
				bi := ac*bCols + bc
				data[di] += a.data[ai] * b.data[bi]
			}
		}
	}

	return nil
}

func MulNaiveJKI(c, a, b *Dense) error {
	aCols := a.Columns()
	aRows := a.Rows()
	bCols := b.Columns()
	var data = c.data

	for ac := 0; ac < aCols; ac++ {
		for bc := 0; bc < bCols; bc++ {
			bi := ac*bCols + bc
			for ar := 0; ar < aRows; ar++ {
				ai := ar*aCols + ac
				di := ar*bCols + bc
				data[di] += a.data[ai] * b.data[bi]
			}
		}
	}

	return nil
}

func MulNaiveKJI(c, a, b *Dense) error {
	aCols := a.Columns()
	aRows := a.Rows()
	bCols := b.Columns()

	var data = c.data

	for bc := 0; bc < bCols; bc++ {
		for ac := 0; ac < aCols; ac++ {
			bi := ac*bCols + bc
			for ar := 0; ar < aRows; ar++ {
				ai := ar*aCols + ac
				di := ar*bCols + bc
				data[di] += a.data[ai] * b.data[bi]
			}
		}
	}

	return nil
}

// MulBlockIJK is lazily implemented if any of the dims aren't divisible by block size it defers to naive IJK.
func MulBlockIJK(blockSize int) func(c, a, b *Dense) error {
	return func(c, a, b *Dense) error {
		aCols := a.Columns()
		aRows := a.Rows()
		bCols := b.Columns()
		var data = c.data

		if aCols%blockSize != 0 || aRows%blockSize != 0 || bCols%blockSize != 0 {
			return MulNaiveIJK(c, a, b)
		}

		for ar := 0; ar < aRows; ar += blockSize {
			for ac := 0; ac < aCols; ac += blockSize {
				for bc := 0; bc < bCols; bc += blockSize {
					for arb := ar; arb < ar+blockSize; arb++ {
						dib := arb * bCols
						aib := arb * aCols
						for acb := ac; acb < ac+blockSize; acb++ {
							bib := acb * bCols
							ai := aib + acb
							s := a.data[ai]
							for bcb := bc; bcb < bc+blockSize; bcb++ {
								di := dib + bcb
								bi := bib + bcb
								data[di] += s * b.data[bi]
							}
						}
					}
				}
			}
		}
		return nil
	}
}

// MulBlockUnrollIJK is lazily implemented if any of the dims aren't divisible by block size it defers to naive IJK.
func MulBlockUnrollIJK(blockSize int) func(c, a, b *Dense) error {
	return func(c, a, b *Dense) error {
		aCols := a.Columns()
		aRows := a.Rows()
		bCols := b.Columns()
		var data = c.data

		if aCols%blockSize != 0 || aRows%blockSize != 0 || bCols%blockSize != 0 {
			return MulNaiveIJK(c, a, b)
		}

		for ar := 0; ar < aRows; ar += blockSize {
			for ac := 0; ac < aCols; ac += blockSize {
				for bc := 0; bc < bCols; bc += blockSize {
					for arb := ar; arb < ar+blockSize; arb++ {
						dib := arb * bCols
						aib := arb * aCols
						for acb := ac; acb < ac+blockSize; acb++ {
							bib := acb * bCols
							ai := aib + acb
							s := a.data[ai]
							for bcb := bc; bcb < bc+blockSize; bcb += 4 {
								di := dib + bcb
								bi := bib + bcb
								data[di] += s * b.data[bi]
								data[di+1] += s * b.data[bi+1]
								data[di+2] += s * b.data[bi+2]
								data[di+3] += s * b.data[bi+3]
							}
						}
					}
				}
			}
		}
		return nil
	}
}

// MulBlockIJK is lazily implemented if any of the dims aren't divisible by block size it defers to naive IJK.
func MulBlockFetchIJK(blockSize int) func(c, a, b *Dense) error {
	return func(c, a, b *Dense) error {
		aCols := a.Columns()
		aRows := a.Rows()
		bCols := b.Columns()
		row := make([]float64, blockSize)
		var data = c.data

		if aCols%blockSize != 0 || aRows%blockSize != 0 || bCols%blockSize != 0 {
			return MulNaiveIJK(c, a, b)
		}

		for ar := 0; ar < aRows; ar += blockSize {
			for ac := 0; ac < aCols; ac += blockSize {
				for bc := 0; bc < bCols; bc += blockSize {
					for arb := ar; arb < ar+blockSize; arb++ {
						dib := arb * bCols
						aib := ar * aCols
						for i := range row {
							row[i] = a.data[aib+ac+i]
						}
						for bcb := bc; bcb < bc+blockSize; bcb++ {
							var sum float64
							for i, e := range row {
								sum += e * b.data[(ac+i)*bCols+bcb]
							}
							di := dib + bcb
							data[di] += sum
						}
					}
				}
			}
		}
		return nil
	}
}

// MulUnroll uses unrolled loops to create the dot product of two matrices.
func MulUnroll(c, a, b *Dense) error {
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
			di := ar*bCols + bc
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
			data[di] = sum
		}
	}

	return nil
}
