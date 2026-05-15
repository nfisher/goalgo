//go:build !amd64 && !arm64

package mat

func dotProductUnitary(a, b []float64) float64 {
	var sum float64
	for i, av := range a {
		sum += av * b[i]
	}
	return sum
}

func dotProductQuad(a, b0, b1, b2, b3 []float64) (s0, s1, s2, s3 float64) {
	for i, av := range a {
		s0 += av * b0[i]
		s1 += av * b1[i]
		s2 += av * b2[i]
		s3 += av * b3[i]
	}
	return s0, s1, s2, s3
}

func dotProductOct(a, b0, b1, b2, b3, b4, b5, b6, b7 []float64) (s0, s1, s2, s3, s4, s5, s6, s7 float64) {
	for i, av := range a {
		s0 += av * b0[i]
		s1 += av * b1[i]
		s2 += av * b2[i]
		s3 += av * b3[i]
		s4 += av * b4[i]
		s5 += av * b5[i]
		s6 += av * b6[i]
		s7 += av * b7[i]
	}
	return s0, s1, s2, s3, s4, s5, s6, s7
}
