//go:build amd64

package mat

//go:noescape
func dotProductUnitary(a, b []float64) float64

//go:noescape
func dotProductQuad(a, b0, b1, b2, b3 []float64) (s0, s1, s2, s3 float64)

func dotProductOct(a, b0, b1, b2, b3, b4, b5, b6, b7 []float64) (s0, s1, s2, s3, s4, s5, s6, s7 float64) {
	s0, s1, s2, s3 = dotProductQuad(a, b0, b1, b2, b3)
	s4, s5, s6, s7 = dotProductQuad(a, b4, b5, b6, b7)
	return s0, s1, s2, s3, s4, s5, s6, s7
}
