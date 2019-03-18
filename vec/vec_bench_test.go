package vec

import "testing"

func Benchmark_Update(b *testing.B) {
	td := []struct {
		name string
		fn   func(*Vector, int, interface{}) *Vector
	}{
		{"digit based", update},
		{"bit based", Update},
	}

	v := Vec(intValues...)
	for _, tc := range td {
		b.Run(tc.name, func(b *testing.B) {
			var v2 *Vector
			for i := 0; i < b.N; i++ {
				v2 = tc.fn(v, 120, 99)
			}
			VecValue = v2
		})
	}
}

func Benchmark_Lookup(b *testing.B) {
	td := []struct {
		name string
		fn   func(*Vector, int) interface{}
	}{
		{"bit based", Lookup},
		{"digit based", lookup},
	}

	v := Vec(intValues...)
	for _, tc := range td {
		b.Run(tc.name, func(b *testing.B) {
			var sum = 0
			for i := 0; i < b.N; i++ {
				value := tc.fn(v, 120)
				sum += value.(int)
			}
			Value = sum
		})
	}
}

