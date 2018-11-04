package mat_test

import (
	"reflect"
	"testing"
)

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

func NewDense(r, c int, data []float64) *Dense {
	if data == nil {
		data = make([]float64, c*r)
	}

	return &Dense{
		columns: c,
		rows:    r,
		data:    data,
	}
}

func Dot(a, b *Dense) (*Dense, error) {
	var data = make([]float64, 0, a.Rows()*b.Columns())

	for ar := 0; ar < a.Rows(); ar++ {
		var sum float64 = 0
		for ac := 0; ac < a.Columns(); ac++ {
			ai := ar*a.Columns() + ac
			bi := ac % b.Rows()
			/*
				log.Printf("a[%v]*b[%v]", ai, bi)
				log.Printf("a[%v,%v]*b[%v,%v]\n", ar, ac, ac%b.Rows(), ar%b.Columns())
			*/
			sum += a.data[ai] * b.data[bi]
		}
		data = append(data, sum)
	}

	c := NewDense(a.Rows(), b.Columns(), data)
	return c, nil
}

func Test_Dot(t *testing.T) {
	td := []struct {
		name     string
		arows    int
		acols    int
		adata    []float64
		brows    int
		bcols    int
		bdata    []float64
		expected []float64
		err      error
	}{
		{"[mat] x [vector]", 3, 2, []float64{1, 2, 3, 4, 5, 6}, 2, 1, []float64{2, 4}, []float64{10, 22, 34}, nil},
		//{"[mat] x [mat]", 3, 4, []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, 3, 2, []float64{2, 7, 1, 2, 3, 6}, []float64{13, 29, 31, 74, 49, 119, 67, 164}, nil},
	}

	for _, tc := range td {
		t.Run(tc.name, func(t *testing.T) {
			a := NewDense(tc.arows, tc.acols, tc.adata)
			b := NewDense(tc.brows, tc.bcols, tc.bdata)
			c, err := Dot(a, b)
			if err != tc.err {
				t.Fatalf("err = %v, want %v", err, tc.err)
			}

			if c.Rows() != tc.arows {
				t.Errorf("rows = %v, want %v", c.Rows(), tc.arows)
			}

			if c.Columns() != tc.bcols {
				t.Errorf("cols = %v, want %v", c.Columns(), tc.bcols)
			}

			if !reflect.DeepEqual(c.Raw(), tc.expected) {
				t.Errorf("raw = %v, want %v", c.Raw(), tc.expected)
			}
		})
	}
}
