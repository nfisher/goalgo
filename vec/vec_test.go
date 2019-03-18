package vec

import (
	"math"
	"reflect"
	"testing"
)

var Value int
var VecValue *Vector

func Test_Lookup(t *testing.T) {
	v := Vec(intValues...)

	actual, ok := Lookup(v, 11).(int)
	if !ok {
		t.Errorf("ok = %v, want true", ok)
	}

	if actual != 12 {
		t.Errorf("Lookup2(v,11) = %v, want %v", actual, 12)
	}
}

func Test_Update(t *testing.T) {
	v := Vec(intValues...)
	v2 := Update(v, 10, 8)

	if Lookup(v, 10) != 11 {
		t.Errorf("Lookup(v, %v) = %v, want %v", 10, Lookup(v, 10), 11)
	}

	if Lookup(v2, 10) != 8 {
		t.Errorf("Lookup(v2, %v) = %v, want %v", 10, Lookup(v2, 10), 8)
	}

	if Lookup(v2, 0) != 1 {
		t.Errorf("Lookup(v2, %v) = %v, want %v", 0, Lookup(v2, 0), 1)
	}

	if Lookup(v2, 127) != 128 {
		t.Errorf("Lookup(v2, %v) = %v, want %v", 127, Lookup(v2, 127), 128)
	}
}

func Test_UpdateDigit(t *testing.T) {
	td := []struct {
		name  string
		index int
		value int
	}{
		{"first element", 0, 2},
		{"fifth element", 4, 6 },
	}

	for _, tc := range td {
		t.Run(tc.name, func(t *testing.T) {
			values := intValues[:48]
			v := buildTree(4, values...)
			v2 := update(v, tc.index, tc.value)

			if reflect.DeepEqual(v, v2) {
				t.Errorf("update(v,%v,%v) = %v, should not equal %v", tc.index, tc.value, v2.root, v.root)
			}

			if lookup(v2, tc.index) != tc.value {
				t.Errorf("lookup(v, %v) = %v, want %v", tc.index, lookup(v2, tc.index).(int), tc.value)
			}
		})
	}
}

func Test_BuildTree(t *testing.T) {
	td := []struct {
		name   string
		length int
		depth  uint
	}{
		{"single layer", 2, 0},
		{"two layers", 5, 1},
		{"three layers", 17, 2},
		{"four layers", 65, 3},
	}

	for _, tc := range td {
		t.Run(tc.name, func(t *testing.T) {
			var vals = intValues[:tc.length]
			v := buildTree(4, vals...)

			if v.Count() != tc.length {
				t.Fatalf("v.Count() = %v, want %v", v.Count(), tc.length)
			}

			if v.Depth() != tc.depth {
				t.Fatalf("v.Depth() = %v, want %v", v.Depth(), tc.depth)
			}

			raw := raw(v)
			if !reflect.DeepEqual(raw, vals) {
				t.Fatalf("raw(v) = %v, want %v", raw, vals)
			}
		})
	}
}

func raw(vec *Vector) []int {
	var values = make([]int, 0, vec.Count())
	for i := 0; i < vec.Count(); i++ {
		v := lookup(vec, i)
		values = append(values, v.(int))
	}

	return values
}

var intValues = []int{
	1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
	25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
	49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72,
	73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96,
	97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120,
	121, 122, 123, 124, 125, 126, 127, 128, 129,
}

// Digit based update for testing purposes.
func update(v *Vector, key int, value interface{}) *Vector {
	var newV = *v
	var node = make([]interface{}, len(v.root), v.branchingFactor)
	copy(node, v.root)
	newV.root = node

	var size = int(math.Pow(float64(v.branchingFactor), float64(v.Depth())))

	for ; size > 1; size = size / v.branchingFactor {
		parent := node
		idx := (key / size) % v.branchingFactor
		refNode := node[idx].([]interface{})
		node = make([]interface{}, len(refNode), v.branchingFactor)
		copy(node, refNode)
		parent[idx] = node
	}

	node[key%v.branchingFactor] = value
	return &newV
}

// Digit based lookup for testing purposes.
func lookup(v *Vector, key int) interface{} {
	var node = v.root
	var size = int(math.Pow(float64(v.branchingFactor), float64(v.depth)))

	for ; size > 1; size = size / v.branchingFactor {
		node = node[(key/size)%v.branchingFactor].([]interface{})
	}

	return node[key%v.branchingFactor]
}
