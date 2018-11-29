package vec

import (
	"math"
	"reflect"
	"testing"
)

const defaultBranchingFactor = 32

func Vec(values ...int) *Vector {
	return buildTree(defaultBranchingFactor, values...)
}

func buildTree(branchingFactor int, values ...int) *Vector {
	size := len(values)
	leafCount := int(math.Ceil(float64(size) / float64(branchingFactor)))
	leaves := make([][]interface{}, leafCount)

	for i := range leaves {
		leaves[i] = make([]interface{}, 0, branchingFactor)
	}

	for i, v := range values {
		n := i / branchingFactor
		leaves[n] = append(leaves[n], v)
	}

	var root []interface{}
	depth := 0
	level := leaves
	for {
		if len(level) == 1 {
			break
		}
		depth++

		parentCount := int(math.Ceil(float64(len(level)) / float64(branchingFactor)))
		parents := make([][]interface{}, parentCount)
		for i := range parents {
			parents[i] = make([]interface{}, branchingFactor)
		}

		for i, v := range level {
			n := i / branchingFactor
			c := i % branchingFactor
			parents[n][c] = v
		}
		level = parents
	}

	root = level[0]

	return &Vector{
		branchingFactor: branchingFactor,
		depth:           depth,
		size:            size,
		root:            root,
	}
}

type Vector struct {
	branchingFactor int
	size            int
	depth           int
	root            []interface{}
}

func (v *Vector) Count() int {
	return v.size
}

func lookup(v *Vector, key int) interface{} {
	var node []interface{} = v.root
	var size int = int(math.Pow(float64(v.branchingFactor), float64(v.depth)))

	for ; size > 1; size = size / v.branchingFactor {
		node = node[(key/size)%v.branchingFactor].([]interface{})
	}

	return node[key%v.branchingFactor]
}

func update(v *Vector, key int, value interface{}) *Vector {
	var newV Vector = *v
	var node []interface{} = make([]interface{}, len(v.root), v.branchingFactor)
	for i := range v.root {
		node[i] = v.root[i]
	}
	newV.root = node

	var size int = int(math.Pow(float64(v.branchingFactor), float64(v.depth)))

	for ; size > 1; size = size / v.branchingFactor {
		parent := node
		idx := (key / size) % v.branchingFactor
		refNode := node[idx].([]interface{})
		node = make([]interface{}, len(refNode), v.branchingFactor)
		for i := range refNode {
			node[i] = refNode[i]
		}
		parent[idx] = node
	}

	node[key%v.branchingFactor] = value
	return &newV
}

func raw(vec *Vector) []int {
	var vals = make([]int, 0, vec.size)
	for i := 0; i < vec.size; i++ {
		v := lookup(vec, i)
		vals = append(vals, v.(int))
	}

	return vals
}

func Test_Update(t *testing.T) {
	td := []struct {
		name  string
		index int
		value int
	}{
		{"first element", 0, 4},
		{"fifth element", 4, 4},
	}

	for _, tc := range td {
		t.Run(tc.name, func(t *testing.T) {
			vals := intValues[:7]
			v := buildTree(4, vals...)
			v2 := update(v, tc.index, tc.value)

			if reflect.DeepEqual(v, v2) {
				t.Errorf("update(v,3,4) = %v, should not equal %v", v2.root, v.root)
			}
		})
	}
}

func Test_Create(t *testing.T) {
	td := []struct {
		name   string
		length int
		depth  int
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

			if v.depth != tc.depth {
				t.Fatalf("v.depth = %v, want %v", v.depth, tc.depth)
			}

			raw := raw(v)
			if !reflect.DeepEqual(raw, vals) {
				t.Fatalf("raw(v) = %v, want %v", raw, vals)
			}
		})
	}
}

var intValues = []int{
	1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
	25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
	49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72,
	73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96,
	97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120,
	121, 122, 123, 124, 125, 126, 127, 128, 129,
}
