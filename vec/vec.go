// Package vec is a Clojure inspired immutable vector.
// Currently it provides only an integer based vector.
//
// As a quick start:
//			v := Vec(intValues...)
//			v2 := Update(v, 10, 8)
//			i := Lookup(v, 10)
//			fmt.Println(i)
//			// Output:
//			// 11
package vec

import (
	"math"
)

const defaultBranchingFactor = 32

// Vec builds an immutable vector using a COW tree.
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
	var depth uint = 0
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

// Vector is the root structure of an immutable vector.
type Vector struct {
	branchingFactor int
	size            int
	depth           uint
	root            []interface{}
}

// Count returns the number of entries in the vector.
func (v *Vector) Count() int {
	return v.size
}

// Depth returns the tree depth of the vector.
func (v *Vector) Depth() uint {
	return v.depth
}

const bits uint = 5
const width int = 1 << bits
const mask = width - 1

// Lookup retrieves the value in vector stored in position key.
func Lookup(v *Vector, idx int) interface{} {
	var node = v.root
	var shift = bits * v.depth

	for level := shift; level > 0; level -= bits {
		node = node[(idx>>uint(level))&mask].([]interface{})
	}

	return node[idx&mask]
}

// Update modifies the value in the vector v at index idx.
func Update(v *Vector, idx int, value interface{}) *Vector {
	var newV = *v // copy value
	var node = v.root
	var shift = bits * v.depth

	newV.root = make([]interface{}, len(v.root), v.branchingFactor)
	copy(newV.root, v.root)
	var pNode = newV.root

	for level := shift; level > 0; level -= bits {
		pos := (idx >> uint(level)) & mask
		node = node[pos].([]interface{})
		newNode := make([]interface{}, len(node), v.branchingFactor)
		pNode[pos] = newNode
		pNode = newNode
		copy(newNode, node)
	}

	pNode[idx&mask] = value
	return &newV
}
