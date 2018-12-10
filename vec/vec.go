package vec

import (
	"math"
)

const defaultBranchingFactor = 32

func Vec(values ...int) *Vector {
	return BuildTree(defaultBranchingFactor, values...)
}

func BuildTree(branchingFactor int, values ...int) *Vector {
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

type Vector struct {
	branchingFactor int
	size            int
	depth           uint
	root            []interface{}
}

func (v *Vector) Count() int {
	return v.size
}

func (v *Vector) Depth() uint {
	return v.depth
}

const Bits uint = 5
const Width int = 1 << Bits
const Mask = Width - 1

func Lookup(v *Vector, key int) interface{} {
	var node = v.root
	var shift = Bits * v.depth

	for level := shift; level > 0; level -= Bits {
		node = node[(key >> uint(level)) & Mask].([]interface{})
	}

	return node[key & Mask]
}

func LookupDigit(v *Vector, key int) interface{} {
	var node = v.root
	var size = int(math.Pow(float64(v.branchingFactor), float64(v.depth)))

	for ; size > 1; size = size / v.branchingFactor {
		node = node[(key/size)%v.branchingFactor].([]interface{})
	}

	return node[key%v.branchingFactor]
}

func Update(v *Vector, key int, value interface{}) *Vector {
	var newV = *v
	var node = make([]interface{}, len(v.root), v.branchingFactor)
	for i := range v.root {
		node[i] = v.root[i]
	}
	newV.root = node

	var size = int(math.Pow(float64(v.branchingFactor), float64(v.Depth())))

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

