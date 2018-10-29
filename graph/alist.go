package graph

import (
	"errors"

	"github.com/nfisher/goalgo/sets/intset"
)

// AdjacencySet is an adjacency list using a set.
type AdjacencySet struct {
	list  []intset.Set
	edges int
}

var (
	// ErrCannotAddVertices is emitted when an invalid edge is specified in the creation of a new vertex.
	ErrCannotAddVertices = errors.New("graph: cannot add vertices with invalid edges")
	// ErrCannotAddEdge is emitted when one or more of the vertices in an edge are invalid/non-existant.
	ErrCannotAddEdge = errors.New("graph: cannot add edge with invalid vertices")
	// ErrVertexNotFound is emitted when a vertex does not exist and therefore has no edge set.
	ErrVertexNotFound = errors.New("graph: vertex not found")
)

// Vertex adds a new vertex, optionally with the specified edges.
func (as *AdjacencySet) Vertex(edges ...int) (id int, err error) {
	edgeset := intset.New()
	l := len(as.list)
	for _, i := range edges {
		if i >= l {
			return -1, ErrCannotAddVertices
		}
		edgeset.Add(i)
	}

	as.edges += len(edgeset)
	as.list = append(as.list, edgeset)

	return l, nil
}

// Edge adds an edge from v to w.
func (as *AdjacencySet) Edge(v, w int) error {
	l := len(as.list)
	if v >= l {
		return ErrCannotAddEdge
	}

	if w >= l {
		return ErrCannotAddEdge
	}

	as.list[v].Add(w)
	as.edges++

	return nil
}

// Adjacent returns all vertices adjacent to this vertex.
func (as *AdjacencySet) Adjacent(v int) ([]int, error) {
	if v >= len(as.list) {
		return nil, ErrVertexNotFound
	}

	var a []int
	for k := range as.list[v] {
		a = append(a, k)
	}

	return a, nil
}

// Vertices returns the number of vertices in the list.
func (as *AdjacencySet) Vertices() int {
	return len(as.list)
}

// Edges returns the number edges in the list.
func (as *AdjacencySet) Edges() int {
	return as.edges
}
