package graph

import (
	"errors"
)

// Graph interface for various forms of graphs.
type Graph interface {
	Vertex(edges ...int) (id int, err error)
	Edge(v, w int) error
	Adjacent(v int) ([]int, error)
	Vertices() int
	Edges() int
}

// Directed returns a new directed graph.
func Directed() *AdjacencyList {
	return &AdjacencyList{}
}

// AdjacencyList is an adjacency list using a set.
type AdjacencyList struct {
	list  [][]int
	edges int
}

var (
	// ErrCannotAddVertices is emitted when an invalid edge is specified in the creation of a new vertex.
	ErrCannotAddVertices = errors.New("graph: cannot add vertices with invalid edges")
	// ErrCannotAddEdge is emitted when one or more of the vertices in an edge are invalid/non-existant.
	ErrCannotAddEdge = errors.New("graph: cannot add edge with invalid vertices")
	// ErrVertexNotFound is emitted when a vertex does not exist and therefore has no edge set.
	ErrVertexNotFound = errors.New("graph: vertex not found")
	// ErrNoVertices is emitted when the graph cannot carry out a calculation due to an absence of vertices.
	ErrNoVertices = errors.New("graph: no vertices in graph")
)

// Average returns the average degree of the list.
func Average(g Graph) (float64, error) {
	if g.Vertices() == 0 {
		return -1.0, ErrNoVertices
	}
	return 2.0 * float64(g.Edges()) / float64(g.Vertices()), nil
}

// Max returns the max degree of the list.
func Max(g Graph) int {
	var max int
	for i := 0; i < g.Vertices(); i++ {
		adj, _ := g.Adjacent(i)
		if len(adj) > max {
			max = len(adj)
		}
	}
	return max
}

// Degree returns the number of edges connected to the vertex.
func Degree(g Graph, v int) (int, error) {
	adj, err := g.Adjacent(v)
	if err != nil {
		return -1, err
	}

	return len(adj), nil
}

// Vertex adds a new vertex, optionally with the specified edges.
func (as *AdjacencyList) Vertex(edges ...int) (id int, err error) {
	var edgeset []int
	l := len(as.list)
	for _, i := range edges {
		if i >= l {
			return -1, ErrCannotAddVertices
		}
		edgeset = append(edgeset, i)
	}

	as.edges += len(edgeset)
	as.list = append(as.list, edgeset)

	return l, nil
}

// Edge adds an edge from v to w.
func (as *AdjacencyList) Edge(v, w int) error {
	l := len(as.list)
	if v >= l {
		return ErrCannotAddEdge
	}

	if w >= l {
		return ErrCannotAddEdge
	}

	as.list[v] = append(as.list[v], w)
	as.edges++

	return nil
}

// Adjacent returns all vertices adjacent to this vertex.
func (as *AdjacencyList) Adjacent(v int) ([]int, error) {
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
func (as *AdjacencyList) Vertices() int {
	return len(as.list)
}

// Edges returns the number edges in the list.
func (as *AdjacencyList) Edges() int {
	return as.edges
}
