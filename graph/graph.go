package graph

import (
	"github.com/nfisher/goalgo/graph/adjacency"
	"github.com/nfisher/goalgo/graph/errors"
)

// Directed returns a new directed graph.
func Directed() *adjacency.List {
	return &adjacency.List{}
}

// Graph interface for various forms of graphs.
type Graph interface {
	Edge(v, w int) error
	Adjacent(v int) ([]int, error)
	Vertices() int
	Edges() int
}

// Average returns the average degree of the list.
func Average(g Graph) (float64, error) {
	if g.Vertices() == 0 {
		return -1.0, errors.ErrNoVertices
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
