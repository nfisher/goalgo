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
	Vertex(out ...int) (int, error)
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

// OutDegree returns the number of edges connected from the vertex.
func OutDegree(g Graph, v int) (int, error) {
	adj, err := g.Adjacent(v)
	if err != nil {
		return -1, err
	}

	return len(adj), nil
}

// Modifier is a graph initialisation modifier.
type Modifier func(Graph)

// Vertices initialises n vertices in the graph.
func Vertices(n int) Modifier {
	return func(g Graph) {
		for i := 0; i < n; i++ {
			g.Vertex()
		}
	}
}

// New builds a graph using the optional modifiers.
func New(mm ...Modifier) Graph {
	g := &adjacency.List{}
	for _, m := range mm {
		m(g)
	}
	return g
}

// Upward builds
func Upward(m map[int][]int) Modifier {
	return func(g Graph) {
		for v := range m {
			for _, w := range m[v] {
				g.Edge(v, w)
			}
		}
	}
}

// Downward builds a downward graph in the reverse specified.
func Downward(m map[int][]int) Modifier {
	return func(g Graph) {
		for v := range m {
			for _, w := range m[v] {
				g.Edge(w, v)
			}
		}
	}
}
