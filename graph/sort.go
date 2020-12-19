package graph

import (
	"errors"
)

// ErrCyclicGraph is emitted when the graph contains cycles.
var ErrCyclicGraph = errors.New("graph is cyclic")

const (
	unmarked = 0
	temporary = 1
	permanent = 2
)

// TopologicalSort using DFS returns a set of vertices in topologically sorted order.
func TopologicalSort(g Graph) ([]int, error) {
	if g.Vertices() == 0 {
		return []int{}, nil
	}
	if g.Vertices() == 1 {
		return []int{0}, nil
	}

	var sorted []int
	tracker := make(map[int]int8)

	var err error
	for n := 0; n < g.Vertices(); n++ {
		sorted, err = visit(g, n, tracker, sorted)
		if err != nil {
			return nil, err
		}
	}

	return sorted, nil
}

// visit is a recursive visitor for topo sort... I'm sure the stack will *never* blow up amirite?
func visit(g Graph, n int, tracker map[int]int8, sorted []int) ([]int, error) {
	discovered := tracker[n]
	if discovered == permanent {
		return sorted, nil
	}

	if discovered == temporary {
		return nil, ErrCyclicGraph
	}

	tracker[n] = temporary

	adj, err := g.Adjacent(n)
	if err != nil {
		return nil, err
	}

	for _, m := range adj {
		sorted, err = visit(g, m, tracker, sorted)
		if err != nil {
			return nil, err
		}
	}
	tracker[n] = permanent

	return append([]int{n}, sorted...), nil
}
