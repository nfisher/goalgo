package graph_test

import (
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/nfisher/goalgo/graph"
)

func Test_cyclic_graph_should_return_error(t *testing.T) {
	g := graph.Directed()
	v0, _ := g.Vertex()
	v1, _ := g.Vertex()

	g.Edge(v0, v1)
	g.Edge(v1, v0)

	_, err := graph.TopologicalSort(g)
	if err != graph.ErrCyclicGraph {
		t.Errorf("err=nil, want ErrCyclicGraph")
	}
}

func Test_acyclic_graphs_when(t *testing.T) {
	tt := map[string]struct {
		g     graph.Graph
		order []int
	}{
		"empty graph":           {graph.New(), []int{}},
		"single node":           {graph.New(graph.Vertices(1)), []int{0}},
		"three node chain":      {graph.New(graph.Vertices(3), graph.Upward(map[int][]int{1: {0}, 2: {1}})), []int{2, 1, 0}},
		"example graph":         {exampleGraph(), []int{2, 1, 4, 0, 3, 7, 6, 5}},
		"reverse example graph": {reverso(), []int{7, 6, 5, 4, 3, 2, 1, 0}},
	}

	for n, tc := range tt {
		t.Run(n, func(t *testing.T) {
			o, err := graph.TopologicalSort(tc.g)
			if err != nil {
				t.Errorf("err=%v, want nil", err)
			}
			if !cmp.Equal(o, tc.order) {
				t.Errorf("topological order incorrect (+want,-got)\n%v\n%v\n%s", o, tc.order, cmp.Diff(o, tc.order))
			}
		})
	}
}

func exampleGraph() graph.Graph {
	return graph.New(
		graph.Vertices(8),
		graph.Upward(map[int][]int{
			0: {3},
			1: {3, 4},
			2: {4, 7},
			3: {5, 6, 7},
			4: {6},
		}),
	)
}

func reverso() graph.Graph {
	return graph.New(
		graph.Vertices(8),
		graph.Downward(map[int][]int{
			0: {3},
			1: {3, 4},
			2: {4, 7},
			3: {5, 6, 7},
			4: {6},
		}),
	)
}
