package graph_test

import (
	"reflect"
	"testing"

	"github.com/nfisher/goalgo/graph"
)

func Test_average_degree(t *testing.T) {
	td := []struct {
		name    string
		list    *graph.AdjacencyList
		average float64
		err     error
	}{
		{"no vertices", newList(), -1.0, graph.ErrNoVertices},
		{"with connections", newList(WithEdges(), WithEdges(), WithEdges(), AddEdge(1, 0), AddEdge(1, 2)), 2.0 / 3.0 * 2.0, nil},
	}

	for _, tc := range td {
		t.Run(tc.name, func(t *testing.T) {
			actual, err := graph.Average(tc.list)
			if actual != tc.average {
				t.Errorf("Average() = %v, want %v", actual, tc.average)
			}
			if err != tc.err {
				t.Errorf("Average() err = %v, want %v", err, tc.err)

			}
		})
	}
}

func Test_max_degree(t *testing.T) {
	td := []struct {
		name string
		list *graph.AdjacencyList
		max  int
	}{
		{"no connections", newList(WithEdges()), 0},
		{"with connections", newList(WithEdges(), WithEdges(), WithEdges(), AddEdge(1, 0), AddEdge(1, 2)), 2},
	}

	for _, tc := range td {
		t.Run(tc.name, func(t *testing.T) {
			actual := graph.Max(tc.list)
			if actual != tc.max {
				t.Errorf("Max() = %v, want %v", actual, tc.max)
			}
		})
	}
}

func Test_degree(t *testing.T) {
	td := []struct {
		name   string
		list   *graph.AdjacencyList
		vertex int
		degree int
		err    error
	}{
		{"no connections", newList(WithEdges()), 0, 0, nil},
		{"with outbound connections", newList(WithEdges(), WithEdges(), WithEdges(), AddEdge(1, 0), AddEdge(1, 2)), 1, 2, nil},
		{"out of range", newList(), 0, -1, graph.ErrVertexNotFound},
	}

	for _, tc := range td {
		t.Run(tc.name, func(t *testing.T) {
			actual, err := graph.Degree(tc.list, tc.vertex)
			if actual != tc.degree {
				t.Errorf("Degree(%v) = %v, want %v", tc.vertex, actual, tc.degree)
			}
			if err != tc.err {
				t.Errorf("Degree(%v) err = %v, want %v", tc.vertex, err, tc.err)
			}
		})
	}
}

func Test_adjacent(t *testing.T) {
	td := []struct {
		name     string
		list     *graph.AdjacencyList
		vertex   int
		expected []int
		err      error
	}{
		{"no vertices", newList(WithEdges(), WithEdges()), 1, nil, nil},
		{"return adjacent vertices", newList(WithEdges(), WithEdges(0)), 1, []int{0}, nil},
		{"error on invalid vertex", newList(), 0, nil, graph.ErrVertexNotFound},
	}

	for _, tc := range td {
		t.Run(tc.name, func(t *testing.T) {
			actual, err := tc.list.Adjacent(tc.vertex)
			if !reflect.DeepEqual(actual, tc.expected) {
				t.Errorf("list.Adjacent(%v) = %v, want %v", tc.vertex, actual, tc.expected)
			}
			if err != tc.err {
				t.Errorf("list.Adjacent(%v) err = %v, want %v", tc.vertex, err, tc.err)
			}
		})
	}
}

func Test_edge(t *testing.T) {
	td := []struct {
		name string
		list *graph.AdjacencyList
		v    int
		w    int
		len  int
		err  error
	}{
		{"add edge with valid vertices", newList(WithEdges(), WithEdges()), 0, 1, 1, nil},
		{"add edge with valid vertices", newList(WithEdges(), WithEdges()), 0, 1, 1, nil},
		{"rejects edge with invalid vertices", newList(), 0, 1, 0, graph.ErrCannotAddEdge},
		{"rejects edge with invalid v vertices", newList(WithEdges()), 1, 0, 0, graph.ErrCannotAddEdge},
		{"rejects edge with invalid w vertice", newList(WithEdges()), 0, 1, 0, graph.ErrCannotAddEdge},
	}

	for _, tc := range td {
		t.Run(tc.name, func(t *testing.T) {
			err := tc.list.Edge(tc.v, tc.w)
			if tc.err != err {
				t.Errorf("list.Edge(%v, %v) = %v, want %v", tc.v, tc.w, err, tc.err)
			}

			if tc.len != tc.list.Edges() {
				t.Errorf("list.Edges() = %v, want %v", tc.list.Edges(), tc.len)
			}
		})
	}
}

func Test_vertice(t *testing.T) {
	td := []struct {
		name  string
		list  *graph.AdjacencyList
		edges []int
		id    int
		err   error
	}{
		{"adds vertice to empty list", newList(), nil, 0, nil},
		{"adds vertice to populated list", newList(WithEdges()), nil, 1, nil},
		{"adds vertice with valid edge", newList(WithEdges()), []int{0}, 1, nil},
		{"rejects vertice with invalid edge", newList(), []int{1}, -1, graph.ErrCannotAddVertices},
	}

	for _, tc := range td {
		t.Run(tc.name, func(t *testing.T) {
			id, err := tc.list.Vertex(tc.edges...)
			if id != tc.id {
				t.Errorf("list.Add(%v) id = %v, want %v", tc.edges, id, tc.id)
			}
			if err != tc.err {
				t.Errorf("list.Add(%v) id = %v, want %v", tc.edges, err, tc.err)
			}
		})
	}
}

func Test_counters(t *testing.T) {
	td := []struct {
		name     string
		actual   int
		expected int
	}{
		{"Vertices() for empty list", newList().Vertices(), 0},
		{"Vertices() for populated list", newList(WithEdges(), WithEdges(0)).Vertices(), 2},
		{"Edges() for empty list", newList().Edges(), 0},
		{"Edges() for populated list", newList(WithEdges()).Edges(), 0},
		{"Edges() for populated list with edge", newList(WithEdges(), WithEdges(0), WithEdges(0, 1)).Edges(), 3},
	}

	for _, tc := range td {
		t.Run(tc.name, func(t *testing.T) {
			if tc.actual != tc.expected {
				t.Errorf("len = %v, want %v", tc.actual, tc.expected)
			}
		})
	}
}

type Modifier func(*graph.AdjacencyList)

func WithEdges(i ...int) Modifier {
	return func(as *graph.AdjacencyList) {
		_, err := as.Vertex(i...)
		if err != nil {
			panic(err)
		}
	}
}

func AddEdge(v, w int) Modifier {
	return func(as *graph.AdjacencyList) {
		err := as.Edge(v, w)
		if err != nil {
			panic(err)
		}
	}
}

func newList(mm ...Modifier) *graph.AdjacencyList {
	s := &graph.AdjacencyList{}
	for _, m := range mm {
		m(s)
	}
	return s
}
