package adjacency_test

import (
	"bytes"
	"encoding/json"
	"reflect"
	"testing"

	"github.com/nfisher/goalgo/graph"
	"github.com/nfisher/goalgo/graph/adjacency"
	"github.com/nfisher/goalgo/graph/errors"
)

func Test_average_degree(t *testing.T) {
	td := []struct {
		name    string
		list    *adjacency.List
		average float64
		err     error
	}{
		{"no vertices", newList(), -1.0, errors.ErrNoVertices},
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
		list *adjacency.List
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
		list   *adjacency.List
		vertex int
		degree int
		err    error
	}{
		{"no connections", newList(WithEdges()), 0, 0, nil},
		{"with outbound connections", newList(WithEdges(), WithEdges(), WithEdges(), AddEdge(1, 0), AddEdge(1, 2)), 1, 2, nil},
		{"out of range", newList(), 0, -1, errors.ErrVertexNotFound},
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
		list     *adjacency.List
		vertex   int
		expected []int
		err      error
	}{
		{"no vertices", newList(WithEdges(), WithEdges()), 1, nil, nil},
		{"return adjacent vertices", newList(WithEdges(), WithEdges(0)), 1, []int{0}, nil},
		{"error on invalid vertex", newList(), 0, nil, errors.ErrVertexNotFound},
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
		list *adjacency.List
		v    int
		w    int
		len  int
		err  error
	}{
		{"add edge with valid vertices", newList(WithEdges(), WithEdges()), 0, 1, 1, nil},
		{"add edge with valid vertices", newList(WithEdges(), WithEdges()), 0, 1, 1, nil},
		{"rejects edge with invalid vertices", newList(), 0, 1, 0, errors.ErrCannotAddEdge},
		{"rejects edge with invalid v vertices", newList(WithEdges()), 1, 0, 0, errors.ErrCannotAddEdge},
		{"rejects edge with invalid w vertice", newList(WithEdges()), 0, 1, 0, errors.ErrCannotAddEdge},
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

func Test_vertices(t *testing.T) {
	td := []struct {
		name  string
		list  *adjacency.List
		edges []int
		id    int
		err   error
	}{
		{"adds vertex to empty list", newList(), nil, 0, nil},
		{"adds vertex to populated list", newList(WithEdges()), nil, 1, nil},
		{"adds vertex with valid edge", newList(WithEdges()), []int{0}, 1, nil},
		{"rejects vertex with invalid edge", newList(), []int{1}, -1, errors.ErrCannotAddVertices},
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

func Test_EncodeJSON(t *testing.T) {
	var buf bytes.Buffer
	l := newList(WithEdges(), WithEdges(0), WithEdges(1), WithEdges(2), AddEdge(1, 3))
	enc := json.NewEncoder(&buf)
	err := enc.Encode(&l)
	if err != nil {
		t.Errorf("Encode() = %v, want nil", err)
	}

	expected := "[[],[0,3],[1],[2]]\n"
	if buf.String() != expected {
		t.Errorf("String() = %v, want %v", buf.String(), expected)
	}
}

func Test_DecodeJSON(t *testing.T) {
	input := "[[],[0,3],[1],[2]]\n"
	buf := bytes.NewBufferString(input)
	var as adjacency.List

	dec := json.NewDecoder(buf)
	err := dec.Decode(&as)
	if err != nil {
		t.Errorf("FromJSON() err = %v, want nil", err)
	}

	if as.Vertices() != 4 {
		t.Errorf("Vertices() = %v, want 4", as.Vertices())
	}

	if as.Edges() != 4 {
		t.Errorf("Edges() = %v, want 4", as.Edges())
	}
}

type Modifier func(*adjacency.List)

func WithEdges(i ...int) Modifier {
	return func(as *adjacency.List) {
		_, err := as.Vertex(i...)
		if err != nil {
			panic(err)
		}
	}
}

func AddEdge(v, w int) Modifier {
	return func(as *adjacency.List) {
		err := as.Edge(v, w)
		if err != nil {
			panic(err)
		}
	}
}

func newList(mm ...Modifier) *adjacency.List {
	s := &adjacency.List{}
	for _, m := range mm {
		m(s)
	}
	return s
}
