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
		list    graph.Graph
		average float64
		err     error
	}{
		{"no vertices", graph.New(), -1.0, errors.ErrNoVertices},
		{"with connections", graph.New( graph.Vertices(3), graph.Upward(map[int][]int{ 1: {0, 2} })), 2.0 / 3.0 * 2.0, nil},
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
		list graph.Graph
		max  int
	}{
		{"no connections", graph.New(graph.Vertices(1)), 0},
		{"with connections", graph.New(graph.Vertices(3), graph.Upward(map[int][]int{1: {0, 2}})), 2},
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
		list   graph.Graph
		vertex int
		degree int
		err    error
	}{
		{"no connections", graph.New(graph.Vertices(1)), 0, 0, nil},
		{"with outbound connections", graph.New(graph.Vertices(3), graph.Upward(map[int][]int{1: {0, 2}})), 1, 2, nil},
		{"out of range", graph.New(), 0, -1, errors.ErrVertexNotFound},
	}

	for _, tc := range td {
		t.Run(tc.name, func(t *testing.T) {
			actual, err := graph.OutDegree(tc.list, tc.vertex)
			if actual != tc.degree {
				t.Errorf("OutDegree(%v) = %v, want %v", tc.vertex, actual, tc.degree)
			}
			if err != tc.err {
				t.Errorf("OutDegree(%v) err = %v, want %v", tc.vertex, err, tc.err)
			}
		})
	}
}

func Test_adjacent(t *testing.T) {
	td := []struct {
		name     string
		list     graph.Graph
		vertex   int
		expected []int
		err      error
	}{
		{"no adjacent vertices", graph.New(graph.Vertices(2)), 1, nil, nil},
		{"return adjacent vertices", graph.New(graph.Vertices(2), graph.Upward(map[int][]int{1: {0}})), 1, []int{0}, nil},
		{"error on invalid vertex", graph.New(), 0, nil, errors.ErrVertexNotFound},
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
		list graph.Graph
		v    int
		w    int
		len  int
		err  error
	}{
		{"add edge with valid vertices", graph.New(graph.Vertices(2)), 0, 1, 1, nil},
		{"rejects edge with invalid vertices", graph.New(), 0, 1, 0, errors.ErrCannotAddEdge},
		{"rejects edge with invalid v vertices", graph.New(graph.Vertices(1)), 1, 0, 0, errors.ErrCannotAddEdge},
		{"rejects edge with invalid w vertice", graph.New(graph.Vertices(1)), 0, 1, 0, errors.ErrCannotAddEdge},
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
		list  graph.Graph
		edges []int
		id    int
		err   error
	}{
		{"adds vertex to empty list", graph.New(), nil, 0, nil},
		{"adds vertex to populated list", graph.New(graph.Vertices(1)), nil, 1, nil},
		{"adds vertex with valid edge", graph.New(graph.Vertices(1)), []int{0}, 1, nil},
		{"rejects vertex with invalid edge", graph.New(), []int{1}, -1, errors.ErrCannotAddVertices},
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
		{"Vertices() for empty list", graph.New().Vertices(), 0},
		{"Vertices() for populated list", graph.New(graph.Vertices(2), graph.Upward(map[int][]int{1: {0}})).Vertices(), 2},
		{"Edges() for empty list", graph.New().Edges(), 0},
		{"Edges() for populated list", graph.New(graph.Vertices(1)).Edges(), 0},
		{"Edges() for populated list with edge", graph.New(graph.Vertices(3), graph.Upward(map[int][]int{1: {0}, 2:{0,1}})).Edges(), 3},
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
	l := graph.New(graph.Vertices(4), graph.Upward(map[int][]int{1: {0,3}, 2:{1}, 3:{2}}))

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

