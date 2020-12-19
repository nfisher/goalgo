package adjacency

import (
	"encoding/json"

	"github.com/nfisher/goalgo/graph/errors"
)

// List is an adjacency list using an array.
type List struct {
	list  [][]int
	edges int
}

// Vertex adds a new vertex, optionally with the specified edges.
func (as *List) Vertex(edges ...int) (id int, err error) {
	var edgeSet = make([]int, 0, len(edges))
	l := len(as.list)
	for _, edge := range edges {
		if edge >= l {
			return -1, errors.ErrCannotAddVertices
		}
		edgeSet = append(edgeSet, edge)
	}

	as.edges += len(edgeSet)
	as.list = append(as.list, edgeSet)

	return l, nil
}

// Edge adds an edge from v to w.
func (as *List) Edge(v, w int) error {
	l := len(as.list)
	if v >= l {
		return errors.ErrCannotAddEdge
	}

	if w >= l {
		return errors.ErrCannotAddEdge
	}

	as.list[v] = append(as.list[v], w)
	as.edges++

	return nil
}

// Adjacent returns all vertices adjacent to this vertex.
func (as *List) Adjacent(v int) ([]int, error) {
	if v >= len(as.list) {
		return nil, errors.ErrVertexNotFound
	}

	var a []int
	for k := range as.list[v] {
		a = append(a, k)
	}

	return a, nil
}

// Vertices returns the number of vertices in the list.
func (as *List) Vertices() int {
	return len(as.list)
}

// Edges returns the number edges in the list.
func (as *List) Edges() int {
	return as.edges
}

// UnmarshalJSON populates the adjacency list from JSON.
func (as *List) UnmarshalJSON(b []byte) error {
	err := json.Unmarshal(b, &as.list)
	if err != nil {
		return err
	}

	var edges int
	for _, v := range as.list {
		edges += len(v)
	}
	as.edges = edges

	return nil
}

// MarshalJSON encodes the adjacency list to JSON.
func (as *List) MarshalJSON() ([]byte, error) {
	return json.Marshal(&as.list)
}
