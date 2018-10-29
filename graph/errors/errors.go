package errors

import "errors"

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
