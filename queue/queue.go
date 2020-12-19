package queue

import "errors"

// Enqueue adds a value to the back of the queue.
func (q *Queue) Enqueue(v interface{}) {
	newTail := &Link{
		Value: v,
	}

	q.depth++

	if q.head == nil {
		q.head = newTail
		q.tail = newTail
		return
	}

	oldTail := q.tail
	oldTail.Next = newTail
	q.tail = newTail
}

// Dequeue returns the value from the front of the queue, ErrNoValues if none present.
func (q *Queue) Dequeue() (interface{}, error) {
	l := q.head
	if l == nil {
		return 0, ErrNoValues
	}

	q.depth--
	q.head = l.Next

	return l.Value, nil
}

// Len returns the current depth of the queue.
func (q *Queue) Len() int {
	return q.depth
}

// New creates a new empty queue.
func New() *Queue {
	return &Queue{}
}

// Queue is a singly linked list based queue.
type Queue struct {
	depth int
	head  *Link
	tail  *Link
}

// Link is a single entry in the Queue.
type Link struct {
	Next  *Link
	Value interface{}
}

// ErrNoValues is an error that is emitted when a queue has no values to dequeue.
var ErrNoValues = errors.New("queue has no values enqueued")
