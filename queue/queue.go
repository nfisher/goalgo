package queue

import "errors"

// Enqueue adds a value to the back of the queue.
func (q *Queue) Enqueue(v int) {
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
func (q *Queue) Dequeue() (int, error) {
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

type Queue struct {
	depth int
	head  *Link
	tail  *Link
}

type Link struct {
	Next  *Link
	Value int
}

var ErrNoValues = errors.New("queue has no values enqueued")

