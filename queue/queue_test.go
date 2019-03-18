package queue_test

import (
	"testing"

	. "github.com/nfisher/goalgo/queue"
)

func Test_enqueue(t *testing.T) {
	q := New()
	q.Enqueue(1)

	if q.Len() != 1 {
		t.Errorf("len = %v, want 1", q.Len())
	}

	q.Enqueue(2)

	if q.Len() != 2 {
		t.Errorf("len = %v, want 2", q.Len())
	}
}

func Test_dequeue(t *testing.T) {
	q := New()

	q.Enqueue(1)

	v, err := q.Dequeue()
	if v != 1 {
		t.Errorf("dequeue = %v, want 1", v)
	}

	if err != nil {
		t.Errorf("dequeue err = %v, want nil", v)
	}

	if q.Len() != 0 {
		t.Errorf("len = %v, want 0", q.Len())
	}

	_, err = q.Dequeue()
	if err != ErrNoValues {
		t.Errorf("dequeue err = %v, want ErrNoValues", err)
	}
}

func Test_multiple_dequeue(t *testing.T) {
	q := New()

	q.Enqueue(1)
	q.Enqueue(2)

	v, err := q.Dequeue()
	if v != 1 {
		t.Errorf("dequeue = %v, want 1", v)
	}

	if err != nil {
		t.Errorf("dequeue err = %v, want nil", err)
	}

	v2, err := q.Dequeue()
	if v2 != 2 {
		t.Errorf("dequeue = %v, want 2", v2)
	}

	if err != nil {
		t.Errorf("dequeue err = %v, want nil", err)
	}

	if q.Len() != 0 {
		t.Errorf("len = %v, want 0", q.Len())
	}
}
