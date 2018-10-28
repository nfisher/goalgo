package intset_test

import (
	"testing"

	"github.com/nfisher/goalgo/sets/intset"
)

func Test_add(t *testing.T) {
	td := []struct {
		name  string
		set   intset.Set
		input int
		len   int
	}{
		{"should add first element to empty set", intset.New(), 1, 1},
		{"should add unknown elements", intset.New(1), 3, 2},
		{"should not add known elements", intset.New(1), 1, 1},
	}

	for _, tc := range td {
		t.Run(tc.name, func(t *testing.T) {
			tc.set.Add(tc.input)
			if len(tc.set) != tc.len {
				t.Errorf("t.Len() = %v, want %v", len(tc.set), tc.len)
			}
		})
	}
}

func Test_contains(t *testing.T) {
	td := []struct {
		name     string
		set      intset.Set
		input    int
		contains bool
	}{
		{"should return false when element unknown", intset.New(), 1, false},
		{"should return true when element known", intset.New(1), 1, true},
	}

	for _, tc := range td {
		t.Run(tc.name, func(t *testing.T) {
			if tc.set.Contains(tc.input) != tc.contains {
				t.Errorf("t.Contains(%v) = %v, want %v", tc.input, tc.set.Contains(tc.input), tc.contains)
			}
		})
	}
}

func Test_remove(t *testing.T) {
	td := []struct {
		name  string
		set   intset.Set
		input int
		len   int
	}{
		{"should ignore removal of unknown element", intset.New(1), 2, 1},
		{"should remove known element", intset.New(1), 1, 0},
	}

	for _, tc := range td {
		t.Run(tc.name, func(t *testing.T) {
			tc.set.Remove(tc.input)
			if len(tc.set) != tc.len {
				t.Errorf("t.Len() = %v, want %v", len(tc.set), tc.len)
			}
		})
	}
}
