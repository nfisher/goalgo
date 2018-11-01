package vclock_test

import (
	"reflect"
	"testing"

	"github.com/nfisher/goalgo/vclock"
)

func TestMerge(t *testing.T) {
	td := []struct {
		name   string
		input  []vclock.VClock
		output vclock.VClock
		err    error
	}{
		{"nil merge", nil, nil, nil},
		{"single vclock", []vclock.VClock{{0, 1}}, vclock.VClock{0, 1}, nil},
		{"cannot merge different sized vclocks", []vclock.VClock{{0, 0}, {0, 0, 0}}, nil, vclock.ErrVClockSizeMismatch},
		{"one way merge multiple vclocks", []vclock.VClock{{0, 0, 0}, {0, 0, 1}}, vclock.VClock{0, 0, 1}, nil},
		{"two way merge", []vclock.VClock{{1, 0, 0}, {0, 0, 1}}, vclock.VClock{1, 0, 1}, nil},
		{"two way, same vclocks", []vclock.VClock{{1, 0, 1}, {1, 0, 1}}, vclock.VClock{1, 0, 1}, nil},
		{"three way merge", []vclock.VClock{{1, 0, 0}, {0, 0, 1}, {0, 1, 0}}, vclock.VClock{1, 1, 1}, nil},
	}

	for _, tc := range td {
		t.Run(tc.name, func(t *testing.T) {
			actual, err := vclock.Merge(tc.input)
			if err != tc.err {
				t.Errorf("Merge(%v) err = %v, want %v", tc.input, err, tc.err)
			}

			if !reflect.DeepEqual(actual, tc.output) {
				t.Errorf("Merge(%v) = %v, want %v", tc.input, actual, tc.output)
			}
		})
	}
}
