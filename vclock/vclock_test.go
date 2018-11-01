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
	}{
		{"nil merge", nil, nil},
		{"single vclock", []vclock.VClock{{0, 1}}, vclock.VClock{0, 1}},
		{"merge different sized vclocks", []vclock.VClock{{2, 0}, {0, 1, 0}, {0, 0, 1}, {1, 0, 0, 0}}, vclock.VClock{2, 1, 1, 0}},
		{"one way merge multiple vclocks", []vclock.VClock{{0, 0, 0}, {0, 0, 1}}, vclock.VClock{0, 0, 1}},
		{"two way merge", []vclock.VClock{{1, 0, 0}, {0, 0, 1}}, vclock.VClock{1, 0, 1}},
		{"two way, same vclocks", []vclock.VClock{{1, 0, 1}, {1, 0, 1}}, vclock.VClock{1, 0, 1}},
		{"three way merge", []vclock.VClock{{1, 0, 0}, {0, 0, 1}, {0, 1, 0}}, vclock.VClock{1, 1, 1}},
	}

	for _, tc := range td {
		t.Run(tc.name, func(t *testing.T) {
			actual := vclock.Merge(tc.input)
			if !reflect.DeepEqual(actual, tc.output) {
				t.Errorf("Merge(%v) = %v, want %v", tc.input, actual, tc.output)
			}
		})
	}
}
