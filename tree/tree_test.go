package tree_test

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
	"testing"
	"time"

	. "github.com/nfisher/goalgo/tree"
)

type scope struct {
	values []int
	min    int
	max    int
}

func gen() (*BinaryTree, scope) {
	v := rand.Int()
	min := v
	max := v
	s := scope{}
	s.values = append(s.values, v)

	root := Insert(nil, v)

	n := rand.Intn(128)
	for i := 0; i < n; i++ {
		v := rand.Int()
		s.values = append(s.values, v)
		if v < min {
			min = v
		}
		if v > max {
			max = v
		}
		Insert(root, v)
	}

	s.min = min
	s.max = max

	return root, s
}

func init() {
	rand.Seed(time.Now().UTC().UnixNano())
}

func Test_nesting_insert(t *testing.T) {
	root := Insert(Insert(Insert(nil, 4), 5), 1)

	Insert(root, 2)

	if root.Left.Value != 1 {
		t.Errorf("[0].left = %v, want 1", root.Left.Value)
	}

	if root.Left.Parent != root {
		t.Errorf("left parent = %v, want root", root.Left.Parent)
	}

	if root.Right.Value != 5 {
		t.Errorf("[0].right = %v, want 5", root.Right.Value)
	}

	if root.Right.Parent != root {
		t.Errorf("right parent = %v, want root", root.Right.Parent)
	}
}

func Test_basic_insert(t *testing.T) {
	tt := []struct {
		desc   string
		first  int
		second int
		isLeft bool
	}{
		{"should be left", 2, 1, true},
		{"should be right", 1, 2, false},
	}

	for _, tc := range tt {
		t.Run(tc.desc, func(t *testing.T) {
			root := &BinaryTree{Value: tc.first}
			Insert(root, tc.second)

			isLeft := root.Left != nil
			isRight := root.Right != nil

			if isLeft != tc.isLeft {
				t.Errorf("isLeft = %v, want %v", isLeft, tc.isLeft)
				return
			} else if isRight == tc.isLeft {
				t.Errorf("isRight = %v, want %v", isRight, tc.isLeft)
				return
			}

			var actual int
			if isLeft {
				actual = root.Left.Value
			} else if isRight {
				actual = root.Right.Value
			}

			if actual != tc.second {
				t.Errorf("got %v, want %v", actual, tc.second)
			}
		})
	}
}

func Test_min_max(t *testing.T) {
	root, scope := gen()

	max, _ := Max(root)
	if scope.max != max.Value {
		t.Errorf("max = %v, want %v", max, scope.max)
	}

	min, _ := Min(root)
	if scope.min != min.Value {
		t.Errorf("min = %v, want %v", min, scope.min)
	}

	single := &BinaryTree{Value: 7}
	s, err := Min(single)
	if err != nil {
		t.Errorf("min err = %v, want nil", err)
	}

	if s == nil || s.Value != 7 {
		t.Errorf("s = %#v, want 7", s)
	}
}

func Test_search(t *testing.T) {
	root, scope := gen()

	sort.Ints(scope.values)

	for _, v := range scope.values {
		n, err := Search(root, v)
		if err != nil {
			t.Fatalf("err = %v, want nil", err)
		}

		if n == nil {
			t.Fatalf("n = nil, want node{%v}", v)
		}

		if n.Value != v {
			t.Fatalf("n = %v, want %v", n.Value, v)
		}
	}

	absent := scope.values[0]
	if absent > math.MinInt64 {
		absent--
	} else if absent < math.MaxInt64 {
		absent++
	}

	n, err := Search(root, absent)
	if err != ErrNotFound {
		t.Fatalf("err = %v, want ErrNotFound", err)
	}

	if n != nil {
		t.Fatalf("n = %v, want nil", n)
	}
}

func Test_nil_check(t *testing.T) {
	_, err := Max(nil)
	if err != ErrNilTree {
		t.Errorf("max err = %v, want %v", err, ErrNilTree)
	}

	_, err = Min(nil)
	if err != ErrNilTree {
		t.Errorf("min err = %v, want %v", err, ErrNilTree)
	}
}

func PrintTree(tree *BinaryTree, pad string) {
	if tree == nil {
		return
	}
	fmt.Println(tree.Value)
	fmt.Printf(pad + "L -> ")
	PrintTree(tree.Left, pad+" ")
	fmt.Printf(pad + "R -> ")
	PrintTree(tree.Right, pad+" ")
	fmt.Println("")
}

func Test_delete(t *testing.T) {
	nums := []int{2, 1, 7, 4, 8, 3, 6, 5}

	tt := []struct {
		name     string
		value    int
		err      error
		isNil    bool
		contains []int
	}{
		{"absent value", 9, ErrNotFound, true, nums},
		{"leaf node", 3, nil, false, []int{2, 1, 7, 4, 8, 6, 5}},
		{"single child", 6, nil, false, []int{2, 1, 7, 4, 8, 3, 5}},
		{"double child low", 4, nil, false, []int{2, 1, 7, 8, 3, 6, 5}},
		{"double child high", 7, nil, false, []int{2, 1, 4, 8, 3, 6, 5}},
	}

	for _, tc := range tt {
		t.Run(tc.name, func(t *testing.T) {
			var root *BinaryTree

			for _, v := range nums {
				root = Insert(root, v)
			}

			n, err := Delete(root, tc.value)
			if err != tc.err {
				t.Fatalf("err = %v, want %v", err, tc.err)
			}

			if (n != nil) == tc.isNil {
				t.Fatalf("n = %v, want %v", n, tc.isNil)
			}

			if n != nil && n.Parent != nil {
				t.Fatalf("parent = %v, want nil", n.Parent)
			}

			if n != nil && n.Left != nil {
				t.Fatalf("left = %v, want nil", n.Left)
			}

			if n != nil && n.Right != nil {
				t.Fatalf("right = %v, want nil", n.Right)
			}

			del, _ := Search(root, tc.value)
			if del != nil {
				t.Fatalf("found n = %v, want nil", del.Value)
			}

			for _, y := range tc.contains {
				n, _ := Search(root, y)
				if n == nil {
					t.Errorf("%v not found", y)
				}
			}
		})
	}
}
