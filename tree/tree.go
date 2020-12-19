package tree

import "errors"

// BinaryTree is the base structure for creating a binary tree.
type BinaryTree struct {
	Value  int
	Left   *BinaryTree
	Right  *BinaryTree
	Parent *BinaryTree
}

// Insert adds a value to the tree.
func Insert(t *BinaryTree, v int) *BinaryTree {
	if t == nil {
		return &BinaryTree{Value: v}
	}

	if t.Value > v {
		t.Left = Insert(t.Left, v)
		t.Left.Parent = t
	} else if t.Value < v {
		t.Right = Insert(t.Right, v)
		t.Right.Parent = t
	}

	return t
}

// Max returns the maximum value in the tree.
func Max(t *BinaryTree) (*BinaryTree, error) {
	if t == nil {
		return nil, ErrNilTree
	}

	if t.Right != nil {
		return Max(t.Right)
	}
	return t, nil
}

// Min returns the minimum value in the tree.
func Min(t *BinaryTree) (*BinaryTree, error) {
	if t == nil {
		return nil, ErrNilTree
	}

	if t.Left != nil {
		return Min(t.Left)
	}
	return t, nil
}

// Search finds the value in the tree.
func Search(t *BinaryTree, v int) (*BinaryTree, error) {
	if t == nil {
		return nil, ErrNotFound
	}

	if v > t.Value {
		return Search(t.Right, v)
	}

	if v < t.Value {
		return Search(t.Left, v)
	}

	// v == t.Value implied
	return t, nil
}

// Delete removes the value from the tree.
func Delete(t *BinaryTree, v int) (*BinaryTree, error) {
	var newChild *BinaryTree

	del, err := Search(t, v)
	if err != nil {
		return nil, err
	}

	delParent := del.Parent
	delLeft := del.Left
	delRight := del.Right

	if delLeft == nil && delRight == nil {
		newChild = nil
	} else if delLeft != nil && delRight != nil {
		min, err := Min(delRight)
		if err != nil {
			return nil, err
		}
		minParent := min.Parent
		minRight := min.Right
		minParent.Left = minRight

		if minRight != nil {
			minRight.Parent = minParent
		}

		if min == delRight {
			delRight = nil
		}

		newChild = min
	} else if delLeft != nil {
		newChild = delLeft
		delLeft = nil
	} else if delRight != nil {
		newChild = delRight
		delRight = nil
	}

	if newChild != nil {
		newChild.Parent = delParent
		newChild.Left = delLeft
		newChild.Right = delRight
	}

	if v > delParent.Value {
		delParent.Right = newChild
	}

	if v < delParent.Value {
		delParent.Left = newChild
	}

	del.Parent = nil
	del.Left = nil
	del.Right = nil

	return del, nil
}

var (
	// ErrNilTree is returned when a provided tree argument is nil.
	ErrNilTree  = errors.New("nil tree invalid")
	// ErrNotFound is returned when a value is not present in the tree.
	ErrNotFound = errors.New("value not found in tree")
)
