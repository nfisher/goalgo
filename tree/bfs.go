package tree

import (
	"fmt"
	"github.com/nfisher/goalgo/queue"
)

// SearchFn is used in tree/graph searches. Returning true will terminate the search.
type SearchFn func(tree *BinaryTree, depth int) bool

type nodeLevel struct {
	tree *BinaryTree
	depth int
}

func BFSPrint(tree *BinaryTree) {
	var level int
	q := queue.New()
	q.Enqueue(&nodeLevel{tree, level})

	for {
		v, err := q.Dequeue()
		if err == queue.ErrNoValues {
			break
		}

		nl, ok := v.(*nodeLevel)
		if !ok {
			continue
		}

		if nl.depth != level {
			fmt.Println("")
			level = nl.depth
		}

		if nl.tree == nil {
			fmt.Printf("[       nil       ]\t")
		} else {
			fmt.Printf("%-18.0d\t", nl.tree.Value)
		}

		if nl.tree != nil {
			q.Enqueue(&nodeLevel{nl.tree.Left, nl.depth + 1})
			q.Enqueue(&nodeLevel{nl.tree.Right, nl.depth + 1})
		}
	}
	fmt.Println("")
}
