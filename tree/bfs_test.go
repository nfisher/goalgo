package tree_test

import (
	"fmt"
	"github.com/nfisher/goalgo/tree"
	"testing"
)

func Test_bfs(t *testing.T) {
	root, scope := gen()

	fmt.Println(len(scope.values))
	tree.BFSPrint(root)
}
