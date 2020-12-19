package tree_test

import (
	"fmt"
	"testing"

	"github.com/nfisher/goalgo/tree"
)

func Test_bfs(t *testing.T) {
	t.Skip("print only")
	root, scope := gen()

	fmt.Println(len(scope.values))
	tree.BFSPrint(root)
}
