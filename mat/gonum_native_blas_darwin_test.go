//go:build darwin && cgo
// +build darwin,cgo

package mat_test

import (
	_ "github.com/nfisher/goalgo/internal/nativeblas"
	"gonum.org/v1/gonum/blas/blas64"
	"gonum.org/v1/netlib/blas/netlib"
)

func init() {
	blas64.Use(netlib.Implementation{})
}
