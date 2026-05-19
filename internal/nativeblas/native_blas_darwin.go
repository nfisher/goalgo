//go:build darwin && cgo
// +build darwin,cgo

package nativeblas

/*
#cgo LDFLAGS: -framework Accelerate
*/
import "C"

// Linked keeps this cgo package non-empty so its linker flags are included.
func Linked() {}
