//go:build !darwin || !cgo
// +build !darwin !cgo

package nativeblas

// Linked is a no-op when no native BLAS linker shim is needed.
func Linked() {}
