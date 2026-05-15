#include "textflag.h"

// func dotProductUnitary(a, b []float64) float64
TEXT ·dotProductUnitary(SB), NOSPLIT, $0-56
	MOVQ a_base+0(FP), SI
	MOVQ b_base+24(FP), DI
	MOVQ a_len+8(FP), CX

	PXOR X0, X0
	PXOR X1, X1

	MOVQ CX, AX
	ANDQ $-4, AX
	JEQ tail

loop4:
	MOVUPD (SI), X2
	MOVUPD (DI), X3
	MULPD X3, X2
	ADDPD X2, X0

	MOVUPD 16(SI), X4
	MOVUPD 16(DI), X5
	MULPD X5, X4
	ADDPD X4, X1

	ADDQ $32, SI
	ADDQ $32, DI
	SUBQ $4, AX
	JNE loop4

tail:
	ADDPD X1, X0
	MOVAPD X0, X2
	UNPCKHPD X2, X2
	ADDSD X2, X0

	ANDQ $3, CX
	JEQ store

tailLoop:
	MOVSD (SI), X2
	MULSD (DI), X2
	ADDSD X2, X0
	ADDQ $8, SI
	ADDQ $8, DI
	DECQ CX
	JNE tailLoop

store:
	MOVSD X0, ret+48(FP)
	RET
