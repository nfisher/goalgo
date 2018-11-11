
#include "textflag.h"

TEXT ·Abs(SB),NOSPLIT,$0
    MOVQ   $(1<<63), BX
    MOVQ   BX, X0 // movsd $(-0.0), x0
    MOVSD  x+0(FP), X1
    ANDNPD X1, X0
    MOVSD  X0, ret+8(FP)
    RET

/*
TEXT ·Mul8(SB),NOSPLIT,$0
    MOVQ   $(1<<63), BX
    MOVQ   BX, X0 // movsd $(-0.0), x0
    MOVSD  x+0(FP), X1
    ANDNPD X1, X0
    MOVSD  X0, ret+8(FP)
    RET
    */