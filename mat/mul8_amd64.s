
#include "textflag.h"

TEXT	·AxpyLoop(SB), NOSPLIT,$88
        VBROADCASTSD s+144(SP), Y0 // s
        MOVQ	bc+152(SP), AX // bc
        MOVQ	AX, bcb+72(SP) // bcb local var
        MOVQ	c+96(SP), DI   // c ptr
        MOVQ	b+120(SP), SI  // b ptr
        MOVQ	c+104(SP), DX  // c len
        MOVQ	b+128(SP), CX  // b len
        MOVQ	cib+160(SP), R9 // cib value
        MOVQ	bib+168(SP), R10 // bib value
        MOVQ	bcb+72(SP), AX // bcb
        MOVQ	bc+152(SP), CX // bc
        MOVQ	blockSize+176(SP), DX // blockSize
        RET

// QuadAxpy(c, b []float64, s float64, ci, bi int)
// OSX capabilities - sysctl -a | grep -i avx
// +16 c ptr
// +24 c len?
// +32 c cap?
// +40 b ptr
// +48 s float64
// +56 ci int
// +64 bi int
TEXT	·QuadAxpy(SB), NOSPLIT, $72
        MOVQ	        c+16(SP), DX // c ptr
        MOVQ	        b+40(SP), DI // b ptr
        VBROADCASTSD    s+64(SP), Y0 // s
        MOVQ	        ci+72(SP), AX // ci
        MOVQ	        bi+80(SP), BX // bi

        VMOVAPD         (DI)(BX*8), Y1
        VMOVAPD         (DX)(AX*8), Y2

        VMULPD          Y0, Y1, Y1
        VADDPD          Y1, Y2, Y2
        VMOVAPD         Y2, (DX)(AX*8)

        RET
