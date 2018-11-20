
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

TEXT	·HexAxpy(SB), NOSPLIT, $72
        MOVQ	        c+16(SP), DX // c ptr
        MOVQ	        b+40(SP), DI // b ptr
        VBROADCASTSD    s+64(SP), Y0 // s
        MOVQ	        ci+72(SP), AX // ci
        MOVQ	        bi+80(SP), BX // bi

        VMOVAPD         (DI)(BX*8), Y1
        VMOVAPD         32(DI)(BX*8), Y3
        VMOVAPD         (DX)(AX*8), Y2
        VMOVAPD         32(DX)(AX*8), Y4

        VMULPD          Y0, Y1, Y1
        VMULPD          Y0, Y3, Y3

        VADDPD          Y1, Y2, Y2
        VADDPD          Y3, Y4, Y4

        VMOVAPD         Y2, (DX)(AX*8)
        VMOVAPD         Y4, 32(DX)(AX*8)

        RET

TEXT	·HexadecAxpy(SB), NOSPLIT, $72
        MOVQ	        c+16(SP), DX // c ptr
        MOVQ	        b+40(SP), DI // b ptr
        VBROADCASTSD    s+64(SP), Y0 // s
        MOVQ	        ci+72(SP), AX // ci
        MOVQ	        bi+80(SP), BX // bi

        VMOVAPD         (DI)(BX*8), Y1
        VMOVAPD         32(DI)(BX*8), Y3
        VMOVAPD         64(DI)(BX*8), Y5
        VMOVAPD         96(DI)(BX*8), Y7

        VMULPD          Y0, Y1, Y1
        VMULPD          Y0, Y3, Y3
        VMULPD          Y0, Y5, Y5
        VMULPD          Y0, Y7, Y7

        VMOVAPD         (DX)(AX*8), Y2
        VMOVAPD         32(DX)(AX*8), Y4
        VMOVAPD         64(DX)(AX*8), Y6
        VMOVAPD         96(DX)(AX*8), Y8

        VADDPD          Y1, Y2, Y2
        VADDPD          Y3, Y4, Y4
        VADDPD          Y5, Y6, Y6
        VADDPD          Y7, Y8, Y8

        VMOVAPD         Y2, (DX)(AX*8)
        VMOVAPD         Y4, 32(DX)(AX*8)
        VMOVAPD         Y6, 64(DX)(AX*8)
        VMOVAPD         Y8, 96(DX)(AX*8)

        RET

TEXT	·WideAxpy(SB), NOSPLIT, $72
        MOVQ	        c+16(SP), DX // c ptr
        MOVQ	        b+40(SP), DI // b ptr
        VBROADCASTSD    s+64(SP), Y0 // s
        MOVQ	        ci+72(SP), AX // ci
        MOVQ	        bi+80(SP), BX // bi

        VMOVAPD         (DI)(BX*8), Y1
        VMOVAPD         32(DI)(BX*8), Y3
        VMOVAPD         64(DI)(BX*8), Y5
        VMOVAPD         96(DI)(BX*8), Y7
        VMOVAPD         128(DI)(BX*8), Y9
        VMOVAPD         160(DI)(BX*8), Y11
        VMOVAPD         192(DI)(BX*8), Y13
        VMOVAPD         224(DI)(BX*8), Y15

        VMULPD          Y0, Y1, Y1
        VMULPD          Y0, Y3, Y3
        VMULPD          Y0, Y5, Y5
        VMULPD          Y0, Y7, Y7
        VMULPD          Y0, Y9, Y9
        VMULPD          Y0, Y11, Y11
        VMULPD          Y0, Y13, Y13
        VMULPD          Y0, Y15, Y15

        VMOVAPD         (DX)(AX*8), Y0
        VMOVAPD         32(DX)(AX*8), Y2
        VMOVAPD         64(DX)(AX*8), Y4
        VMOVAPD         96(DX)(AX*8), Y6
        VMOVAPD         128(DX)(AX*8), Y8
        VMOVAPD         160(DX)(AX*8), Y10
        VMOVAPD         192(DX)(AX*8), Y12
        VMOVAPD         224(DX)(AX*8), Y14

        VADDPD          Y1, Y0, Y0
        VADDPD          Y3, Y2, Y2
        VADDPD          Y5, Y4, Y4
        VADDPD          Y7, Y6, Y6
        VADDPD          Y9, Y8, Y8
        VADDPD          Y11, Y10, Y10
        VADDPD          Y13, Y12, Y12
        VADDPD          Y15, Y14, Y14

        VMOVAPD         Y0, (DX)(AX*8)
        VMOVAPD         Y2, 32(DX)(AX*8)
        VMOVAPD         Y4, 64(DX)(AX*8)
        VMOVAPD         Y6, 96(DX)(AX*8)
        VMOVAPD         Y8, 128(DX)(AX*8)
        VMOVAPD         Y10, 160(DX)(AX*8)
        VMOVAPD         Y12, 192(DX)(AX*8)
        VMOVAPD         Y14, 224(DX)(AX*8)

        RET
