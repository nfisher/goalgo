package vclock

import "errors"

type VClock []int

func Merge(vclocks []VClock) (VClock, error) {
	cnt := len(vclocks)
	if cnt < 1 {
		return nil, nil
	} else if cnt == 1 {
		return vclocks[0], nil
	}
	var merged = vclocks[0]
	sz := len(merged)

	for i := 1; i < cnt; i++ {
		cur := vclocks[i]
		if len(cur) != sz {
			return nil, ErrVClockSizeMismatch
		}

		for i, v := range cur {
			if v > merged[i] {
				merged[i] = v
			}
		}
	}

	return merged, nil
}

var (
	ErrVClockSizeMismatch = errors.New("vclock: size mismatch")
)
