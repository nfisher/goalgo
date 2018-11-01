package vclock

// VClock is a collection of logical clocks that combine to represent a vector clock.
type VClock []int

// Merge receives a collection of vector clocks and merges them into the least upper bound for each index.
func Merge(vclocks []VClock) VClock {
	cnt := len(vclocks)
	if cnt < 1 {
		return nil
	} else if cnt == 1 {
		return vclocks[0]
	}

	var merged = vclocks[0]

	for i := 1; i < cnt; i++ {
		cur := vclocks[i]

		for j, v := range cur {
			if j >= len(merged) {
				merged = append(merged, v)
			}

			if v > merged[j] {
				merged[j] = v
			}
		}
	}

	return merged
}
