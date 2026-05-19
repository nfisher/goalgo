package mat

import "sync"

type matMulWork struct {
	start int
	end   int
	fn    func(start, end int)
	wg    *sync.WaitGroup
}

type matMulWorkerPool struct {
	jobs chan matMulWork
}

var matMulPools = struct {
	sync.Mutex
	pools map[int]*matMulWorkerPool
}{
	pools: make(map[int]*matMulWorkerPool),
}

func getMatMulWorkerPool(workers int) *matMulWorkerPool {
	if workers < 2 {
		return nil
	}

	matMulPools.Lock()
	defer matMulPools.Unlock()

	pool := matMulPools.pools[workers]
	if pool == nil {
		pool = newMatMulWorkerPool(workers)
		matMulPools.pools[workers] = pool
	}
	return pool
}

func newMatMulWorkerPool(workers int) *matMulWorkerPool {
	pool := &matMulWorkerPool{
		jobs: make(chan matMulWork, workers*2),
	}
	for i := 0; i < workers; i++ {
		go pool.worker()
	}
	return pool
}

func (p *matMulWorkerPool) worker() {
	for work := range p.jobs {
		work.fn(work.start, work.end)
		work.wg.Done()
	}
}

func (p *matMulWorkerPool) parallel(start, end, chunk int, fn func(start, end int)) {
	if p == nil || start >= end {
		return
	}
	if chunk <= 0 {
		fn(start, end)
		return
	}

	var wg sync.WaitGroup
	for jobStart := start; jobStart < end; jobStart += chunk {
		jobEnd := jobStart + chunk
		if jobEnd > end {
			jobEnd = end
		}
		wg.Add(1)
		p.jobs <- matMulWork{
			start: jobStart,
			end:   jobEnd,
			fn:    fn,
			wg:    &wg,
		}
	}
	wg.Wait()
}
