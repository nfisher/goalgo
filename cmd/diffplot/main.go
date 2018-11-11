package main

import (
	"encoding/csv"
	"flag"
	"log"
	"os"
	"sort"
	"strconv"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/plotutil"
	"gonum.org/v1/plot/vg"
)

func main() {
	var filename string
	var base string
	var threshold float64
	var height float64

	flag.StringVar(&base, "base", "naive_IJK", "base performance comparison function")
	flag.StringVar(&filename, "in", "results.csv", "input file")
	flag.Float64Var(&threshold, "threshold", 200, "cut-off threshold to ignore algorithms worse than")
	flag.Float64Var(&height, "height", 5.0, "height of the image, (width is calculated as 1.618h)")
	flag.Parse()

	r, err := os.Open(filename)
	if err != nil {
		log.Fatal(err)
	}

	csvr := csv.NewReader(r)
	rows, err := csvr.ReadAll()
	if err != nil {
		log.Fatal(err)
	}

	var groups []string
	var seen = make(map[string]bool)
	valueMap := make(map[string]plotter.Values)

	for _, row := range rows {
		f, g, ts := row[0], row[1], row[2]

		if !seen[g] {
			groups = append(groups, g)
		}

		t, err := strconv.Atoi(ts)
		if err != nil {
			log.Fatal(err)
		}

		v, ok := valueMap[f]
		if !ok {
			v = plotter.Values{}
		}
		valueMap[f] = append(v, float64(t))
		seen[g] = true
	}

	baseValues, ok := valueMap[base]
	if !ok {
		log.Fatalf("base %v not in value map\n", base)
	}

	p, err := plot.New()
	if err != nil {
		log.Println(err)
	}

	p.Title.Text = "Relative performance vs " + base + "\n(-ve is better)"
	p.Y.Label.Text = "% difference vs " + base

	w := vg.Points(8)

	var i int
	var keys []string
	for k := range valueMap {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	for _, k := range keys {
		if k == base {
			continue
		}
		v := valueMap[k]

		skip := false
		for i := range baseValues {
			bv := baseValues[i]
			nv := (v[i] - bv) / bv * 100.0
			v[i] = nv
			if nv > threshold {
				skip = true
			}
		}

		if skip {
			continue
		}

		bars, err := plotter.NewBarChart(v, w)
		if err != nil {
			log.Println(err)
		}
		bars.LineStyle.Width = vg.Length(0)
		bars.Color = plotutil.Color(i)
		bars.Offset = w*vg.Length(i) + vg.Length(i+2)
		p.Add(bars)
		p.Legend.Add(k, bars)
		p.Legend.Top = false
		p.Legend.Left = false
		p.Legend.Padding = vg.Points(1)
		i++
	}
	p.NominalX(groups...)

	log.Printf("%v\n", groups)

	err = p.Save(vg.Length(1.618*height)*vg.Inch, vg.Length(height)*vg.Inch, "results.png")
	if err != nil {
		log.Println(err)
	}
}
