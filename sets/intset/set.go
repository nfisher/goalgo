package intset

// Set provides integer based set primitives.
type Set map[int]bool

// New creates an integer set optionally with initial values.
func New(initial ...int) Set {
	s := make(Set, len(initial))
	for _, i := range initial {
		s.Add(i)
	}
	return s
}

// Add adds an element to the set.
func (s *Set) Add(i int) {
	(*s)[i] = true
}

// Contains checks if a value is contained in the set.
func (s *Set) Contains(i int) bool {
	return (*s)[i]
}

func (s *Set) Remove(i int) {
	delete((*s), i)
}
