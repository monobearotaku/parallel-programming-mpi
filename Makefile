.PHONY: one
one:
	mpiexec -n 1 cmake-build-debug/mpi

.PHONY: two
two:
	mpiexec -n 2 cmake-build-debug/mpi


.PHONY: three
three:
	mpiexec -n 3 cmake-build-debug/mpi

.PHONY: four
four:
	mpiexec -n 4 cmake-build-debug/mpi
