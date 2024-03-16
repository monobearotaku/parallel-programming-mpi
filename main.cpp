#include <chrono>
#include <ctime>

#include "matrix/matrix.h"
#include "operations/operations.h"
#include <mpi.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    srand(time(nullptr) + world_rank);
    int n = 0;

    if (world_rank == 0) {
        std::cout << "Please provide size of matrix: ";
        std::cin >> n;
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    Matrix a(n), b(n), m(n);

    if (world_rank == 0) {
        a.SetRandom();
        b.SetIdentity();
        m = Matrix(a);
    }

    auto buf = a.Flatten();
    MPI_Bcast(buf.data(), n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    a.Unflatten(buf);

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    auto m_raw = m.ToRawDoubleArray();
    auto b_raw = b.ToRawDoubleArray();

    inverse_parallel(m_raw, b_raw, n, world_rank, world_size);

    MPI_Barrier(MPI_COMM_WORLD);
    auto stop = std::chrono::high_resolution_clock::now();

    m = FromRawDoubleArray(m_raw, n);
    b = FromRawDoubleArray(b_raw, n);

    if (world_rank == 0) {
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
        std::cout << "Matrix inversion took " << duration << " milliseconds." << std::endl;
        std::cout<<a;
        std::cout<<b;
        std::cout<<production(a, b);
    }

    MPI_Finalize();
}