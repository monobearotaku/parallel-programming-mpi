//
// Created by deck on 2/25/24.
//

#include "operations.h"

Matrix inverse(const Matrix &m) {
    int n = m.Size();

    Matrix I = Matrix(m.Size());
    Matrix A = Matrix(m);

    I.SetIdentity();

    for (int i = 0; i < n; i++) {
        double pivot = A[i][i];
        for (int j = 0; j < n; j++) {
            A[i][j] /= pivot;
            I[i][j] /= pivot;
        }

        for (int row = 0; row < n; row++) {
            if (row != i) {
                double factor = A[row][i];
                for (int col = 0; col < n; col++) {
                    A[row][col] -= factor * A[i][col];
                    I[row][col] -= factor * I[i][col];
                }
            }
        }
    }

    return I;
}

void inverse_parallel(double **A, double **I, int size, int world_rank, int world_size) {
    int n = size;
    double* bufA = new double[n]; // Temporary buffer for A's row
    double* bufI = new double[n]; // Temporary buffer for I's row

    for (int i = 0; i < n; i++) {
        double pivot = A[i][i];

        std::cout<<i<<": "<<pivot<<" ";

        int root = i % world_size;
        if (world_rank == root) {
            for (int j = 0; j < n; j++) {
                bufA[j] = A[i][j] / pivot;
                bufI[j] = I[i][j] / pivot;
            }
        }

        MPI_Bcast(bufA, n, MPI_DOUBLE, root, MPI_COMM_WORLD);
        MPI_Bcast(bufI, n, MPI_DOUBLE, root, MPI_COMM_WORLD);

        // All processes update their matrices with the received row
        for (int j = 0; j < n; j++) {
            A[i][j] = bufA[j];
            I[i][j] = bufI[j];
        }

        MPI_Barrier(MPI_COMM_WORLD);

        for (int row = 0; row < n; row++) {
            if (row != i) {
                double factor = A[row][i];
                for (int col = 0; col < n; col++) {
                    A[row][col] -= factor * A[i][col];
                    I[row][col] -= factor * I[i][col];
                }
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }
}

Matrix production(const Matrix& A, const Matrix& B) {
    Matrix C = Matrix(A.Size());
    C.SetEmpty();

    for (int i = 0; i < A.Size(); i++) {
        for (int j = 0; j < B.Size(); j++) {
            for (int k = 0; k < C.Size(); k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    return C;
}
