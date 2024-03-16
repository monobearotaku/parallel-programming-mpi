//
// Created by deck on 2/25/24.
//

#ifndef OPERATIONS_H
#define OPERATIONS_H

#include <mpi.h>
#include "../matrix/matrix.h"

Matrix inverse(const Matrix& m);
void inverse_parallel(double **A, double **I, int size, int world_rank, int world_size);

Matrix production(const Matrix& A, const Matrix& B);

#endif //OPERATIONS_H
