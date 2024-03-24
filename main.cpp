#include <mpi.h>

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>

int rank, commsize, lb, ub, nrows, n;

void get_chunk(int *l, int *u) {
  int rows_per_process = n / commsize;
  int remaining_rows = n % commsize;

  if (rank < remaining_rows) {
    *l = rank * (rows_per_process + 1);
    *u = *l + rows_per_process;
  } else {
    *l = remaining_rows * (rows_per_process + 1) + (rank - remaining_rows) * rows_per_process;
    *u = *l + rows_per_process - 1;
  }
}

int get_proc(int idx) {
  int rows_per_process = n / commsize;
  int remaining_rows = n % commsize;
  int threshold = remaining_rows * (rows_per_process + 1);
  if (idx < threshold) {
    return idx / (rows_per_process + 1);
  } else {
    return remaining_rows + (idx - threshold) / rows_per_process;
  }
}

double *get_input_matrix() {
  double *matrix = (double *) malloc(nrows * n * sizeof(double));

  for (int i = 0; i < nrows; i++) {
    for (int j = 0; j < n; j++) {
      matrix[i * n + j] = std::min(n - j, n - i - rank * nrows);
    }
  }

  return matrix;
}

double *get_connected_matrix() {
  double *x = (double *) malloc(nrows * n * sizeof(double));
  for (int i = 0; i < nrows; i++) {
    for (int j = 0; j < n; j++) {
      if (i + rank * nrows == j) {
        x[i * n + j] = 1.0;
      } else {
        x[i * n + j] = 0.0;
      }
    }
  }
  return x;
}

void inverse_matrix(double *a, double *x, int cur_col) {
  double local_max = 0.0;
  int local_index = -1;
  for (int i = 0; i < nrows; i++) {
    int global_index = i + lb;
    if (global_index >= cur_col) {
      double value = std::fabs(a[i * n + cur_col]);

      if (value > local_max) {
        local_max = value;
        local_index = global_index;
      }
    }
  }

  struct {
    double value;
    int index;
  } local_data = {local_max, local_index}, global_data;

  MPI_Allreduce(&local_data, &global_data, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);

  int diag_p = get_proc(cur_col);
  int main_p = get_proc(global_data.index);

  double *s1 = (double *) malloc(n * 4 * sizeof(double));
  for (int i = 0; i < n * 4; i++) {
    s1[i] = 0.0;
  }

  if (rank == diag_p) {
    for (int i = n; i < n * 2; ++i) {
      s1[i] = a[(cur_col - rank * nrows) * n + i - n];
      s1[i + n * 2] = x[(cur_col - rank * nrows) * n + i - n];
    }
  }

  if (rank == main_p) {
    for (int i = 0; i < n; ++i) {
      s1[i] = a[(global_data.index - rank * nrows) * n + i];
      s1[i + n * 2] = x[(global_data.index - rank * nrows) * n + i];
    }
  }

  double *s2 = (double *) malloc(n * 4 * sizeof(double));
  MPI_Allreduce(s1, s2, n * 4, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  double c = s2[cur_col];
  for (int i = 0; i < n; i++) {
    s2[i] = s2[i] / c;
    s2[i + n * 2] = s2[i + n * 2] / c;
  }

  if (rank == main_p) {
    for (int i = n; i < n * 2; i++) {
      a[(global_data.index - rank * nrows) * n + i - n] = s2[i];
      x[(global_data.index - rank * nrows) * n + i - n] = s2[i + n * 2];
    }
  }

  if (rank == diag_p) {
    for (int i = 0; i < n; i++) {
      a[(cur_col - rank * nrows) * n + i] = s2[i];
      x[(cur_col - rank * nrows) * n + i] = s2[i + n * 2];
    }
  }

  for (int i = 0; i < nrows; i++) {
    if (i + rank * nrows != cur_col) {
      c = a[i * n + cur_col];
      for (int j = 0; j < n; j++) {
        a[i * n + j] = a[i * n + j] - s2[j] * c;
        x[i * n + j] = x[i * n + j] - s2[j + n * 2] * c;
      }
    }
  }

  free(s1);
  free(s2);
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &commsize);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    printf("Enter matrix size: ");
    scanf("%d", &n);
  }

  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);

  get_chunk(&lb, &ub);
  nrows = ub - lb + 1;

  double *a = get_input_matrix();
  double *x = get_connected_matrix();

  double t = -MPI_Wtime();

  for (int i = 0; i < n; ++i) {
    inverse_matrix(a, x, i);
  }
  t += MPI_Wtime();

  double *recvbuf = nullptr;
  int *recvcounts = (int *) malloc(commsize * sizeof(int));
  int *displs = (int *) malloc(commsize * sizeof(int));

  if (rank == 0) {
    recvbuf = (double *) malloc(n * n * sizeof(double));
    for (int i = 0; i < commsize; ++i) {
      int l, u;
      get_chunk(&l, &u);
      recvcounts[i] = (u - l + 1) * n;
      displs[i] = l * n;
    }
  }

  MPI_Gatherv(a, nrows * n, MPI_DOUBLE, recvbuf, recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    std::cout << commsize << " procs, n = " << n << ", t = " << t << " sec\n";
  }

  free(a);
  free(x);
  free(recvcounts);
  free(displs);
  if (rank == 0) {
    free(recvbuf);
  }

  MPI_Finalize();
  return 0;
}
