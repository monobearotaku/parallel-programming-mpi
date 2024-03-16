cmake_minimum_required(VERSION 3.28)
project(mpi)

set(CMAKE_CXX_STANDARD 17)

find_package(MPI REQUIRED)
if (MPI_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${MPI_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MPI_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${MPI_EXE_LINKER_FLAGS}")
endif()

include_directories(${MPI_INCLUDE_PATH})
add_executable(mpi main.cpp
        ../matrix/matrix.cpp
        ../matrix/matrix.h
        operations/operations.cpp
        operations/operations.h
)
target_link_libraries(mpi ${MPI_CXX_LIBRARIES})