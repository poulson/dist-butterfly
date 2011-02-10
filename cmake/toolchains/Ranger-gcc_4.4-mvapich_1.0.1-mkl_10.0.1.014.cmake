# We need MPI C and CXX compilers
set(CMAKE_C_COMPILER /opt/apps/gcc4_4/mvapich/1.0.1/bin/mpicc)
set(CMAKE_CXX_COMPILER /opt/apps/gcc4_4/mvapich/1.0.1/bin/mpicxx)

set(CXX_FLAGS "-O3")

set(MATH_LIBS 
    "-L/opt/apps/intel/mkl/10.0.1.014/lib/em64t -lmkl_em64t -lmkl -lguide -lpthread /opt/apps/gcc_amd/4.4.5/lib64/libgfortran.a -lm")

