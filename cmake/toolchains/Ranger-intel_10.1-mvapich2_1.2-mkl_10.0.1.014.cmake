# Set the serial Intel compilers
set(CMAKE_C_COMPILER       /opt/apps/intel10_1/cc/bin/icc)
set(CMAKE_CXX_COMPILER     /opt/apps/intel10_1/cc/bin/icpc)
set(CMAKE_Fortran_COMPILER /opt/apps/intel10_1/fc/bin/ifort)

# Set the MPI wrappers for the C and C++ compilers
set(MPI_C_COMPILER   /opt/apps/intel10_1/mvapich2/1.2/bin/mpicc)
set(MPI_CXX_COMPILER /opt/apps/intel10_1/mvapich2/1.2/bin/mpicxx)

set(CXX_FLAGS "-O3")

set(MATH_LIBS 
    "-L/opt/apps/intel/mkl/10.0.1.014/lib/em64t -lmkl_em64t -lmkl -lguide -lpthread /opt/apps/intel/10.1/fc/lib/libifcore.a /opt/apps/intel/10.1/fc/lib/libsvml.a -lm")

set(INTEL_INC "/opt/apps/intel/mkl/10.0.1.014/include")
include_directories(${INTEL_INC})

