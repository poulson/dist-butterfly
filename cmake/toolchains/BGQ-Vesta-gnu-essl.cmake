#set(CMAKE_SYSTEM_NAME BlueGeneQ-static)

# The serial compilers are typically set here, but, at least at some point,
# it was easier to set them to the MPI compilers.
set(MPI_ROOT  "/bgsys/drivers/ppcfloor/comm/gcc")
set(CMAKE_C_COMPILER       ${MPI_ROOT}/bin/mpicc)
set(CMAKE_CXX_COMPILER     ${MPI_ROOT}/bin/mpicxx)

# The MPI wrappers for the C and C++ compilers
set(MPI_C_COMPILER   ${MPI_ROOT}/bin/mpicc)
set(MPI_CXX_COMPILER ${MPI_ROOT}/bin/mpicxx)

set(PAMI_ROOT "/bgsys/drivers/ppcfloor/comm/sys")
set(SPI_ROOT  "/bgsys/drivers/ppcfloor/spi")
set(MPI_C_COMPILE_FLAGS   "")
set(MPI_CXX_COMPILE_FLAGS "")
set(MPI_C_INCLUDE_PATH   "${MPI_ROOT}/include")
set(MPI_CXX_INCLUDE_PATH "${MPI_ROOT}/include")
set(MPI_C_LINK_FLAGS   "-L${MPI_ROOT}/lib -L${PAMI_ROOT}/lib -L${SPI_ROOT}/lib")
set(MPI_CXX_LINK_FLAGS ${MPI_C_LINK_FLAGS})
set(MPI_C_LIBRARIES "-lmpich -lopa -lmpl -ldl -lpami -lSPI -lSPI_cnk -lpthread -lrt -lstdc++")
set(MPI_CXX_LIBRARIES "-lcxxmpich ${MPI_C_LIBRARIES}")

set(CXX_FLAGS "-O3")

set(CMAKE_THREAD_LIBS_INIT "-fopenmp")
set(OpenMP_CXX_FLAGS "-fopenmp")

##############################################################

# set the search path for the environment coming with the compiler
# and a directory where you can install your own compiled software
set(CMAKE_FIND_ROOT_PATH
    /bgsys/drivers/ppcfloor/
    /bgsys/drivers/ppcfloor/gnu-linux/powerpc64-bgq-linux
    /bgsys/drivers/ppcfloor/comm/gcc
    /bgsys/drivers/ppcfloor/comm/sys/
    /bgsys/drivers/ppcfloor/spi/)

# adjust the default behaviour of the FIND_XXX() commands:
# search headers and libraries in the target environment, search
# programs in the host environment
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

##############################################################

set(LAPACK_LIB "/soft/libraries/alcf/current/gcc/LAPACK/lib")
set(ESSL_LIB "/soft/libraries/essl/current/essl/5.1/lib64")
set(IBMCMP_ROOT "/soft/compilers/ibmcmp-nov2012")
set(XLF_LIB "${IBMCMP_ROOT}/xlf/bg/14.1/bglib64")
set(XLSMP_LIB "${IBMCMP_ROOT}/xlsmp/bg/3.1/bglib64")

set(LAPACK_FLAGS "-L${LAPACK_LIB} -llapack")
set(XLF_FLAGS "-L${XLF_LIB} -lxlf90_r")
set(ESSL_FLAGS "-L${ESSL_LIB} -lesslbg")
set(XL_FLAGS "-L${XLSMP_LIB} -lxlomp_ser")
set(XLMASS_ROOT "${IBMCMP_ROOT}/xlmass/bg/7.3/bglib64")
set(XLMASS_FLAGS "-L${XLMASS_ROOT} -lmassv -lmass")

#set(CMAKE_FIND_LIBRARY_SUFFIXES .a ${CMAKE_FIND_LIBRARY_SUFFIXES})
#set(CMAKE_EXE_LINKER_FLAGS "-static")
set(MATH_LIBS "${LAPACK_FLAGS} ${ESSL_FLAGS} ${XLF_FLAGS} ${XL_FLAGS} ${XLMASS_FLAGS} -lxlopt -lxlfmath -lxl -lgfortran -lm -lpthread -ldl -Wl,--allow-multiple-definition")
