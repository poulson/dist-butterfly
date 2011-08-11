set(CMAKE_SYSTEM_NAME BlueGeneP-static)

option(BGP "We are on BG/P" ON)
# We can't activate this on non-cartesian comms, it causes segfaults
option(BGP_MPIDO_USE_REDUCESCATTER "Avoid known perf bug in reduce-scatter" OFF)

# The serial XL compilers
set(CMAKE_C_COMPILER       /soft/apps/ibmcmp-apr2011/vacpp/bg/9.0/bin/bgxlc_r)
set(CMAKE_CXX_COMPILER     /soft/apps/ibmcmp-apr2011/vacpp/bg/9.0/bin/bgxlC_r)

# The MPI wrappers for the XL C and C++ compilers
set(MPI_C_COMPILER   /bgsys/drivers/ppcfloor/comm/bin/mpixlc_r)
set(MPI_CXX_COMPILER /bgsys/drivers/ppcfloor/comm/bin/mpixlcxx_r)

set(CXX_FLAGS "-g -O4 -DBGP")

set(ESSL_BASE "/soft/apps/ESSL-4.3.1-1")
set(IBMCMP_BASE "/soft/apps/ibmcmp-apr2011")
set(XLF_BASE "${IBMCMP_BASE}/xlf/bg/11.1/bglib")
set(XLSMP_BASE "${IBMCMP_BASE}/xlsmp/bg/1.7/bglib")
set(BGP_LAPACK "-L/soft/apps/LAPACK -llapack_bgp")
set(ESSL "-L${ESSL_BASE}/lib -lesslbg")
set(XLF_LIBS "-L${XLF_BASE} -lxlfmath -lxlf90_r")
set(XLOMP_SER "-L${XLSMP_BASE} -lxlomp_ser")

set(XLMASS_LIB "/soft/apps/ibmcmp-apr2011/xlmass/bg/4.4/bglib")
set(XLMASS "-L${XLMASS_LIB} -lmassv -lmass")

set(MATH_LIBS "${BGP_LAPACK};${ESSL};${XLF_LIBS};${XLOMP_SER};${XLMASS}")

# Make sure we can find the ESSL headers
set(ESSL_INC "/soft/apps/ESSL-4.3.1-1/include")
set(XLMASS_INC "/soft/apps/ibmcmp-apr2011/xlmass/bg/4.4/include")
include_directories(${ESSL_INC})
include_directories(${XLMASS_INC})

