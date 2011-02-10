set(CMAKE_SYSTEM_NAME BlueGeneP-static)

# We need MPI C and CXX compilers
set(CMAKE_C_COMPILER /bgsys/drivers/ppcfloor/comm/bin/mpixlc_r)
set(CMAKE_CXX_COMPILER /bgsys/drivers/ppcfloor/comm/bin/mpixlcxx_r)

set(CXX_FLAGS "-g -O4 -DBGP")

set(ESSL_BASE "/soft/apps/ESSL-4.3.1-1")
set(IBMCMP_BASE "/soft/apps/ibmcmp-dec2010")
set(XLF_BASE "${IBMCMP_BASE}/xlf/bg/11.1/bglib")
set(XLSMP_BASE "${IBMCMP_BASE}/xlsmp/bg/1.7/bglib")
set(BGP_LAPACK "-L/soft/apps/LAPACK -llapack_bgp")
set(ESSL "-L${ESSL_BASE}/lib -lesslbg")
set(XLF_LIBS "-L${XLF_BASE} -lxlfmath -lxlf90_r")
set(XLOMP_SER "-L${XLSMP_BASE} -lxlomp_ser")

set(XLMASS_LIB "/soft/apps/ibmcmp-dec2010/xlmass/bg/4.4/bglib")
set(XLMASS "-L${XLMASS_LIB} -lmassv -lmass")

set(MATH_LIBS "${BGP_LAPACK};${ESSL};${XLF_LIBS};${XLOMP_SER};${XLMASS}")

# Make sure we can find the ESSL headers
set(ESSL_INC "/soft/apps/ESSL-4.3.1-1/include")
set(XLMASS_INC "/soft/apps/ibmcmp-dec2010/xlmass/bg/4.4/include")
include_directories(${ESSL_INC})
include_directories(${XLMASS_INC})

