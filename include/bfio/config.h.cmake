/*
   Copyright (C) 2010-2013 Jack Poulson and Lexing Ying
 
   This file is part of ButterflyFIO and is under the GNU General Public 
   License, which can be found in the LICENSE file in the root directory, or at
   <http://www.gnu.org/licenses/>.
*/
#pragma once
#ifndef BFIO_CONFIG_HPP
#define BFIO_CONFIG_HPP

#define BFIO_VERSION_MAJOR @BFIO_VERSION_MAJOR@
#define BFIO_VERSION_MINOR @BFIO_VERSION_MINOR@
#define RESTRICT @RESTRICT@
#cmakedefine RELEASE
#cmakedefine TIMING
#cmakedefine BLAS_POST
#cmakedefine LAPACK_POST
#cmakedefine AVOID_COMPLEX_MPI
#cmakedefine MKL
#cmakedefine ESSL
#cmakedefine BGP
#cmakedefine BGP_MPIDO_USE_REDUCESCATTER

#endif /* ifndef BFIO_CONFIG_HPP */
