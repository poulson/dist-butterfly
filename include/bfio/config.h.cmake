/*
   ButterflyFIO: a distributed-memory fast algorithm for applying FIOs.
   Copyright (C) 2010-2011 Jack Poulson <jack.poulson@gmail.com>
 
   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.
 
   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.
 
   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/
#ifndef BFIO_CONFIG_HPP
#define BFIO_CONFIG_HPP 1

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

#endif /* BFIO_CONFIG_HPP */
