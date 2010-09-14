/*
   ButterflyFIO: a distributed-memory fast algorithm for applying FIOs.
   Copyright (C) 2010 Jack Poulson <jack.poulson@gmail.com>
 
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
#pragma once
#ifndef BFIO_TOOLS_UNIFORM_HPP
#define BFIO_TOOLS_UNIFORM_HPP 1

#include <cstdlib>

namespace bfio {

// Samples uniformly within (0,1]
template<typename R>
inline R
Uniform()
{ return static_cast<R>(rand())/RAND_MAX; }

} // bfio

#endif // BFIO_TOOLS_UNIFORM_HPP

