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
#ifndef BFIO_CONSTANTS_HPP
#define BFIO_CONSTANTS_HPP 1

namespace bfio {

static const double Pi    = 3.141592653589793;
static const double TwoPi = 6.283185307179586;

// Pow<x,y>::val returns x to the y'th power at compile-time
template<unsigned x,unsigned y>
struct Pow
{ enum { val = x * Pow<x,y-1>::val }; };

template<unsigned x>
struct Pow<x,1>
{ enum { val = x }; };

template<unsigned x>
struct Pow<x,0>
{ enum { val = 1 }; };

} // bfio

#endif // BFIO_CONSTANTS_HPP

