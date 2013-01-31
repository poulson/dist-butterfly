/*
   Copyright (C) 2010-2013 Jack Poulson and Lexing Ying
 
   This file is part of ButterflyFIO and is under the GNU General Public 
   License, which can be found in the LICENSE file in the root directory, or at
   <http://www.gnu.org/licenses/>.
*/
#pragma once
#ifndef BFIO_CONSTANTS_HPP
#define BFIO_CONSTANTS_HPP

#include <cstddef>

namespace bfio {

static const double Pi    = 3.141592653589793;
static const double TwoPi = 6.283185307179586;

// Pow<x,y>::val returns x to the y'th power at compile-time
template<std::size_t x,std::size_t y>
struct Pow
{ enum { val = x * Pow<x,y-1>::val }; };

template<std::size_t x>
struct Pow<x,1>
{ enum { val = x }; };

template<std::size_t x>
struct Pow<x,0>
{ enum { val = (size_t)1 }; };

enum Direction { FORWARD, ADJOINT };

} // bfio

#endif // ifndef BFIO_CONSTANTS_HPP
