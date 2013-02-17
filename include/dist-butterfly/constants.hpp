/*
   Copyright (C) 2010-2013 Jack Poulson and Lexing Ying
 
   This file is part of DistButterfly and is under the GNU General Public 
   License, which can be found in the LICENSE file in the root directory, or at
   <http://www.gnu.org/licenses/>.
*/
#pragma once
#ifndef DBF_CONSTANTS_HPP
#define DBF_CONSTANTS_HPP

#include <cstddef>

namespace dbf {

template<typename R>
R Pi();

template<>
inline float Pi<float>()
{ return 3.14159265358979f; }

template<>
inline double Pi<double>()
{ return 3.141592653589793238462; }

template<typename R>
R TwoPi();

template<>
inline float TwoPi<float>()
{ return 6.28318530717958f; }

template<>
inline double TwoPi<double>()
{ return 6.283185307179586476924; }

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

} // dbf

#endif // ifndef DBF_CONSTANTS_HPP
