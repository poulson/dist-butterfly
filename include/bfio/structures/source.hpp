/*
   Copyright (C) 2010-2013 Jack Poulson and Lexing Ying
 
   This file is part of ButterflyFIO and is under the GNU General Public 
   License, which can be found in the LICENSE file in the root directory, or at
   <http://www.gnu.org/licenses/>.
*/
#pragma once
#ifndef BFIO_STRUCTURES_SOURCE_HPP
#define BFIO_STRUCTURES_SOURCE_HPP

#include <cstddef>
#include <complex>
#include "bfio/structures/array.hpp"

namespace bfio {

template<typename R,std::size_t d>
struct Source 
{ 
    Array<R,d> p;
    std::complex<R> magnitude;
};

} // bfio

#endif // ifndef BFIO_STRUCTURES_SOURCE_HPP
