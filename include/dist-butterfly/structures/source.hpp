/*
   Copyright (C) 2010-2013 Jack Poulson and Lexing Ying
 
   This file is part of DistButterfly and is under the GNU General Public 
   License, which can be found in the LICENSE file in the root directory, or at
   <http://www.gnu.org/licenses/>.
*/
#pragma once
#ifndef DBF_STRUCTURES_SOURCE_HPP
#define DBF_STRUCTURES_SOURCE_HPP

#include <array>
#include <cstddef>
#include <complex>

namespace dbf {

template<typename R,std::size_t d>
struct Source 
{ 
    std::array<R,d> p;
    std::complex<R> magnitude;
};

} // dbf

#endif // ifndef DBF_STRUCTURES_SOURCE_HPP
