/*
   Copyright (C) 2010-2013 Jack Poulson and Lexing Ying
 
   This file is part of DistButterfly and is under the GNU General Public 
   License, which can be found in the LICENSE file in the root directory, or at
   <http://www.gnu.org/licenses/>.
*/
#pragma once
#ifndef DBF_STRUCTURES_BOX_HPP
#define DBF_STRUCTURES_BOX_HPP

#include <array>
#include <cstddef>

namespace dbf {

template<typename R,std::size_t d>
struct Box
{
    std::array<R,d> widths;
    std::array<R,d> offsets;
};

} // dbf

#endif // ifndef DBF_STRUCTURES_BOX_HPP
