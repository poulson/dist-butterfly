/*
   Copyright (C) 2010-2013 Jack Poulson and Lexing Ying
 
   This file is part of ButterflyFIO and is under the GNU General Public 
   License, which can be found in the LICENSE file in the root directory, or at
   <http://www.gnu.org/licenses/>.
*/
#pragma once
#ifndef BFIO_STRUCTURES_BOX_HPP
#define BFIO_STRUCTURES_BOX_HPP

#include <cstddef>
#include "bfio/structures/array.hpp"

namespace bfio {

template<typename R,std::size_t d>
struct Box
{
    Array<R,d> widths;
    Array<R,d> offsets;
};

} // bfio

#endif // ifndef BFIO_STRUCTURES_BOX_HPP
