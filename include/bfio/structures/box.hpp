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
#ifndef BFIO_STRUCTURES_BOX_HPP
#define BFIO_STRUCTURES_BOX_HPP 1

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

#endif // BFIO_STRUCTURES_BOX_HPP

