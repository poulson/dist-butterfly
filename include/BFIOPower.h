/*
   Copyright 2010 Jack Poulson

   This file is part of ButterflyFIO.

   This program is free software: you can redistribute it and/or modify it under
   the terms of the GNU Lesser General Public License as published by the
   Free Software Foundation; either version 3 of the License, or 
   (at your option) any later version.

   This program is distributed in the hope that it will be useful, but 
   WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public License
   along with this program. If not, see <http://www.gnu.org/licenses/>.
*/
#ifndef BFIO_POWER_H
#define BFIO_POWER_H 1

namespace BFIO
{
    // Power<x,y>::value returns x to the y'th power at compile-time
    template<unsigned x,unsigned y>
    struct Power
    { enum { value = x * Power<x,y-1>::value }; };

    template<unsigned x>
    struct Power<x,1>
    { enum { value = x }; };

    template<unsigned x>
    struct Power<x,0>
    { enum { value = 1 }; };
}

#endif /* BFIO_POWER_H */

