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
#ifndef BFIO_POW_HPP
#define BFIO_POW_HPP 1

namespace BFIO
{
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
}

#endif /* BFIO_POW_HPP */
