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
#pragma once
#ifndef BFIO_PHASE_FUNCTOR_HPP
#define BFIO_PHASE_FUNCTOR_HPP 1

#include "bfio/structures/data.hpp"

namespace bfio {

// You will need to derive from this class and override the operator()
template<typename R,unsigned d>
class PhaseFunctor
{
public:
    virtual ~PhaseFunctor() {}
    virtual R operator() 
    ( const Array<R,d>& x, const Array<R,d>& p ) const = 0;
};

} // bfio

#endif // BFIO_PHASE_FUNCTOR_HPP

