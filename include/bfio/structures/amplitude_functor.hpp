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
#ifndef BFIO_AMPLITUDE_FUNCTOR_HPP
#define BFIO_AMPLITUDE_FUNCTOR_HPP 1

#include "bfio/structures/data.hpp"

namespace bfio {

/*
   The MiddleSwitch algorithm is the approach suggested by Nicholas Maxwell 
   and Laurent Demanet, where one simply scales the weights at the interpolation
   switch by the amplitude function. No assumptions are made on the amplitude 
   function other than that its oscillations are sufficiently resolved at the 
   middle level. This approach often requires very high order interpolation.

   Since the amplitude function is strictly positive in many applications, an 
   approach similar to that used for interpolating the phase function can be 
   used for the amplitude function: attempt to prefactor the oscillations 
   by dividing them out, interpolating, and adding them back in.
*/
enum AmplitudeAlgorithm { MiddleSwitch, Prefactor };

// You will need to derive from this class and override the operator()
template<typename R,unsigned d>
class AmplitudeFunctor
{
public:
    AmplitudeAlgorithm algorithm;

    AmplitudeFunctor() : algorithm(MiddleSwitch) {}
    AmplitudeFunctor( AmplitudeAlgorithm alg ) : algorithm(alg) {}

    virtual ~AmplitudeFunctor() {}
    virtual std::complex<R> operator() 
    ( const Array<R,d>& x, const Array<R,d>& p ) const = 0;
};

} // bfio

#endif // BFIO_AMPLITUDE_FUNCTOR_HPP

