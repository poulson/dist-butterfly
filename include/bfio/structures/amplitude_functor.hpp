/*
   Copyright (c) 2010, Jack Poulson
   All rights reserved.

   This file is part of ButterflyFIO.

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are met:

    - Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.

    - Redistributions in binary form must reproduce the above copyright notice,
      this list of conditions and the following disclaimer in the documentation
      and/or other materials provided with the distribution.

    - Neither the name of the owner nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
   AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
   IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
   ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
   LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
   CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
   SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
   INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
   CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
   ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
   POSSIBILITY OF SUCH DAMAGE.
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

#endif /* BFIO_AMPLITUDE_FUNCTOR_HPP */

