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
#ifndef BFIO_LOW_RANK_POTENTIAL_HPP
#define BFIO_LOW_RANK_POTENTIAL_HPP 1

#include "bfio/structures/data.hpp"
#include "bfio/structures/amplitude_functor.hpp"
#include "bfio/structures/phase_functor.hpp"
#include "bfio/tools/lagrange.hpp"

namespace bfio {

template<typename R,unsigned d,unsigned q>
class LowRankPotential
{
    const AmplitudeFunctor<R,d>& _Amp;
    const PhaseFunctor<R,d>& _Phi;
    unsigned _N;
    Array<R,d> _x0;
    Array<R,d> _p0;
    PointGrid<R,d,q> _pointGrid;
    WeightGrid<R,d,q> _weightGrid;

public:
    LowRankPotential
    ( const AmplitudeFunctor<R,d>& Amp, 
      const PhaseFunctor<R,d>& Phi, 
      unsigned N )
    : _Amp(Amp), _Phi(Phi), _N(N)
    { }

    const AmplitudeFunctor<R,d>&
    GetAmplitudeFunctor() const
    { return _Amp; }

    const PhaseFunctor<R,d>&
    GetPhaseFunctor() const
    { return _Phi; }

    unsigned
    GetN() const
    { return _N; }

    const Array<R,d>&
    GetSpatialCenter() const
    { return _x0; }

    void 
    SetSpatialCenter( const Array<R,d>& x0 )
    { _x0 = x0; }

    const Array<R,d>&
    GetFreqCenter() const
    { return _p0; }

    void 
    SetFreqCenter( const Array<R,d>& p0 )
    { _p0 = p0; }

    const PointGrid<R,d,q>&
    GetPointGrid() const
    { return _pointGrid; }

    void 
    SetPointGrid( const PointGrid<R,d,q>& pointGrid )
    { _pointGrid = pointGrid; }

    const WeightGrid<R,d,q>&
    GetWeightGrid() const
    { return _weightGrid; }

    void 
    SetWeightGrid( const WeightGrid<R,d,q>& weightGrid )
    { _weightGrid = weightGrid; }
    
    std::complex<R> operator()( const Array<R,d>& x );
};

} // bfio

// Implementations
namespace bfio {

template<typename R,unsigned d,unsigned q>
inline std::complex<R>
LowRankPotential<R,d,q>::operator()( const Array<R,d>& x )
{
    typedef std::complex<R> C;

    // Convert x to the reference domain of [-1/2,+1/2]^d
    Array<R,d> xRef;
    for( unsigned j=0; j<d; ++j )
        xRef[j] = (x[j]-_x0[j])*_N;

    C value(0.,0.);
    for( unsigned t=0; t<Pow<q,d>::val; ++t )
    {
        R alpha = TwoPi * _Phi(_pointGrid[t],_p0);
        if( _Amp.algorithm == MiddleSwitch )
        {
            value += Lagrange<R,d,q>(t,xRef) * _weightGrid[t] / 
                     C(cos(alpha),sin(alpha));
        }
        else if( _Amp.algorithm == Prefactor )
        {
            value += Lagrange<R,d,q>(t,xRef) * _weightGrid[t] / 
                     ( _Amp(_pointGrid[t],_p0) * C(cos(alpha),sin(alpha)) );
        }
    }
    R alpha = TwoPi * _Phi(x,_p0);
    if( _Amp.algorithm == MiddleSwitch )
    {
        value *= C(cos(alpha),sin(alpha));
    }
    else if( _Amp.algorithm == Prefactor )
    {
        value *= _Amp(x,_p0) * C(cos(alpha),sin(alpha));
    }
    return value;
}

} // bfio

#endif /* BFIO_LOW_RANK_POTENTIAL_HPP */

