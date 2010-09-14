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
#ifndef BFIO_LOW_RANK_SOURCE_HPP
#define BFIO_LOW_RANK_SOURCE_HPP 1

#include "bfio/structures/data.hpp"
#include "bfio/structures/amplitude_functor.hpp"
#include "bfio/structures/phase_functor.hpp"
#include "bfio/tools/lagrange.hpp"

namespace bfio {

// Low-rank source
template<typename R,unsigned d,unsigned q>
class LowRankSource
{
    const AmplitudeFunctor<R,d>& _Amp;
    const PhaseFunctor<R,d>& _Phi;
    Array<R,d> _wB;
    Array<R,d> _x0;
    Array<R,d> _p0;
    PointGrid<R,d,q> _pointGrid;
    WeightGrid<R,d,q> _weightGrid;

public:
    LowRankSource
    ( const AmplitudeFunctor<R,d>& Amp,
      const PhaseFunctor<R,d>& Phi )
    : _Amp(Amp), _Phi(Phi)
    { }

    const AmplitudeFunctor<R,d>&
    GetAmplitudeFunctor() const
    { return _Amp; }

    const PhaseFunctor<R,d>&
    GetPhaseFunctor() const
    { return _Phi; }

    const Array<R,d>&
    GetSpatialCenter() const
    { return _x0; }

    void 
    SetSpatialCenter( const Array<R,d>& x0 )
    { _x0 = x0; }

    const Array<R,d>&
    GetFreqWidths() const
    { return _wB; }

    void
    SetFreqWidths( const Array<R,d>& wB )
    { _wB = wB; }

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
    
    std::complex<R> operator()( const Array<R,d>& p );
};

} // bfio

// Implementations
namespace bfio {

template<typename R,unsigned d,unsigned q>
inline std::complex<R>
LowRankSource<R,d,q>::operator()( const Array<R,d>& p )
{
    typedef std::complex<R> C;

    // Convert p to the reference domain of [-1/2,+1/2]^d
    Array<R,d> pRef;
    for( unsigned j=0; j<d; ++j )
        pRef[j] = (p[j]-_p0[j])/_wB[j];

    C value(0.,0.);
    for( unsigned t=0; t<Pow<q,d>::val; ++t )
    {
        C beta = ImagExp( TwoPi*_Phi(_x0,_pointGrid[t]) );
        if( _Amp.algorithm == MiddleSwitch )
        {
            value += Lagrange<R,d,q>(t,pRef) * _weightGrid[t] / beta;
        }
        else if( _Amp.algorithm == Prefactor )
        {
            value += Lagrange<R,d,q>(t,pRef) * _weightGrid[t] /
                     ( _Amp(_x0,_pointGrid[t]) * beta );
        }
    }
    C beta = ImagExp( TwoPi*_Phi(_x0,p) );
    if( _Amp.algorithm == MiddleSwitch )
    {
        value *= beta;
    }
    else if( _Amp.algorithm == Prefactor )
    {
        value *= _Amp(_x0,p) * beta;
    }

    return value;
}

} // bfio

#endif // BFIO_LOW_RANK_SOURCE_HPP

