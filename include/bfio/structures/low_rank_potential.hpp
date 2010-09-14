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
    Array<R,d> _wA;
    Array<R,d> _x0;
    Array<R,d> _p0;
    PointGrid<R,d,q> _pointGrid;
    WeightGrid<R,d,q> _weightGrid;

public:
    LowRankPotential
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
    GetSpatialWidths() const
    { return _wA; }

    void
    SetSpatialWidths( const Array<R,d>& wA )
    { _wA = wA; }

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
        xRef[j] = (x[j]-_x0[j])/_wA[j];

    C value(0.,0.);
    for( unsigned t=0; t<Pow<q,d>::val; ++t )
    {
        C beta = ImagExp( TwoPi*_Phi(_pointGrid[t],_p0) );
        if( _Amp.algorithm == MiddleSwitch )
        {
            value += Lagrange<R,d,q>(t,xRef) * _weightGrid[t] / beta;
        }
        else if( _Amp.algorithm == Prefactor )
        {
            value += Lagrange<R,d,q>(t,xRef) * _weightGrid[t] / 
                     ( _Amp(_pointGrid[t],_p0) * beta );
        }
    }
    C beta = ImagExp( TwoPi*_Phi(x,_p0) );
    if( _Amp.algorithm == MiddleSwitch )
    {
        value *= beta;
    }
    else if( _Amp.algorithm == Prefactor )
    {
        value *= _Amp(x,_p0) * beta;
    }
    return value;
}

} // bfio

#endif // BFIO_LOW_RANK_POTENTIAL_HPP

