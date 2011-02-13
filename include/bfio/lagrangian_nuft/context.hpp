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
#ifndef BFIO_LAGRANGIAN_NUFT_CONTEXT_HPP
#define BFIO_LAGRANGIAN_NUFT_CONTEXT_HPP 1

#include "bfio/fio_from_ft/context.hpp"

namespace bfio {

namespace lagrangian_nuft {
template<typename R,std::size_t d,std::size_t q>
class Context
{
    const fio_from_ft::Context<R,d,q> _fioContext;
    const std::size_t _N;
    const Box<R,d> _sourceBox;
    const Box<R,d> _targetBox;

    Array< std::vector<R>, d > _realOffsetEvaluations;
    Array< std::vector<R>, d > _imagOffsetEvaluations;

    void GenerateOffsetEvaluations();

public:        
    Context
    ( const std::size_t N,
      const Box<R,d>& sourceBox,
      const Box<R,d>& targetBox );

    const fio_from_ft::Context<R,d,q>&
    GetFIOContext() const;

    const Array< std::vector<R>, d >&
    GetRealOffsetEvaluations() const;

    const Array< std::vector<R>, d >&
    GetImagOffsetEvaluations() const;
};
} // lagrangian_nuft

// Implementations

template<typename R,std::size_t d,std::size_t q>
void
lagrangian_nuft::Context<R,d,q>::GenerateOffsetEvaluations()
{
    const std::size_t log2N = Log2( _N );
    const std::size_t middleLevel = log2N/2;

    Array<R,d> wAMiddle, wBMiddle;
    for( std::size_t j=0; j<d; ++j )
    {
        wAMiddle[j] = _targetBox.widths[j] / (1<<middleLevel);
        wBMiddle[j] = _sourceBox.widths[j] / (1<<(log2N-middleLevel));
    }

    // Form the offset grid evaluations
    std::vector<R> phaseEvaluations(q*q);
    const std::vector<R>& chebyshevNodes = _fioContext.GetChebyshevNodes();
    const R* chebyshevBuffer = &chebyshevNodes[0];
    for( std::size_t j=0; j<d; ++j )
    {
        for( std::size_t t=0; t<q; ++t )
            for( std::size_t tPrime=0; tPrime<q; ++tPrime )
                phaseEvaluations[t*q+tPrime] =
                    TwoPi*wAMiddle[j]*wBMiddle[j]*
                    chebyshevBuffer[t]*chebyshevBuffer[tPrime];
        SinCosBatch
        ( phaseEvaluations, 
          _imagOffsetEvaluations[j], _realOffsetEvaluations[j] );
    }
}

template<typename R,std::size_t d,std::size_t q>
lagrangian_nuft::Context<R,d,q>::Context
( std::size_t N, const Box<R,d>& sourceBox, const Box<R,d>& targetBox ) 
: _fioContext(), _N(N), _sourceBox(sourceBox), _targetBox(targetBox)
{ GenerateOffsetEvaluations(); }

template<typename R,std::size_t d,std::size_t q>
const fio_from_ft::Context<R,d,q>&
lagrangian_nuft::Context<R,d,q>::GetFIOContext() const
{ return _fioContext; }

template<typename R,std::size_t d,std::size_t q>
const Array< std::vector<R>, d >&
lagrangian_nuft::Context<R,d,q>::GetRealOffsetEvaluations() const
{ return _realOffsetEvaluations; }

template<typename R,std::size_t d,std::size_t q>
const Array< std::vector<R>, d >&
lagrangian_nuft::Context<R,d,q>::GetImagOffsetEvaluations() const
{ return _imagOffsetEvaluations; }

} // bfio

#endif // BFIO_LAGRANGIAN_NUFT_CONTEXT_HPP

