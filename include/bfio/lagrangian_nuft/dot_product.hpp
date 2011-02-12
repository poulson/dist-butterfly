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
#ifndef BFIO_LAGRANGIAN_NUFT_DOT_PRODUCT_HPP
#define BFIO_LAGRANGIAN_NUFT_DOT_PRODUCT_HPP 1

#include "bfio/functors/phase_functor.hpp"

namespace bfio {

namespace lagrangian_nuft {
template<typename R,std::size_t d>
class DotProduct : public PhaseFunctor<R,d>
{
public:
    // Default constructors are required to initialize const classes.
    DotProduct();

    // This is the only routine that must be implemented if the object will not
    // be const.
    virtual R
    operator()
    ( const bfio::Array<R,d>& x, const bfio::Array<R,d>& p ) const;

    // We can optionally override the batched application for better efficiency
    virtual void
    BatchEvaluate
    ( const std::vector< bfio::Array<R,d> >& xPoints,
      const std::vector< bfio::Array<R,d> >& pPoints,
            std::vector< R                >& results ) const;
};
} // lagrangian_nuft

// Implementations

template<typename R,std::size_t d>
lagrangian_nuft::DotProduct<R,d>::DotProduct()
{ }

template<typename R,std::size_t d>
inline R
lagrangian_nuft::DotProduct<R,d>::operator()
( const bfio::Array<R,d>& x, const bfio::Array<R,d>& p ) const
{
    R sum = 0;
    const R* xBuffer = &x[0];
    const R* pBuffer = &p[0];
    for( std::size_t j=0; j<d; ++j )
        sum += xBuffer[j]*pBuffer[j];
    return sum;
}

template<typename R,std::size_t d>
void
lagrangian_nuft::DotProduct<R,d>::BatchEvaluate
( const std::vector< bfio::Array<R,d> >& xPoints,
  const std::vector< bfio::Array<R,d> >& pPoints,
        std::vector< R                >& results ) const
{
    const std::size_t nxPoints = xPoints.size();
    const std::size_t npPoints = pPoints.size();
    results.resize( nxPoints*npPoints );

    R* resultsBuffer = &results[0];
    std::memset( resultsBuffer, 0, nxPoints*npPoints*sizeof(R) );
    const R* xPointsBuffer = &(xPoints[0][0]);
    const R* pPointsBuffer = &(pPoints[0][0]);
    for( std::size_t i=0; i<nxPoints; ++i )
        for( std::size_t j=0; j<npPoints; ++j )
            for( std::size_t k=0; k<d; ++k )
                resultsBuffer[i*npPoints+j] += 
                    xPointsBuffer[i*d+k]*pPointsBuffer[j*d+k];
}

} // bfio

#endif // BFIO_LAGRANGIAN_NUFT_DOT_PRODUCT_HPP

