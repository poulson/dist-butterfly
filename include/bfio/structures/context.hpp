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
#ifndef BFIO_STRUCTURES_CONTEXT_HPP
#define BFIO_STRUCTURES_CONTEXT_HPP 1

#include <memory>
#include <vector>
#include "bfio/constants.hpp"
#include "bfio/structures/array.hpp"

namespace bfio {

template<typename R,std::size_t d,std::size_t q>
class Context
{
    Array<R,q> _chebyshevNodes;
    std::vector< Array<std::size_t,d> > _chebyshevIndices;
    std::vector< Array<R,d> > _chebyshevGrid;
    std::vector<R> _freqMaps;
    std::vector<R> _spatialMaps;
    std::vector< Array<R,d> > _freqChildGrids;

    void GenerateChebyshevNodes();
    void GenerateChebyshevIndices();
    void GenerateChebyshevGrid();
    void GenerateFreqMapsAndChildGrids();
    void GenerateSpatialMaps();
    
public:        
    Context();

    // Evaluate the t'th Lagrangian basis function at point p in [-1/2,+1/2]^d
    R Lagrange( std::size_t t, const Array<R,d>& p ) const;

    void LagrangeBatch
    ( std::size_t t, 
      const std::vector< Array<R,d> >& p,
            std::vector< R          >& results ) const;

    const Array<R,d>&
    GetChebyshevNodes() const;

    const std::vector< Array<std::size_t,d> >&
    GetChebyshevIndices() const;

    const std::vector< Array<R,d> >&
    GetChebyshevGrid() const;

    const std::vector<R>&
    GetFreqMaps() const;

    const std::vector<R>&
    GetSpatialMaps() const;

    const std::vector< Array<R,d> >&
    GetFreqChildGrids() const;
};

// Implementations

template<typename R,std::size_t d,std::size_t q>
void Context<R,d,q>::GenerateChebyshevNodes()
{
    for( std::size_t t=0; t<q; ++t )
        _chebyshevNodes[t] = 0.5*cos(static_cast<R>(t*Pi/(q-1)));
}

template<typename R,std::size_t d,std::size_t q>
void Context<R,d,q>::GenerateChebyshevIndices()
{
    const std::size_t q_to_d = _chebyshevIndices.size();

    for( std::size_t t=0; t<q_to_d; ++t )
    {
        std::size_t qToThej = 1;
        for( std::size_t j=0; j<d; ++j )
        {
            std::size_t i = (t/qToThej) % q;
            _chebyshevIndices[t][j] = i;
            qToThej *= q;
        }
    }
}

template<typename R,std::size_t d,std::size_t q>
void Context<R,d,q>::GenerateChebyshevGrid()
{
    const std::size_t q_to_d = _chebyshevGrid.size();

    for( std::size_t t=0; t<q_to_d; ++t )
    {
        std::size_t qToThej = 1;
        for( std::size_t j=0; j<d; ++j )
        {
            std::size_t i = (t/qToThej)%q;
            _chebyshevGrid[t][j] = 0.5*cos(static_cast<R>(i*Pi/(q-1)));
            qToThej *= q;
        }
    }
}

template<typename R,std::size_t d,std::size_t q>
void Context<R,d,q>::GenerateFreqMapsAndChildGrids()
{
    const std::size_t q_to_d = _chebyshevGrid.size();
    const std::size_t q_to_2d = q_to_d * q_to_d;

    for( std::size_t c=0; c<(1u<<d); ++c )
    {
        for( std::size_t tPrime=0; tPrime<q_to_d; ++tPrime )
        {

            // Map p_t'(Bc) to the reference domain ([-1/2,+1/2]^d) of B
            for( std::size_t j=0; j<d; ++j )
            {
                _freqChildGrids[c*q_to_d+tPrime][j] = 
                    ( (c>>j)&1 ? (2*_chebyshevGrid[tPrime][j]+1)/4 
                               : (2*_chebyshevGrid[tPrime][j]-1)/4 );
            }
        }
    }

    // Store all of the Lagrangian evaluations on p_t'(Bc)'s
    for( std::size_t c=0; c<(1u<<d); ++c )
    {
        for( std::size_t t=0; t<q_to_d; ++t )
        {
            for( std::size_t tPrime=0; tPrime<q_to_d; ++tPrime )
            {
                _freqMaps[c*q_to_2d+tPrime*q_to_d+t] = 
                    Lagrange( t, _freqChildGrids[c*q_to_d+tPrime] ); 
            }
        }
    }
}

template<typename R,std::size_t d,std::size_t q>
void Context<R,d,q>::GenerateSpatialMaps()
{
    const std::size_t q_to_d = _chebyshevGrid.size();
    const std::size_t q_to_2d = q_to_d * q_to_d;
    for( std::size_t p=0; p<(1u<<d); ++p )
    {
        for( std::size_t t=0; t<q_to_d; ++t )
        {
            // Map x_t(A) to the reference domain ([-1/2,+1/2]^d) of its parent.
            Array<R,d> xtARefAp;
            for( std::size_t j=0; j<d; ++j )
            {
                xtARefAp[j] = 
                    ( (p>>j)&1 ? (2*_chebyshevGrid[t][j]+1)/4 
                               : (2*_chebyshevGrid[t][j]-1)/4 ); 
            }

            for( std::size_t tPrime=0; tPrime<q_to_d; ++tPrime )
            {
                _spatialMaps[p*q_to_2d + t+tPrime*q_to_d] = 
                    Lagrange( tPrime, xtARefAp );
            }
        }
    }
}

template<typename R,std::size_t d,std::size_t q>
Context<R,d,q>::Context() 
: _chebyshevIndices(Pow<q,d>::val), 
  _chebyshevGrid(Pow<q,d>::val),
  _freqMaps( Pow<q,2*d>::val<<d ),
  _spatialMaps( Pow<q,2*d>::val<<d ),
  _freqChildGrids( Pow<q,d>::val<<d )
{
    GenerateChebyshevNodes();
    GenerateChebyshevIndices();
    GenerateChebyshevGrid();
    GenerateFreqMapsAndChildGrids();
    GenerateSpatialMaps();
}

template<typename R,std::size_t d,std::size_t q>
R
Context<R,d,q>::Lagrange
( std::size_t t, const Array<R,d>& p ) const
{
    R product = static_cast<R>(1);
    const R* pBuffer = &p[0];
    const R* chebyshevNodeBuffer = &_chebyshevNodes[0];
    const std::size_t* chebyshevIndicesBuffer = &(_chebyshevIndices[t][0]);
    for( std::size_t j=0; j<d; ++j )
    {
        std::size_t i = chebyshevIndicesBuffer[j];
        for( std::size_t k=0; k<q; ++k )
        {
            if( i != k )
            {
                const R iNode = chebyshevNodeBuffer[i];
                const R kNode = chebyshevNodeBuffer[k];
                const R nodeDist = iNode - kNode;
                product *= (pBuffer[j]-kNode) / nodeDist;
            }
        }
    }
    return product;
}

template<typename R,std::size_t d,std::size_t q>
void
Context<R,d,q>::LagrangeBatch
( std::size_t t, 
  const std::vector< Array<R,d> >& p, 
        std::vector< R          >& results ) const
{
    results.resize( p.size() );
    R* resultsBuffer = &results[0];
    for( std::size_t i=0; i<p.size(); ++i )
        resultsBuffer[i] = 1;

    const R* pBuffer = &(p[0][0]);
    const R* chebyshevNodeBuffer = &_chebyshevNodes[0];
    const std::size_t* chebyshevIndicesBuffer = &(_chebyshevIndices[t][0]);
    for( std::size_t j=0; j<d; ++j )
    {
        std::size_t i = chebyshevIndicesBuffer[j];
        for( std::size_t k=0; k<q; ++k )
        {
            if( i != k )
            {
                const R iNode = chebyshevNodeBuffer[i];
                const R kNode = chebyshevNodeBuffer[k];
                const R nodeDist = iNode - kNode;
                for( std::size_t r=0; r<p.size(); ++r )
                    resultsBuffer[r] *= (pBuffer[r*d+j]-kNode) / nodeDist;
            }
        }
    }
}

template<typename R,std::size_t d,std::size_t q>
const Array<R,d>&
Context<R,d,q>::GetChebyshevNodes() const
{ return _chebyshevNodes; }

template<typename R,std::size_t d,std::size_t q>
const std::vector< Array<std::size_t,d> >&
Context<R,d,q>::GetChebyshevIndices() const
{ return _chebyshevIndices; }

template<typename R,std::size_t d,std::size_t q>
const std::vector< Array<R,d> >&
Context<R,d,q>::GetChebyshevGrid() const
{ return _chebyshevGrid; }

template<typename R,std::size_t d,std::size_t q>
const std::vector<R>&
Context<R,d,q>::GetFreqMaps() const
{ return _freqMaps; }

template<typename R,std::size_t d,std::size_t q>
const std::vector<R>&
Context<R,d,q>::GetSpatialMaps() const
{ return _spatialMaps; }

template<typename R,std::size_t d,std::size_t q>
const std::vector< Array<R,d> >&
Context<R,d,q>::GetFreqChildGrids() const
{ return _freqChildGrids; }

} // bfio

#endif // BFIO_STRUCTURES_CONTEXT_HPP

