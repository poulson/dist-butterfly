/*
   Copyright (C) 2010-2013 Jack Poulson and Lexing Ying
 
   This file is part of ButterflyFIO and is under the GNU General Public 
   License, which can be found in the LICENSE file in the root directory, or at
   <http://www.gnu.org/licenses/>.
*/
#pragma once
#ifndef BFIO_RFIO_CONTEXT_HPP
#define BFIO_RFIO_CONTEXT_HPP

#include <memory>
#include <vector>
#include "bfio/constants.hpp"
#include "bfio/structures/array.hpp"

namespace bfio {

namespace rfio {
template<typename R,std::size_t d,std::size_t q>
class Context
{
    std::vector<R> _chebyshevNodes;
    std::vector<R> _leftChebyshevMap;
    std::vector<R> _rightChebyshevMap;
    std::vector< Array<std::size_t,d> > _chebyshevIndices;
    std::vector< Array<R,          d> > _chebyshevGrid;
    std::vector< Array<R,          d> > _sourceChildGrids;

    void GenerateChebyshevNodes();
    void GenerateChebyshevIndices();
    void GenerateChebyshevGrid();
    void GenerateChebyshevMaps();
    void GenerateChildGrids();
    
public:        
    Context();

    // Evaluate a 1d Lagrangian basis function at point p in [-1/2,+1/2]
    R Lagrange1d( std::size_t i, R p ) const;

    // Evaluate the t'th Lagrangian basis function at point p in [-1/2,+1/2]^d
    R Lagrange( std::size_t t, const Array<R,d>& p ) const;

    void LagrangeBatch
    ( std::size_t t, 
      const std::vector< Array<R,d> >& p,
            std::vector< R          >& results ) const;

    const std::vector<R>&
    GetChebyshevNodes() const;

    const std::vector< Array<std::size_t,d> >&
    GetChebyshevIndices() const;

    const std::vector< Array<R,d> >&
    GetChebyshevGrid() const;

    const std::vector<R>&
    GetLeftChebyshevMap() const;

    const std::vector<R>&
    GetRightChebyshevMap() const;

    const std::vector< Array<R,d> >&
    GetSourceChildGrids() const;
};
} // rfio

// Implementations

template<typename R,std::size_t d,std::size_t q>
void 
rfio::Context<R,d,q>::GenerateChebyshevNodes()
{
    for( std::size_t t=0; t<q; ++t )
        _chebyshevNodes[t] = 0.5*cos(static_cast<R>(t*Pi/(q-1)));
}

template<typename R,std::size_t d,std::size_t q>
void 
rfio::Context<R,d,q>::GenerateChebyshevIndices()
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
void 
rfio::Context<R,d,q>::GenerateChebyshevGrid()
{
    const std::size_t q_to_d = _chebyshevGrid.size();

    for( std::size_t t=0; t<q_to_d; ++t )
    {
        std::size_t q_to_j = 1;
        for( std::size_t j=0; j<d; ++j )
        {
            std::size_t i = (t/q_to_j) % q;
            _chebyshevGrid[t][j] = 0.5*cos(static_cast<R>(i*Pi/(q-1)));
            q_to_j *= q;
        }
    }
}

template<typename R,std::size_t d,std::size_t q>
void
rfio::Context<R,d,q>::GenerateChebyshevMaps()
{
    // Create 1d Lagrangian evaluation maps being left and right of the center
    for( std::size_t i=0; i<q; ++i )
        for( std::size_t k=0; k<q; ++k )
            _leftChebyshevMap[k*q+i] = 
                Lagrange1d( i, (2*_chebyshevNodes[k]-1)/4 );
    for( std::size_t i=0; i<q; ++i )
        for( std::size_t k=0; k<q; ++k )
            _rightChebyshevMap[k*q+i] = 
                Lagrange1d( i, (2*_chebyshevNodes[k]+1)/4 );
}

template<typename R,std::size_t d,std::size_t q>
void 
rfio::Context<R,d,q>::GenerateChildGrids()
{
    const std::size_t q_to_d = _chebyshevGrid.size();

    for( std::size_t c=0; c<(1u<<d); ++c )
    {
        for( std::size_t tPrime=0; tPrime<q_to_d; ++tPrime )
        {

            // Map p_t'(Bc) to the reference domain ([-1/2,+1/2]^d) of B
            for( std::size_t j=0; j<d; ++j )
            {
                _sourceChildGrids[c*q_to_d+tPrime][j] = 
                    ( (c>>j)&1 ? (2*_chebyshevGrid[tPrime][j]+1)/4 
                               : (2*_chebyshevGrid[tPrime][j]-1)/4 );
            }
        }
    }
}

template<typename R,std::size_t d,std::size_t q>
rfio::Context<R,d,q>::Context() 
: _chebyshevNodes( q ),
  _leftChebyshevMap( q*q ),
  _rightChebyshevMap( q*q ),
  _chebyshevIndices( Pow<q,d>::val ), 
  _chebyshevGrid( Pow<q,d>::val ),
  _sourceChildGrids( Pow<q,d>::val<<d )
{
    GenerateChebyshevNodes();
    GenerateChebyshevIndices();
    GenerateChebyshevGrid();
    GenerateChebyshevMaps();
    GenerateChildGrids();
}

template<typename R,std::size_t d,std::size_t q>
R
rfio::Context<R,d,q>::Lagrange1d
( std::size_t i, R p ) const
{
    R product = static_cast<R>(1);
    const R* chebyshevNodeBuffer = &_chebyshevNodes[0];
    for( std::size_t k=0; k<q; ++k )
    {
        if( i != k )
        {
            const R iNode = chebyshevNodeBuffer[i];
            const R kNode = chebyshevNodeBuffer[k];
            product *= (p-kNode) / (iNode-kNode);
        }
    }
    return product;
}

template<typename R,std::size_t d,std::size_t q>
R
rfio::Context<R,d,q>::Lagrange
( std::size_t t, const Array<R,d>& p ) const
{
    R product = static_cast<R>(1);
    const R* RESTRICT pBuffer = &p[0];
    const R* RESTRICT chebyshevNodeBuffer = &_chebyshevNodes[0];
    const std::size_t* RESTRICT chebyshevIndicesBuffer = 
        &_chebyshevIndices[t][0];
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
rfio::Context<R,d,q>::LagrangeBatch
( std::size_t t, 
  const std::vector< Array<R,d> >& p, 
        std::vector< R          >& results ) const
{
    results.resize( p.size() );
    R* resultsBuffer = &results[0];
    for( std::size_t i=0; i<p.size(); ++i )
        resultsBuffer[i] = 1;

    const R* RESTRICT pBuffer = &p[0][0];
    const R* RESTRICT chebyshevNodeBuffer = &_chebyshevNodes[0];
    const std::size_t* RESTRICT chebyshevIndicesBuffer = 
        &_chebyshevIndices[t][0];
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
inline const std::vector<R>&
rfio::Context<R,d,q>::GetChebyshevNodes() const
{ return _chebyshevNodes; }

template<typename R,std::size_t d,std::size_t q>
inline const std::vector< Array<std::size_t,d> >&
rfio::Context<R,d,q>::GetChebyshevIndices() const
{ return _chebyshevIndices; }

template<typename R,std::size_t d,std::size_t q>
inline const std::vector< Array<R,d> >&
rfio::Context<R,d,q>::GetChebyshevGrid() const
{ return _chebyshevGrid; }

template<typename R,std::size_t d,std::size_t q>
inline const std::vector<R>& 
rfio::Context<R,d,q>::GetLeftChebyshevMap() const
{ return _leftChebyshevMap; }

template<typename R,std::size_t d,std::size_t q>
inline const std::vector<R>& 
rfio::Context<R,d,q>::GetRightChebyshevMap() const
{ return _rightChebyshevMap; }

template<typename R,std::size_t d,std::size_t q>
inline const std::vector< Array<R,d> >&
rfio::Context<R,d,q>::GetSourceChildGrids() const
{ return _sourceChildGrids; }

} // bfio

#endif // ifndef BFIO_RFIO_CONTEXT_HPP
