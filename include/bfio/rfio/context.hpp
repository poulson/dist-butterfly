/*
   Copyright (C) 2010-2013 Jack Poulson and Lexing Ying
 
   This file is part of ButterflyFIO and is under the GNU General Public 
   License, which can be found in the LICENSE file in the root directory, or at
   <http://www.gnu.org/licenses/>.
*/
#pragma once
#ifndef BFIO_RFIO_CONTEXT_HPP
#define BFIO_RFIO_CONTEXT_HPP

#include <array>
#include <memory>
#include <vector>

#include "bfio/constants.hpp"

namespace bfio {

using std::array;
using std::size_t;
using std::vector;

namespace rfio {

template<typename R,size_t d,size_t q>
class Context
{
    vector<R> chebyshevNodes_;
    vector<R> leftChebyshevMap_, rightChebyshevMap_;
    vector<array<size_t,d>> chebyshevIndices_;
    vector<array<R,d>> chebyshevGrid_, sourceChildGrids_;

    void GenerateChebyshevNodes();
    void GenerateChebyshevIndices();
    void GenerateChebyshevGrid();
    void GenerateChebyshevMaps();
    void GenerateChildGrids();
    
public:        
    Context();

    // Evaluate a 1d Lagrangian basis function at point p in [-1/2,+1/2]
    R Lagrange1d( size_t i, R p ) const;

    // Evaluate the t'th Lagrangian basis function at point p in [-1/2,+1/2]^d
    R Lagrange( size_t t, const array<R,d>& p ) const;

    void LagrangeBatch
    ( size_t t, 
      const vector<array<R,d>>& p,
            vector<R         >& results ) const;

    const vector<R>& GetChebyshevNodes() const;

    const vector<array<size_t,d>>& GetChebyshevIndices() const;
    const vector<array<R,d>>& GetChebyshevGrid() const;

    const vector<R>& GetLeftChebyshevMap() const;
    const vector<R>& GetRightChebyshevMap() const;
    const vector<array<R,d>>& GetSourceChildGrids() const;
};

// Implementations

template<typename R,size_t d,size_t q>
inline void 
Context<R,d,q>::GenerateChebyshevNodes()
{
    const R pi = Pi<R>();
    for( size_t t=0; t<q; ++t )
        chebyshevNodes_[t] = 0.5*cos(R(t*pi/(q-1)));
}

template<typename R,size_t d,size_t q>
inline void 
Context<R,d,q>::GenerateChebyshevIndices()
{
    const size_t q_to_d = chebyshevIndices_.size();

    for( size_t t=0; t<q_to_d; ++t )
    {
        size_t qToThej = 1;
        for( size_t j=0; j<d; ++j )
        {
            size_t i = (t/qToThej) % q;
            chebyshevIndices_[t][j] = i;
            qToThej *= q;
        }
    }
}

template<typename R,size_t d,size_t q>
inline void 
Context<R,d,q>::GenerateChebyshevGrid()
{
    const R pi = Pi<R>();
    const size_t q_to_d = chebyshevGrid_.size();

    for( size_t t=0; t<q_to_d; ++t )
    {
        size_t q_to_j = 1;
        for( size_t j=0; j<d; ++j )
        {
            size_t i = (t/q_to_j) % q;
            chebyshevGrid_[t][j] = 0.5*cos(R(i*pi/(q-1)));
            q_to_j *= q;
        }
    }
}

template<typename R,size_t d,size_t q>
inline void
Context<R,d,q>::GenerateChebyshevMaps()
{
    // Create 1d Lagrangian evaluation maps being left and right of the center
    for( size_t i=0; i<q; ++i )
        for( size_t k=0; k<q; ++k )
            leftChebyshevMap_[k*q+i] = 
                Lagrange1d( i, (2*chebyshevNodes_[k]-1)/4 );
    for( size_t i=0; i<q; ++i )
        for( size_t k=0; k<q; ++k )
            rightChebyshevMap_[k*q+i] = 
                Lagrange1d( i, (2*chebyshevNodes_[k]+1)/4 );
}

template<typename R,size_t d,size_t q>
inline void 
Context<R,d,q>::GenerateChildGrids()
{
    const size_t q_to_d = chebyshevGrid_.size();

    for( size_t c=0; c<(1u<<d); ++c )
    {
        for( size_t tPrime=0; tPrime<q_to_d; ++tPrime )
        {

            // Map p_t'(Bc) to the reference domain ([-1/2,+1/2]^d) of B
            for( size_t j=0; j<d; ++j )
            {
                sourceChildGrids_[c*q_to_d+tPrime][j] = 
                    ( (c>>j)&1 ? (2*chebyshevGrid_[tPrime][j]+1)/4 
                               : (2*chebyshevGrid_[tPrime][j]-1)/4 );
            }
        }
    }
}

template<typename R,size_t d,size_t q>
inline Context<R,d,q>::Context() 
: chebyshevNodes_( q ),
  leftChebyshevMap_( q*q ),
  rightChebyshevMap_( q*q ),
  chebyshevIndices_( Pow<q,d>::val ), 
  chebyshevGrid_( Pow<q,d>::val ),
  sourceChildGrids_( Pow<q,d>::val<<d )
{
    GenerateChebyshevNodes();
    GenerateChebyshevIndices();
    GenerateChebyshevGrid();
    GenerateChebyshevMaps();
    GenerateChildGrids();
}

template<typename R,size_t d,size_t q>
inline R
Context<R,d,q>::Lagrange1d( size_t i, R p ) const
{
    R product = R(1);
    const R* chebyshevNodeBuffer = &chebyshevNodes_[0];
    for( size_t k=0; k<q; ++k )
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

template<typename R,size_t d,size_t q>
inline R
Context<R,d,q>::Lagrange( size_t t, const array<R,d>& p ) const
{
    R product = R(1);
    const R* RESTRICT pBuffer = &p[0];
    const R* RESTRICT chebyshevNodeBuffer = &chebyshevNodes_[0];
    const size_t* RESTRICT chebyshevIndicesBuffer = &chebyshevIndices_[t][0];
    for( size_t j=0; j<d; ++j )
    {
        size_t i = chebyshevIndicesBuffer[j];
        for( size_t k=0; k<q; ++k )
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

template<typename R,size_t d,size_t q>
inline void
Context<R,d,q>::LagrangeBatch
( size_t t, 
  const vector<array<R,d>>& p, 
        vector<R         >& results ) const
{
    results.resize( p.size() );
    R* resultsBuffer = &results[0];
    for( size_t i=0; i<p.size(); ++i )
        resultsBuffer[i] = 1;

    const R* RESTRICT pBuffer = &p[0][0];
    const R* RESTRICT chebyshevNodeBuffer = &chebyshevNodes_[0];
    const size_t* RESTRICT chebyshevIndicesBuffer = &chebyshevIndices_[t][0];
    for( size_t j=0; j<d; ++j )
    {
        size_t i = chebyshevIndicesBuffer[j];
        for( size_t k=0; k<q; ++k )
        {
            if( i != k )
            {
                const R iNode = chebyshevNodeBuffer[i];
                const R kNode = chebyshevNodeBuffer[k];
                const R nodeDist = iNode - kNode;
                for( size_t r=0; r<p.size(); ++r )
                    resultsBuffer[r] *= (pBuffer[r*d+j]-kNode) / nodeDist;
            }
        }
    }
}

template<typename R,size_t d,size_t q>
inline const vector<R>&
Context<R,d,q>::GetChebyshevNodes() const
{ return chebyshevNodes_; }

template<typename R,size_t d,size_t q>
inline const vector<array<size_t,d>>&
Context<R,d,q>::GetChebyshevIndices() const
{ return chebyshevIndices_; }

template<typename R,size_t d,size_t q>
inline const vector<array<R,d>>&
Context<R,d,q>::GetChebyshevGrid() const
{ return chebyshevGrid_; }

template<typename R,size_t d,size_t q>
inline const vector<R>& 
Context<R,d,q>::GetLeftChebyshevMap() const
{ return leftChebyshevMap_; }

template<typename R,size_t d,size_t q>
inline const vector<R>& 
Context<R,d,q>::GetRightChebyshevMap() const
{ return rightChebyshevMap_; }

template<typename R,size_t d,size_t q>
inline const vector<array<R,d>>&
Context<R,d,q>::GetSourceChildGrids() const
{ return sourceChildGrids_; }

} // rfio
} // bfio

#endif // ifndef BFIO_RFIO_CONTEXT_HPP
