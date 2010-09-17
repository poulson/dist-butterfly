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
#ifndef BFIO_STRUCTURES_CONTEXT_HPP
#define BFIO_STRUCTURES_CONTEXT_HPP 1

#include <memory>
#include <vector>
#include "bfio/structures/data.hpp"

namespace bfio {

template<typename R,unsigned d,unsigned q>
class Context
{
    Array<R,q> _chebyshevNodes;
    std::vector< Array<unsigned,d> > _chebyshevIndices;
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
    R Lagrange( unsigned t, const Array<R,d>& p ) const;

    const Array<R,d>&
    GetChebyshevNodes() const;

    const std::vector< Array<unsigned,d> >&
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

template<typename R,unsigned d,unsigned q>
void Context<R,d,q>::GenerateChebyshevNodes()
{
    for( unsigned t=0; t<q; ++t )
        _chebyshevNodes[t] = 0.5*cos(t*Pi/(q-1));
}

template<typename R,unsigned d,unsigned q>
void Context<R,d,q>::GenerateChebyshevIndices()
{
    const unsigned q_to_d = _chebyshevIndices.size();

    for( unsigned t=0; t<q_to_d; ++t )
    {
        unsigned qToThej = 1;
        for( unsigned j=0; j<d; ++j )
        {
            unsigned i = (t/qToThej) % q;
            _chebyshevIndices[t][j] = i;
            qToThej *= q;
        }
    }
}

template<typename R,unsigned d,unsigned q>
void Context<R,d,q>::GenerateChebyshevGrid()
{
    const unsigned q_to_d = _chebyshevGrid.size();

    for( unsigned t=0; t<q_to_d; ++t )
    {
        unsigned qToThej = 1;
        for( unsigned j=0; j<d; ++j )
        {
            unsigned i = (t/qToThej)%q;
            _chebyshevGrid[t][j] = 0.5*cos(static_cast<R>(i*Pi/(q-1)));
            qToThej *= q;
        }
    }
}

template<typename R,unsigned d,unsigned q>
void Context<R,d,q>::GenerateFreqMapsAndChildGrids()
{
    const unsigned q_to_d = _chebyshevGrid.size();
    const unsigned q_to_2d = q_to_d * q_to_d;

    for( unsigned c=0; c<(1u<<d); ++c )
    {
        for( unsigned tPrime=0; tPrime<q_to_d; ++tPrime )
        {

            // Map p_t'(Bc) to the reference domain ([-1/2,+1/2]^d) of B
            for( unsigned j=0; j<d; ++j )
            {
                _freqChildGrids[(c*q_to_d+tPrime)*d][j] = 
                    ( (c>>j)&1 ? (2*_chebyshevGrid[tPrime][j]+1)/4 
                               : (2*_chebyshevGrid[tPrime][j]-1)/4 );
            }
        }
    }

    // Store all of the Lagrangian evaluations on p_t'(Bc)'s
    for( unsigned c=0; c<(1u<<d); ++c )
    {
        for( unsigned t=0; t<q_to_d; ++t )
        {
            for( unsigned tPrime=0; tPrime<q_to_d; ++tPrime )
            {
                _freqMaps[c*q_to_2d+tPrime*q_to_d+t] = 
                    Lagrange( t, _freqChildGrids[(c*q_to_d+tPrime)*d] ); 
            }
        }
    }
}

template<typename R,unsigned d,unsigned q>
void Context<R,d,q>::GenerateSpatialMaps()
{
    const unsigned q_to_d = _chebyshevGrid.size();
    const unsigned q_to_2d = q_to_d * q_to_d;
    for( unsigned p=0; p<(1u<<d); ++p )
    {
        for( unsigned t=0; t<q_to_d; ++t )
        {
            // Map x_t(A) to the reference domain ([-1/2,+1/2]^d) of its parent.
            Array<R,d> xtARefAp;
            for( unsigned j=0; j<d; ++j )
            {
                xtARefAp[j] = 
                    ( (p>>j)&1 ? (2*_chebyshevGrid[t][j]+1)/4 
                               : (2*_chebyshevGrid[t][j]-1)/4 ); 
            }

            for( unsigned tPrime=0; tPrime<q_to_d; ++tPrime )
            {
                _spatialMaps[p*q_to_2d + t+tPrime*q_to_d] = 
                    Lagrange( tPrime, xtARefAp );
            }
        }
    }
}

template<typename R,unsigned d,unsigned q>
Context<R,d,q>::Context() 
: _chebyshevNodes(Pow<q,d>::val), 
  _chebyshevIndices(Pow<q,d>::val), 
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

template<typename R,unsigned d,unsigned q>
R
Context<R,d,q>::Lagrange
( unsigned t, const Array<R,d>& z ) const
{
    R product = static_cast<R>(1);
    for( unsigned j=0; j<d; ++j )
    {
        unsigned i = _chebyshevIndices[t][j];
        for( unsigned k=0; k<q; ++k )
        {
            if( i != k )
            {
                product *= (z[j]-_chebyshevNodes[k]) /
                           (_chebyshevNodes[i]-_chebyshevNodes[k]);
            }
        }
    }
    return product;
}

template<typename R,unsigned d,unsigned q>
const Array<R,d>&
Context<R,d,q>::GetChebyshevNodes() const
{ return _chebyshevNodes; }

template<typename R,unsigned d,unsigned q>
const std::vector< Array<unsigned,d> >&
Context<R,d,q>::GetChebyshevIndices() const
{ return _chebyshevIndices; }

template<typename R,unsigned d,unsigned q>
const std::vector< Array<R,d> >&
Context<R,d,q>::GetChebyshevGrid() const
{ return _chebyshevGrid; }

template<typename R,unsigned d,unsigned q>
const std::vector<R>&
Context<R,d,q>::GetFreqMaps() const
{ return _freqMaps; }

template<typename R,unsigned d,unsigned q>
const std::vector<R>&
Context<R,d,q>::GetSpatialMaps() const
{ return _spatialMaps; }

template<typename R,unsigned d,unsigned q>
const std::vector< Array<R,d> >&
Context<R,d,q>::GetFreqChildGrids() const
{ return _freqChildGrids; }

} // bfio

#endif // BFIO_STRUCTURES_CONTEXT_HPP

