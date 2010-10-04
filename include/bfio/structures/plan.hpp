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
#ifndef BFIO_STRUCTURES_PLAN_HPP
#define BFIO_STRUCTURES_PLAN_HPP 1

#include <bitset>
#include <stdexcept>
#include <vector>

#include "bfio/tools/twiddle.hpp"
#include "mpi.h"

namespace bfio {

template<std::size_t d>
class Plan
{
protected:
    MPI_Comm _comm;
    const std::size_t _N;

    // Does not depend on the problem size
    int _rank;
    int _numProcesses;
    std::size_t _log2NumProcesses;
    MPI_Group _group;
    std::size_t _log2N;
    Array<std::size_t,d> _myInitialSourceBoxCoords;
    Array<std::size_t,d> _myFinalTargetBoxCoords;
    Array<std::size_t,d> _log2InitialSourceBoxesPerDim;
    Array<std::size_t,d> _log2FinalTargetBoxesPerDim;
    bool _backwardSourcePartitioning;
    bool _backwardTargetPartitioning;

    // Depends on the problem size
    std::vector<MPI_Comm> _subcomms;
    std::vector<MPI_Group> _subgroups;
    std::vector< Array<bool,d> > _sourceDimsToMerge;
    std::vector< Array<bool,d> > _targetDimsToCut;
    std::vector< Array<bool,d> > _rightSideOfCut;

    virtual void GeneratePlan() = 0;
    
public:        
    Plan( MPI_Comm comm, std::size_t N ) 
    : _comm(comm), _N(N)
    { 
        MPI_Comm_rank( comm, &_rank );
        MPI_Comm_size( comm, &_numProcesses );
        MPI_Comm_group( comm, &_group );

        if( ! IsPowerOfTwo(N) )
            throw std::runtime_error("Must use power of 2 problem size");
        if( ! IsPowerOfTwo(_numProcesses) )
            throw std::runtime_error("Must use power of 2 number of processes");
        _log2N = Log2( N );
        _log2NumProcesses = Log2( _numProcesses );
        if( _log2NumProcesses > d*_log2N )
            throw std::runtime_error("Cannot use more than N^d processes");
    }
    
    virtual ~Plan()
    {
        for( std::size_t i=0; i<_subcomms.size(); ++i )
        {
            MPI_Comm_free( &_subcomms[i] );
            MPI_Group_free( &_subgroups[i] );
        }
        MPI_Group_free( &_group );
    }

    MPI_Comm GetComm() const { return _comm; }

    std::size_t GetN() const { return _N; }

    bool BackwardSourcePartitioning() const 
    { return _backwardSourcePartitioning; }

    bool BackwardTargetPartitioning() const
    { return _backwardTargetPartitioning; }

    template<typename R>
    Box<R,d> GetMyInitialSourceBox( const Box<R,d>& sourceBox ) const
    {
        Box<R,d> myInitialSourceBox;
        for( std::size_t j=0; j<d; ++j )
        {
            myInitialSourceBox.widths[j] = 
                sourceBox.widths[j] / (1u<<_log2InitialSourceBoxesPerDim[j]);
            myInitialSourceBox.offsets[j] = 
                sourceBox.offsets[j] + 
                _myInitialSourceBoxCoords[j]*myInitialSourceBox.widths[j];
        }
        return myInitialSourceBox;
    }

    template<typename R>
    Box<R,d> GetMyFinalTargetBox( const Box<R,d>& targetBox ) const
    {
        Box<R,d> myFinalTargetBox;
        for( std::size_t j=0; j<d; ++j )
        {
            myFinalTargetBox.widths[j] = 
                targetBox.widths[j] / (1u<<_log2FinalTargetBoxesPerDim[j]);
            myFinalTargetBox.offsets[j] = 
                targetBox.offsets[j] + 
                _myFinalTargetBoxCoords[j]*myFinalTargetBox.widths[j];
        }
        return myFinalTargetBox;
    }

    const Array<std::size_t,d>& GetMyInitialSourceBoxCoords() const
    { return _myInitialSourceBoxCoords; }

    const Array<std::size_t,d>& GetMyFinalTargetBoxCoords() const
    { return _myFinalTargetBoxCoords; }

    const Array<std::size_t,d>& GetLog2InitialSourceBoxesPerDim() const
    { return _log2InitialSourceBoxesPerDim; }

    const Array<std::size_t,d>& GetLog2FinalTargetBoxesPerDim() const
    { return _log2FinalTargetBoxesPerDim; }

    MPI_Comm GetSubcommunicator( std::size_t i ) const
    { return _subcomms[i]; }

    const Array<bool,d>&
    GetSourceDimsToMerge( std::size_t i ) const
    { return _sourceDimsToMerge[i]; }

    const Array<bool,d>& 
    GetTargetDimsToCut( std::size_t i ) const
    { return _targetDimsToCut[i]; }

    const Array<bool,d>& 
    GetRightSideOfCut( std::size_t i ) const
    { return _rightSideOfCut[i]; }
};

template<std::size_t d>
class FreqToSpatialPlan : public Plan<d>
{
    virtual void GeneratePlan();
public:
    FreqToSpatialPlan
    ( MPI_Comm comm, std::size_t N );
};

template<std::size_t d>
class SpatialToFreqPlan : public Plan<d>
{
    virtual void GeneratePlan();
public:
    SpatialToFreqPlan
    ( MPI_Comm comm, std::size_t N );
};

// Implementations
template<std::size_t d>
FreqToSpatialPlan<d>::FreqToSpatialPlan
( MPI_Comm comm, std::size_t N )
: Plan<d>( comm, N )
{ GeneratePlan(); }

template<std::size_t d>
void
FreqToSpatialPlan<d>::GeneratePlan()
{
    this->_backwardSourcePartitioning = false;
    this->_backwardTargetPartitioning = true;
    std::bitset<8*sizeof(int)> rankBits(this->_rank);

    // Compute the number of source boxes per dimension and our coordinates
    for( std::size_t j=0; j<d; ++j )
    {
        this->_myInitialSourceBoxCoords[j] = 0;
        this->_log2InitialSourceBoxesPerDim[j] = 0;
    }
    std::size_t nextSourceDimToCut = 0;
    for( std::size_t m=this->_log2NumProcesses; m>0; --m )
    {
        this->_myInitialSourceBoxCoords[nextSourceDimToCut] <<= 1;
        if( rankBits[m-1] )
            ++this->_myInitialSourceBoxCoords[nextSourceDimToCut];
        ++this->_log2InitialSourceBoxesPerDim[nextSourceDimToCut];
        nextSourceDimToCut = (nextSourceDimToCut+1) % d;
    }

    // Generate subcommunicator vector by walking through the FreqToSpatial
    // process
    std::size_t numTargetCuts = 0;
    std::size_t nextTargetDimToCut = d-1;
    std::size_t log2LocalSourceBoxes = 0;
    for( std::size_t j=0; j<d; ++j )
    {
        log2LocalSourceBoxes += 
            this->_log2N-this->_log2InitialSourceBoxesPerDim[j];
        this->_myFinalTargetBoxCoords[j] = 0;
        this->_log2FinalTargetBoxesPerDim[j] = 0;
    }
    for( std::size_t level=1; level<=this->_log2N; ++level )
    {
        if( log2LocalSourceBoxes >= d )
        {
            log2LocalSourceBoxes -= d;
        }
        else
        {
            const std::size_t log2NumMergingProcesses = d-log2LocalSourceBoxes;
            const std::size_t numMergingProcesses = 1u<<log2NumMergingProcesses;
            log2LocalSourceBoxes = 0;

            // Construct the group and communicator for our current cluster
            int myClusterRank = 0;
            const std::size_t log2Stride = numTargetCuts;
            const int startRank = 
                this->_rank & ~((numMergingProcesses-1)<<log2Stride);
            std::vector<int> ranks( numMergingProcesses );
            for( std::size_t j=0; j<numMergingProcesses; ++j )
            {
                // We need to reverse the order of the last
                // log2NumMergingProcesses bits of j and add the result
                // multiplied by the stride onto the startRank
                std::size_t jReversed = 0;
                for( std::size_t k=0; k<log2NumMergingProcesses; ++k )
                    jReversed |= ((j>>k)&1)<<(log2NumMergingProcesses-1-k);
                ranks[j] = startRank+(jReversed<<log2Stride);
                if( ranks[j] == this->_rank )
                    myClusterRank = j;
            }
            MPI_Group clusterGroup;
            MPI_Group_incl
            ( this->_group, numMergingProcesses, &ranks[0], &clusterGroup );
            this->_subgroups.push_back( clusterGroup );
            MPI_Comm clusterComm;
            MPI_Comm_create( this->_comm, clusterGroup, &clusterComm );
            this->_subcomms.push_back( clusterComm );

            Array<bool,d> sourceDimsToMerge(0);
            Array<bool,d> targetDimsToCut(0);
            Array<bool,d> rightSideOfCut(0);
            for( std::size_t j=0; j<log2NumMergingProcesses; ++j )
            {
                sourceDimsToMerge[j] = true;
                targetDimsToCut[nextTargetDimToCut] = true;    
                this->_myFinalTargetBoxCoords[nextTargetDimToCut] <<= 1;
                if( rankBits[numTargetCuts] )
                {
                    rightSideOfCut[nextTargetDimToCut] = true;
                    ++this->_myFinalTargetBoxCoords[nextTargetDimToCut];
                }
                ++this->_log2FinalTargetBoxesPerDim[nextTargetDimToCut];
                ++numTargetCuts;
                nextTargetDimToCut = (nextTargetDimToCut+d-1) % d;
            }
            this->_sourceDimsToMerge.push_back( sourceDimsToMerge );
            this->_targetDimsToCut.push_back( targetDimsToCut );
            this->_rightSideOfCut.push_back( rightSideOfCut );
        }
    }
}

template<std::size_t d>
SpatialToFreqPlan<d>::SpatialToFreqPlan
( MPI_Comm comm, std::size_t N )
: Plan<d>( comm, N )
{ GeneratePlan(); }

template<std::size_t d>
void
SpatialToFreqPlan<d>::GeneratePlan()
{
    this->_backwardSourcePartitioning = true;
    this->_backwardTargetPartitioning = false;
    std::bitset<8*sizeof(int)> rankBits(this->_rank);

    if( this->_log2NumProcesses%d != 0 )
        throw std::runtime_error
              ("SpatialToFreqPlan currently requires 2^{nd} processes.");

    // Compute the number of source boxes per dimension and our coordinates
    for( std::size_t j=0; j<d; ++j )
    {
        this->_myInitialSourceBoxCoords[j] = 0;
        this->_log2InitialSourceBoxesPerDim[j] = 0;
    }
    std::size_t nextSourceDimToCut = d-1;
    for( std::size_t m=0; m<this->_log2NumProcesses; ++m )
    {
        this->_myInitialSourceBoxCoords[nextSourceDimToCut] <<= 1;
        if( rankBits[m] )
            ++this->_myInitialSourceBoxCoords[nextSourceDimToCut];
        ++this->_log2InitialSourceBoxesPerDim[nextSourceDimToCut];
        nextSourceDimToCut = (nextSourceDimToCut+d-1) % d;
    }

    // Generate subcommunicator vector by walking through the SpatialToFreq
    // process
    //
    // The following choice ensures that the first communication partitions 
    // dimensions of the form 0 -> c, and the rest are of the form 0 -> d-1.
    std::size_t numTargetCuts = 0;
    std::size_t nextTargetDimToCut = 0;
    std::size_t log2LocalSourceBoxes = 0;
    for( std::size_t j=0; j<d; ++j )
    {
        log2LocalSourceBoxes +=
            this->_log2N-this->_log2InitialSourceBoxesPerDim[j];
        this->_myFinalTargetBoxCoords[j] = 0;
        this->_log2FinalTargetBoxesPerDim[j] = 0;
    }
    for( std::size_t level=1; level<=this->_log2N; ++level )
    {
        if( log2LocalSourceBoxes >= d )
        {
            log2LocalSourceBoxes -= d;
        }
        else
        {
            const std::size_t log2NumMergingProcesses = d-log2LocalSourceBoxes;
            const std::size_t numMergingProcesses = 1u<<log2NumMergingProcesses;
            log2LocalSourceBoxes = 0;

            // Construct the group and communicator for our current cluster
            int myClusterRank = 0;
            const std::size_t log2Stride = 
                this->_log2NumProcesses-numTargetCuts-log2NumMergingProcesses;
            const int startRank = 
                this->_rank & ~((numMergingProcesses-1)<<log2Stride);
            std::vector<int> ranks( numMergingProcesses ); 
            // The bits of j must be shuffled according to the ordering 
            // on the partition dimensions:
            //  - The last d-nextTargetDimToCut bits are reversed and last
            //  - The remaining bits are reversed but occur first
            const std::size_t lastBitsetSize = 
                std::min(d-nextTargetDimToCut,log2NumMergingProcesses);
            const std::size_t firstBitsetSize = 
                log2NumMergingProcesses-lastBitsetSize;
            for( std::size_t j=0; j<numMergingProcesses; ++j )
            {
                std::size_t jWrapped = 0;
                for( std::size_t k=0; k<firstBitsetSize; ++k )
                {
                    jWrapped |= 
                        ((j>>k)&1)<<(firstBitsetSize-1-k);
                }
                for( std::size_t k=0; k<lastBitsetSize; ++k )
                {
                    jWrapped |=
                        ((j>>(k+firstBitsetSize))&1)<<
                        (lastBitsetSize-1-k+firstBitsetSize);
                }
                ranks[j] = startRank + (jWrapped<<log2Stride);
                if( ranks[j] == this->_rank )
                    myClusterRank = j;
            }
            MPI_Group clusterGroup;
            MPI_Group_incl
            ( this->_group, numMergingProcesses, &ranks[0], &clusterGroup );
            this->_subgroups.push_back( clusterGroup );
            MPI_Comm clusterComm;
            MPI_Comm_create( this->_comm, clusterGroup, &clusterComm );
            this->_subcomms.push_back( clusterComm );

            Array<bool,d> sourceDimsToMerge(0);
            Array<bool,d> targetDimsToCut(0);
            Array<bool,d> rightSideOfCut(0);
            for( std::size_t j=0; j<log2NumMergingProcesses; ++j )
            {
                sourceDimsToMerge[j] = true;
                targetDimsToCut[nextTargetDimToCut] = true;
                this->_myFinalTargetBoxCoords[nextTargetDimToCut] <<= 1;
                //if( rankBits[numTargetCuts] )
                if( rankBits[this->_log2NumProcesses-1-numTargetCuts] )
                {
                    rightSideOfCut[nextTargetDimToCut] = true;
                    ++this->_myFinalTargetBoxCoords[nextTargetDimToCut];
                }
                ++this->_log2FinalTargetBoxesPerDim[nextTargetDimToCut];
                ++numTargetCuts;
                nextTargetDimToCut = (nextTargetDimToCut+1) % d;
            }
            this->_sourceDimsToMerge.push_back( sourceDimsToMerge );
            this->_targetDimsToCut.push_back( targetDimsToCut );
            this->_rightSideOfCut.push_back( rightSideOfCut );
        }
    }
}

} // bfio

#endif // BFIO_STRUCTURES_PLAN_HPP

