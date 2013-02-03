/*
   Copyright (C) 2010-2013 Jack Poulson and Lexing Ying
 
   This file is part of ButterflyFIO and is under the GNU General Public 
   License, which can be found in the LICENSE file in the root directory, or at
   <http://www.gnu.org/licenses/>.
*/
#pragma once
#ifndef BFIO_STRUCTURES_PLAN_HPP
#define BFIO_STRUCTURES_PLAN_HPP

#ifndef RELEASE
# include <iostream>
#endif

#include <array>
#include <bitset>
#include <stdexcept>
#include <vector>

#include "bfio/constants.hpp"
#include "bfio/tools/twiddle.hpp"
#include "mpi.h"
#ifdef BGP
# include "mpix.h"
#endif

namespace bfio {

using std::size_t;
using std::vector;

template<size_t d>
class PlanBase
{
protected:
    MPI_Comm _comm;
    const Direction _direction;
    const size_t _N;

    // Does not depend on the problem size
    int _rank;
    int _numProcesses;
    size_t _log2NumProcesses;
    size_t _log2N;
    array<size_t,d> _myInitialSBoxCoords;
    array<size_t,d> _myFinalTBoxCoords;
    array<size_t,d> _log2InitialSBoxesPerDim;
    array<size_t,d> _log2FinalTBoxesPerDim;

    // Depends on the problem size
    const size_t _bootstrapSkip;
    MPI_Comm _bootstrapClusterComm;
    vector<size_t> _bootstrapSDimsToMerge;
    vector<size_t> _bootstrapTDimsToCut;
    vector<bool> _bootstrapRightSideOfCut;
    vector<MPI_Comm> _clusterComms;
    vector<size_t> _log2SubclusterSizes;
    vector<vector<size_t>> _sDimsToMerge;
    vector<vector<size_t>> _tDimsToCut;
    vector<vector<bool>> _rightSideOfCut;

    PlanBase
    ( MPI_Comm comm, Direction direction, size_t N, size_t bootstrapSkip );

public:        
    virtual ~PlanBase();

    virtual size_t 
    LocalToClusterSourceIndex( size_t level, size_t cLocal ) const = 0;

    virtual size_t
    LocalToBootstrapClusterSourceIndex( size_t cLocal ) const = 0;

    MPI_Comm GetComm() const;
    Direction GetDirection() const;
    size_t GetN() const;
    size_t GetBootstrapSkip() const;

    template<typename R>
    Box<R,d> GetMyInitialSourceBox( const Box<R,d>& sBox ) const;

    template<typename R>
    Box<R,d> GetMyFinalTargetBox( const Box<R,d>& tBox ) const;

    const array<size_t,d>& GetMyInitialSourceBoxCoords() const;
    const array<size_t,d>& GetMyFinalTargetBoxCoords() const;
    const array<size_t,d>& GetLog2InitialSourceBoxesPerDim() const;
    const array<size_t,d>& GetLog2FinalTargetBoxesPerDim() const;

    MPI_Comm GetBootstrapClusterComm() const;
    MPI_Comm GetClusterComm( size_t level ) const;
    size_t GetLog2SubclusterSize( size_t level ) const;
    size_t GetLog2NumMergingProcesses( size_t level ) const;

    const vector<size_t>& GetBootstrapSourceDimsToMerge() const;
    const vector<size_t>& GetBootstrapTargetDimsToCut() const;
    const vector<bool>&   GetBootstrapRightSideOfCut() const;
    const vector<size_t>& GetSourceDimsToMerge( size_t level ) const;
    const vector<size_t>& GetTargetDimsToCut( size_t level ) const;
    const vector<bool>&   GetRightSideOfCut( size_t level ) const;
};

template<size_t d>
class Plan : public PlanBase<d>
{
    //---------------------------------//
    // For forward plans               //
    //---------------------------------//
    size_t _myBootstrapClusterRank;
    vector<size_t> _myClusterRanks;
    void GenerateForwardPlan();

    size_t
    ForwardLocalToBootstrapClusterSourceIndex( size_t cLocal ) const;

    size_t 
    ForwardLocalToClusterSourceIndex( size_t level, size_t cLocal ) const;

    //---------------------------------//
    // For adjoint plans               //
    //---------------------------------//
    size_t _myBootstrapMappedRank;
    vector<size_t> _myMappedRanks;
    void GenerateAdjointPlan();

    size_t
    AdjointLocalToBootstrapClusterSourceIndex( size_t cLocal ) const;

    size_t 
    AdjointLocalToClusterSourceIndex( size_t level, size_t cLocal ) const;

public:
    Plan
    ( MPI_Comm comm, Direction direction, size_t N, size_t bootstrapSkip=0 );

    virtual size_t
    LocalToBootstrapClusterSourceIndex( size_t cLocal ) const;

    virtual size_t 
    LocalToClusterSourceIndex( size_t level, size_t cLocal ) const;
};

// Implementations
    
template<size_t d>
PlanBase<d>::PlanBase
( MPI_Comm comm, Direction direction, size_t N, size_t bootstrapSkip ) 
: _comm(comm), _direction(direction), _N(N), _bootstrapSkip(bootstrapSkip)
{ 
    MPI_Comm_rank( comm, &_rank );
    MPI_Comm_size( comm, &_numProcesses );

    if( ! IsPowerOfTwo(N) )
        throw std::runtime_error("Must use power of 2 problem size");
    if( ! IsPowerOfTwo(_numProcesses) )
        throw std::runtime_error("Must use power of 2 number of processes");
    _log2N = Log2( N );
    _log2NumProcesses = Log2( _numProcesses );
    if( _log2NumProcesses > d*_log2N )
        throw std::runtime_error("Cannot use more than N^d processes");
    if( bootstrapSkip > _log2N/2 )
        throw std::runtime_error("Cannot bootstrap past the middle switch");

    _clusterComms.resize( _log2N );
    _log2SubclusterSizes.resize( _log2N );
    _sDimsToMerge.resize( _log2N );
    _tDimsToCut.resize( _log2N );
    _rightSideOfCut.resize( _log2N );
}

template<size_t d>
PlanBase<d>::~PlanBase()
{
    MPI_Comm_free( &_bootstrapClusterComm );
    for( size_t level=1; level<=_log2N; ++level )
        MPI_Comm_free( &_clusterComms[level-1] );
}

template<size_t d>
inline MPI_Comm PlanBase<d>::GetComm() const 
{ return _comm; }

template<size_t d>
inline Direction
PlanBase<d>::GetDirection() const
{ return _direction; }

template<size_t d>
inline size_t 
PlanBase<d>::GetN() const 
{ return _N; }

template<size_t d>
inline size_t
PlanBase<d>::GetBootstrapSkip() const
{ return _bootstrapSkip; }

template<size_t d> 
template<typename R>
Box<R,d> 
PlanBase<d>::GetMyInitialSourceBox( const Box<R,d>& sBox ) const
{
    Box<R,d> myInitialSBox;
    for( size_t j=0; j<d; ++j )
    {
        myInitialSBox.widths[j] = 
            sBox.widths[j] / (1u<<_log2InitialSBoxesPerDim[j]);
        myInitialSBox.offsets[j] = 
            sBox.offsets[j] + _myInitialSBoxCoords[j]*myInitialSBox.widths[j];
    }
    return myInitialSBox;
}

template<size_t d> 
template<typename R>
Box<R,d> 
PlanBase<d>::GetMyFinalTargetBox( const Box<R,d>& tBox ) const
{
    Box<R,d> myFinalTBox;
    for( size_t j=0; j<d; ++j )
    {
        myFinalTBox.widths[j] = 
            tBox.widths[j] / (1u<<_log2FinalTBoxesPerDim[j]);
        myFinalTBox.offsets[j] = 
            tBox.offsets[j] + _myFinalTBoxCoords[j]*myFinalTBox.widths[j];
    }
    return myFinalTBox;
}

template<size_t d>
inline const array<size_t,d>& 
PlanBase<d>::GetMyInitialSourceBoxCoords() const
{ return _myInitialSBoxCoords; }

template<size_t d>
inline const array<size_t,d>& 
PlanBase<d>::GetMyFinalTargetBoxCoords() const
{ return _myFinalTBoxCoords; }

template<size_t d>
inline const array<size_t,d>& 
PlanBase<d>::GetLog2InitialSourceBoxesPerDim() const
{ return _log2InitialSBoxesPerDim; }

template<size_t d>
inline const array<size_t,d>& 
PlanBase<d>::GetLog2FinalTargetBoxesPerDim() const
{ return _log2FinalTBoxesPerDim; }

template<size_t d>
inline MPI_Comm 
PlanBase<d>::GetClusterComm( size_t level ) const
{ return _clusterComms[level-1]; }

template<size_t d>
inline MPI_Comm
PlanBase<d>::GetBootstrapClusterComm() const
{ return _bootstrapClusterComm; }

template<size_t d>
inline const vector<size_t>&
PlanBase<d>::GetBootstrapSourceDimsToMerge() const
{ return _bootstrapSDimsToMerge; }

template<size_t d>
inline const vector<size_t>&
PlanBase<d>::GetBootstrapTargetDimsToCut() const
{ return _bootstrapTDimsToCut; }

template<size_t d>
inline const vector<bool>& 
PlanBase<d>::GetBootstrapRightSideOfCut() const
{ return _bootstrapRightSideOfCut; }

template<size_t d>
inline size_t 
PlanBase<d>::GetLog2SubclusterSize( size_t level ) const
{ return _log2SubclusterSizes[level-1]; }

template<size_t d>
inline size_t
PlanBase<d>::GetLog2NumMergingProcesses( size_t level ) const
{ return _sDimsToMerge[level-1].size(); }

template<size_t d>
inline const vector<size_t>&
PlanBase<d>::GetSourceDimsToMerge( size_t level ) const
{ return _sDimsToMerge[level-1]; }

template<size_t d>
inline const vector<size_t>& 
PlanBase<d>::GetTargetDimsToCut( size_t level ) const
{ return _tDimsToCut[level-1]; }

template<size_t d>
inline const vector<bool>& 
PlanBase<d>::GetRightSideOfCut( size_t level ) const
{ return _rightSideOfCut[level-1]; }

template<size_t d>
Plan<d>::Plan
( MPI_Comm comm, Direction direction, size_t N, size_t bootstrapSkip )
: PlanBase<d>( comm, direction, N, bootstrapSkip )
{ 

    if( direction == FORWARD )
        GenerateForwardPlan();
    else
        GenerateAdjointPlan();
}

template<size_t d>
inline size_t
Plan<d>::ForwardLocalToBootstrapClusterSourceIndex( size_t cLocal ) const
{
    return (cLocal<<this->_bootstrapSDimsToMerge.size()) +
           this->_myBootstrapClusterRank;
}

template<size_t d>
inline size_t
Plan<d>::ForwardLocalToClusterSourceIndex( size_t level, size_t cLocal ) const
{
    return (cLocal<<this->_sDimsToMerge[level-1].size()) +     
           this->_myClusterRanks[level-1];
}

template<size_t d>
void
Plan<d>::GenerateForwardPlan()
{
    std::bitset<8*sizeof(int)> rankBits(this->_rank);
        
    _myClusterRanks.resize( this->_log2N );

    // Compute the number of source boxes per dimension and our coordinates
    size_t nextSDimToCut = 0;
    size_t lastSDimCut = 0; // initialize to avoid compiler warnings
    for( size_t j=0; j<d; ++j )
    {
        this->_myInitialSBoxCoords[j] = 0;
        this->_log2InitialSBoxesPerDim[j] = 0;
    }
    for( size_t m=this->_log2NumProcesses; m>0; --m )
    {
#ifndef RELEASE
        if( this->_rank == 0 )
        {
            std::cout << "Cutting source dimension " << nextSDimToCut
                      << std::endl;
        }
#endif
        lastSDimCut = nextSDimToCut;
        this->_myInitialSBoxCoords[nextSDimToCut] <<= 1;
        if( rankBits[m-1] )
            ++this->_myInitialSBoxCoords[nextSDimToCut];
        ++this->_log2InitialSBoxesPerDim[nextSDimToCut];
        nextSDimToCut = (nextSDimToCut+1) % d;
    }
#ifndef RELEASE
    for( int p=0; p<this->_numProcesses; ++p )
    {
        if( this->_rank == p )
        {
            std::cout << "Rank " << p << "'s initial source box coords: ";
            for( size_t j=0; j<d; ++j )
                std::cout << this->_myInitialSBoxCoords[j] << " ";
            std::cout << std::endl;
        }
        MPI_Barrier( this->_comm );
        usleep( 100000 );
    }
#endif

    // Generate subcommunicator vector by walking through the forward process
    size_t numTCuts = 0;
    size_t nextTDimToCut = d-1;
    size_t nextSDimToMerge = lastSDimCut;
    size_t log2LocalSBoxes = 0;
    for( size_t j=0; j<d; ++j )
    {
        log2LocalSBoxes += this->_log2N-this->_log2InitialSBoxesPerDim[j];
        this->_myFinalTBoxCoords[j] = 0;
        this->_log2FinalTBoxesPerDim[j] = 0;
    }
    // Generate the bootstrap communicator
    if( log2LocalSBoxes >= d*this->_bootstrapSkip )
    {
        this->_myBootstrapClusterRank = 0; 

        MPI_Comm_split
        ( this->_comm, this->_rank, 0, &this->_bootstrapClusterComm );

#ifndef RELEASE
        if( this->_rank == 0 )
            std::cout << "No communication during bootstrapping." << std::endl;
#endif
    }
    else
    {
        const size_t log2NumMergingProcesses = 
            this->_bootstrapSkip*d - log2LocalSBoxes;
        const size_t numMergingProcesses = 1u<<log2NumMergingProcesses;

#ifndef RELEASE
        if( this->_rank == 0 )
        {
            std::cout << "Merging " << log2NumMergingProcesses
                      << " dimension(s) during bootstrapping, starting with "
                      << nextSDimToMerge << " (cutting starting with "
                      << nextTDimToCut << ")" << std::endl;
        }
#endif

        // Construct the communicator for the bootstrap cluster
        const int startRank = this->_rank & ~(numMergingProcesses-1);
        vector<int> ranks( numMergingProcesses );
        for( size_t j=0; j<numMergingProcesses; ++j )
        {
            // We need to reverse the order of the last log2NumMergingProcesses
            // bits of j and add the result onto the startRank
            size_t jReversed = 0;
            for( size_t k=0; k<log2NumMergingProcesses; ++k )
                jReversed |= ((j>>k)&1)<<(log2NumMergingProcesses-1-k);
            ranks[j] = startRank + jReversed;
            if( this->_rank == ranks[j] )
                this->_myBootstrapClusterRank = j;
        }

#ifndef RELEASE
        for( int p=0; p<this->_numProcesses; ++p )        
        {
            if( this->_rank == p )
            {
                std::cout << "  process " << p 
                          << "'s bootstrap cluster ranks: ";
                for( size_t j=0; j<numMergingProcesses; ++j )
                    std::cout << ranks[j] << " ";
                std::cout << std::endl;
            }
            MPI_Barrier( this->_comm );
            usleep( 100000 );
        }
#endif

        MPI_Comm_split
        ( this->_comm, ranks[0], this->_myBootstrapClusterRank, 
          &this->_bootstrapClusterComm );

        this->_bootstrapSDimsToMerge.resize( log2NumMergingProcesses );
        this->_bootstrapTDimsToCut.resize( log2NumMergingProcesses );
        this->_bootstrapRightSideOfCut.resize( log2NumMergingProcesses );
        size_t nextBootstrapSDimToMerge = nextSDimToMerge;
        size_t nextBootstrapTDimToCut = nextTDimToCut;
        for( size_t j=0; j<log2NumMergingProcesses; ++j )
        {
            this->_bootstrapSDimsToMerge[j] = nextBootstrapSDimToMerge;
            this->_bootstrapTDimsToCut[j] = nextBootstrapTDimToCut;
            this->_bootstrapRightSideOfCut[j] = rankBits[j];

            nextBootstrapTDimToCut = (nextBootstrapTDimToCut+d-1) % d;
            nextBootstrapSDimToMerge = (nextBootstrapSDimToMerge+d-1) % d;
        }
#ifndef RELEASE
        for( int p=0; p<this->_numProcesses; ++p )
        {
            if( this->_rank == p )
            {
                std::cout << "  process " << p << "'s bootstrap cluster rank: "
                          << this->_myBootstrapClusterRank << std::endl;
                std::cout << "  process " << p 
                          << "'s bootstrap cluster children: ";
                const size_t numLocalChildren =
                    (1u<<(this->_bootstrapSkip*d-log2NumMergingProcesses));
                for( size_t i=0; i<numLocalChildren; ++i )
                    std::cout << this->LocalToBootstrapClusterSourceIndex( i )
                              << " ";
                std::cout << std::endl;
            }
            MPI_Barrier( this->_comm );
            usleep( 100000 );
        }
#endif
    }
    // Generate the single-level communicators
    for( size_t level=1; level<=this->_log2N; ++level )
    {
        if( log2LocalSBoxes >= d )
        {
            log2LocalSBoxes -= d;

            this->_myClusterRanks[level-1] = 0;
            
            MPI_Comm_split
            ( this->_comm, this->_rank, 0, &this->_clusterComms[level-1] );
            this->_log2SubclusterSizes[level-1] = 0;

#ifndef RELEASE
            if( this->_rank == 0 )
            {
                std::cout << "No communication at level " << level
                          << ", there are now 2^" << log2LocalSBoxes
                          << " local source boxes." << std::endl;
            }
#endif
        }
        else
        {
            const size_t log2NumMergingProcesses = d-log2LocalSBoxes;
            const size_t numMergingProcesses = 1u<<log2NumMergingProcesses;
            log2LocalSBoxes = 0;

#ifndef RELEASE
            if( this->_rank == 0 )
            {
                std::cout << "Merging " << log2NumMergingProcesses
                          << " dimension(s), starting with "
                          << nextSDimToMerge << " (cutting starting with "
                          << nextTDimToCut << ")" << std::endl;
            }
#endif

            // Construct the communicator for our current cluster
            const size_t log2Stride = numTCuts;
            const int startRank = 
                this->_rank & ~((numMergingProcesses-1)<<log2Stride);
            vector<int> ranks( numMergingProcesses );
            for( size_t j=0; j<numMergingProcesses; ++j )
            {
                // We need to reverse the order of the last
                // log2NumMergingProcesses bits of j and add the result
                // multiplied by the stride onto the startRank
                size_t jReversed = 0;
                for( size_t k=0; k<log2NumMergingProcesses; ++k )
                    jReversed |= ((j>>k)&1)<<(log2NumMergingProcesses-1-k);
                ranks[j] = startRank+(jReversed<<log2Stride);
                if( this->_rank == ranks[j] )
                    this->_myClusterRanks[level-1] = j;
            }
#ifndef RELEASE
            for( int p=0; p<this->_numProcesses; ++p )
            {
                if( this->_rank == p )
                {
                    std::cout << "  process " << p << "'s cluster ranks: ";
                    for( size_t j=0; j<numMergingProcesses; ++j )
                        std::cout << ranks[j] << " ";
                    std::cout << std::endl;
                }
                MPI_Barrier( this->_comm );
                usleep( 100000 );
            }
#endif
            MPI_Comm_split
            ( this->_comm, ranks[0], this->_myClusterRanks[level-1],
              &this->_clusterComms[level-1] );

#ifdef BGP
# ifdef BGP_MPIDO_USE_REDUCESCATTER
            MPIX_Set_property
            ( this->_clusterComms[level-1], MPIDO_USE_REDUCESCATTER, 1 );
# else
            MPIX_Set_property
            ( this->_clusterComms[level-1], MPIDO_USE_REDUCESCATTER, 0 );
# endif
#endif

            this->_log2SubclusterSizes[level-1] = 0;

            this->_sDimsToMerge[level-1].resize( log2NumMergingProcesses );
            this->_tDimsToCut[level-1].resize( log2NumMergingProcesses );
            this->_rightSideOfCut[level-1].resize( log2NumMergingProcesses );
            for( size_t j=0; j<log2NumMergingProcesses; ++j )
            {
                const size_t thisBit = numTCuts;

                this->_sDimsToMerge[level-1][j] = nextSDimToMerge;
                this->_tDimsToCut[level-1][j] = nextTDimToCut;
                this->_rightSideOfCut[level-1][j] = rankBits[thisBit];

                this->_myFinalTBoxCoords[nextTDimToCut] <<= 1;
                if( rankBits[thisBit] )
                    ++this->_myFinalTBoxCoords[nextTDimToCut];
                ++this->_log2FinalTBoxesPerDim[nextTDimToCut];

                ++numTCuts;
                nextTDimToCut = (nextTDimToCut+d-1) % d;
                nextSDimToMerge = (nextSDimToMerge+d-1) % d;
            }
#ifndef RELEASE
            for( int p=0; p<this->_numProcesses; ++p )
            {
                if( this->_rank == p )
                {
                    std::cout << "  process " << p << "'s cluster rank: "
                              << this->_myClusterRanks[level-1] << std::endl;
                    std::cout << "  process " << p << "'s cluster children: ";
                    const size_t numLocalChildren =
                        (1u<<(d-log2NumMergingProcesses));
                    for( size_t i=0; i<numLocalChildren; ++i )
                        std::cout << this->LocalToClusterSourceIndex( level, i )
                                  << " ";
                    std::cout << std::endl;
                }
                MPI_Barrier( this->_comm );
                usleep( 100000 );
            }
#endif
        }
    }
}

template<size_t d>
inline size_t
Plan<d>::AdjointLocalToBootstrapClusterSourceIndex( size_t cLocal ) const
{
    return (this->_myBootstrapMappedRank<<
            (this->_bootstrapSkip*d-this->_bootstrapSDimsToMerge.size()))
           + cLocal;
}

template<size_t d>
inline size_t
Plan<d>::AdjointLocalToClusterSourceIndex( size_t level, size_t cLocal ) const
{
    return (this->_myMappedRanks[level-1]<<
            (d-this->_sDimsToMerge[level-1].size())) + cLocal;
}

template<size_t d>
void
Plan<d>::GenerateAdjointPlan()
{
    std::bitset<8*sizeof(int)> rankBits(this->_rank);
    _myMappedRanks.resize( this->_log2N );

    // Compute the number of source boxes per dimension and our coordinates
    size_t nextSDimToCut = d-1;
    size_t lastSDimCut = 0; // initialize to avoid compiler warnings
    for( size_t j=0; j<d; ++j )
    {
        this->_myInitialSBoxCoords[j] = 0;
        this->_log2InitialSBoxesPerDim[j] = 0;
    }
    for( size_t m=0; m<this->_log2NumProcesses; ++m )
    {
#ifndef RELEASE
        if( this->_rank == 0 )
        {
            std::cout << "Cutting source dimension " << nextSDimToCut
                      << std::endl;
        }
#endif
        lastSDimCut = nextSDimToCut;
        this->_myInitialSBoxCoords[nextSDimToCut] <<= 1;
        if( rankBits[m] )
            ++this->_myInitialSBoxCoords[nextSDimToCut];
        ++this->_log2InitialSBoxesPerDim[nextSDimToCut];
        nextSDimToCut = (nextSDimToCut+d-1) % d;
    }
#ifndef RELEASE
    for( int p=0; p<this->_numProcesses; ++p )
    {
        if( this->_rank == p )
        {
            std::cout << "Rank " << p << "'s initial source box coords: ";
            for( size_t j=0; j<d; ++j )
                std::cout << this->_myInitialSBoxCoords[j] << " ";
            std::cout << std::endl;
        }
        MPI_Barrier( this->_comm );
        usleep( 100000 );
    }
#endif

    // Generate subcommunicator vector by walking through the inverse process
    //
    // The following choice ensures that the first communication partitions 
    // dimensions of the form 0 -> c, and the rest are of the form 0 -> d-1.
    size_t numTCuts = 0;
    size_t nextTDimToCut = 0;
    size_t nextSDimToMerge = lastSDimCut;
    size_t log2LocalSBoxes = 0;
    for( size_t j=0; j<d; ++j )
    {
        log2LocalSBoxes += this->_log2N-this->_log2InitialSBoxesPerDim[j];
        this->_myFinalTBoxCoords[j] = 0;
        this->_log2FinalTBoxesPerDim[j] = 0;
    }
    // Generate the bootstrap communicator
    if( log2LocalSBoxes >= d*this->_bootstrapSkip )
    {
        this->_myBootstrapMappedRank = 0;

        MPI_Comm_split
        ( this->_comm, this->_rank, 0, &this->_bootstrapClusterComm );

#ifndef RELEASE
        if( this->_rank == 0 )
            std::cout << "No communication during bootstrapping." << std::endl;
#endif
    }
    else
    {
        const size_t log2NumMergingProcesses =
            this->_bootstrapSkip*d - log2LocalSBoxes;
        const size_t numMergingProcesses = 1u<<log2NumMergingProcesses;

#ifndef RELEASE
        if( this->_rank == 0 )
        {
            std::cout << "Merging " << log2NumMergingProcesses
                      << " dimension(s) during bootstrapping, starting with "
                      << nextSDimToMerge << " (cutting starting with "
                      << nextTDimToCut << ")" << std::endl;
        }
#endif

        // Construct the communicator for the bootstrap cluster
        const size_t log2Stride = 
            this->_log2NumProcesses-log2NumMergingProcesses;
        const int startRank = 
            this->_rank & ~((numMergingProcesses-1)<<log2Stride);
        vector<int> ranks( numMergingProcesses );
        for( size_t j=0; j<numMergingProcesses; ++j )
        {
            // We need to reverse the order of the last log2NumMergingProcesses
            // bits of j and add the result multiplied by the stride onto the 
            // startRank
            size_t jReversed = 0;
            for( size_t k=0; k<log2NumMergingProcesses; ++k )
                jReversed |= ((j>>k)&1)<<(log2NumMergingProcesses-1-k);
            ranks[j] = startRank + (jReversed<<log2Stride);
            if( this->_rank == ranks[j] )
                this->_myBootstrapMappedRank = j;
        }

#ifndef RELEASE
        for( int p=0; p<this->_numProcesses; ++p )
        {
            if( this->_rank == p )
            {
                std::cout << "  process " << p
                          << "'s bootstrap cluster ranks: ";
                for( size_t j=0; j<numMergingProcesses; ++j )
                    std::cout << ranks[j] << " ";
                std::cout << std::endl;
            }
            MPI_Barrier( this->_comm );
            usleep( 100000 );
        }
#endif

        MPI_Comm_split
        ( this->_comm, ranks[0], this->_myBootstrapMappedRank, 
          &this->_bootstrapClusterComm );

        this->_bootstrapSDimsToMerge.resize( log2NumMergingProcesses );
        this->_bootstrapTDimsToCut.resize( log2NumMergingProcesses );
        this->_bootstrapRightSideOfCut.resize( log2NumMergingProcesses );
        size_t nextBootstrapSDimToMerge = nextSDimToMerge;
        size_t nextBootstrapTDimToCut = nextTDimToCut;
        for( size_t j=0; j<log2NumMergingProcesses; ++j )
        {
            const size_t thisBit = (this->_log2NumProcesses-1)-j;

            this->_bootstrapSDimsToMerge[j] = nextBootstrapSDimToMerge;
            this->_bootstrapTDimsToCut[j] = nextBootstrapTDimToCut;
            this->_bootstrapRightSideOfCut[j] = rankBits[thisBit];

            nextBootstrapTDimToCut = (nextBootstrapTDimToCut+1) % d;
            nextBootstrapSDimToMerge = (nextBootstrapSDimToMerge+1) % d;
        }
#ifndef RELEASE
        for( int p=0; p<this->_numProcesses; ++p )
        {
            if( this->_rank == p )
            {
                std::cout << "  process " << p << "'s bootstrap cluster rank: "
                          << this->_myBootstrapClusterRank << std::endl;
                std::cout << "  process " << p
                          << "'s bootstrap cluster children: ";
                const size_t numLocalChildren =
                    (1u<<(this->_bootstrapSkip*d-log2NumMergingProcesses));
                for( size_t i=0; i<numLocalChildren; ++i )
                    std::cout << this->LocalToBootstrapClusterSourceIndex( i )
                              << " ";
                std::cout << std::endl;
            }
            MPI_Barrier( this->_comm );
            usleep( 100000 );
        }
#endif
    }
    // Generate the single-level communicators
    for( size_t level=1; level<=this->_log2N; ++level )
    {
        if( log2LocalSBoxes >= d )
        {

            log2LocalSBoxes -= d;

            this->_myMappedRanks[level-1] = 0;

            MPI_Comm_split
            ( this->_comm, this->_rank, 0, &this->_clusterComms[level-1] );

            this->_log2SubclusterSizes[level-1] =  0;

#ifndef RELEASE
            if( this->_rank == 0 )
            {
                std::cout << "No communication at level " << level 
                          << ", there are now 2^" << log2LocalSBoxes
                          << " local source boxes." << std::endl;
            }
#endif
        }
        else
        {
            const size_t log2NumMergingProcesses = d-log2LocalSBoxes;
            const size_t numMergingProcesses = 1u<<log2NumMergingProcesses;
            log2LocalSBoxes = 0;

#ifndef RELEASE
            if( this->_rank == 0 )
            {
                std::cout << "Merging " << log2NumMergingProcesses 
                          << " dimension(s), starting with " 
                          << nextSDimToMerge << " (cutting starting with "
                          << nextTDimToCut << ")" << std::endl;
            }
#endif

            // Construct the communicator for our current cluster
            const size_t log2Stride = 
                this->_log2NumProcesses-numTCuts-log2NumMergingProcesses;
            const int startRank = 
                this->_rank & ~((numMergingProcesses-1)<<log2Stride);
            vector<int> ranks( numMergingProcesses ); 
            {
                // The bits of j must be shuffled according to the ordering 
                // on the partition dimensions:
                //  - The last d-nextTDimToCut bits are reversed and last
                //  - The remaining bits are reversed but occur first
                const size_t lastBitsetSize = 
                    std::min(d-nextTDimToCut,log2NumMergingProcesses);
                const size_t firstBitsetSize = 
                    log2NumMergingProcesses-lastBitsetSize;
                for( size_t j=0; j<numMergingProcesses; ++j )
                {
                    size_t jWrapped = 0;
                    for( size_t k=0; k<firstBitsetSize; ++k )
                    {
                        jWrapped |= 
                            ((j>>k)&1)<<(firstBitsetSize-1-k);
                    }
                    for( size_t k=0; k<lastBitsetSize; ++k )
                    {
                        jWrapped |=
                            ((j>>(k+firstBitsetSize))&1)<<
                            (lastBitsetSize-1-k+firstBitsetSize);
                    }
                    ranks[j] = startRank + (jWrapped<<log2Stride);
                }
            }
#ifndef RELEASE
            for( int p=0; p<this->_numProcesses; ++p )
            {
                if( this->_rank == p )
                {
                    std::cout << "  process " << p << "'s cluster ranks: ";
                    for( size_t j=0; j<numMergingProcesses; ++j )
                        std::cout << ranks[j] << " ";
                    std::cout << std::endl;
                }
                MPI_Barrier( this->_comm );
                usleep( 100000 );
            }
#endif

            {
                const size_t lastBitsetSize = 
                    std::min(d-nextSDimToMerge,log2NumMergingProcesses);
                const size_t firstBitsetSize = 
                    log2NumMergingProcesses-lastBitsetSize;
                // Apply the same transformation to our shifted rank
                const size_t shiftedRank = this->_rank-startRank;
                const size_t scaledRank = shiftedRank >> log2Stride;
                size_t wrappedRank = 0;
                for( size_t k=0; k<firstBitsetSize; ++k )
                {
                    wrappedRank |=
                        ((scaledRank>>k)&1)<<(firstBitsetSize-1-k);
                }
                for( size_t k=0; k<lastBitsetSize; ++k )
                {
                    wrappedRank |=
                        ((scaledRank>>(k+firstBitsetSize))&1)<<
                         (lastBitsetSize-1-k+firstBitsetSize);
                }
                this->_myMappedRanks[level-1] = wrappedRank;
            }

            MPI_Comm_split
            ( this->_comm, ranks[0], this->_myMappedRanks[level-1],
              &this->_clusterComms[level-1] );
#ifdef BGP
# ifdef BGP_MPIDO_USE_REDUCESCATTER
            MPIX_Set_property
            ( this->_clusterComms[level-1], MPIDO_USE_REDUCESCATTER, 1 );
# else
            MPIX_Set_property
            ( this->_clusterComms[level-1], MPIDO_USE_REDUCESCATTER, 0 );
# endif
#endif

            this->_log2SubclusterSizes[level-1] = 
                this->_log2NumProcesses % d;

            this->_sDimsToMerge[level-1].resize( log2NumMergingProcesses );
            this->_tDimsToCut[level-1].resize( log2NumMergingProcesses );
            this->_rightSideOfCut[level-1].resize( log2NumMergingProcesses );
            for( size_t j=0; j<log2NumMergingProcesses; ++j )
            {
                const size_t thisBit = 
                    (this->_log2NumProcesses-1) - numTCuts;

                this->_sDimsToMerge[level-1][j] =  nextSDimToMerge;
                this->_tDimsToCut[level-1][j] = nextTDimToCut;
                this->_rightSideOfCut[level-1][j] =  rankBits[thisBit];

                this->_myFinalTBoxCoords[nextTDimToCut] <<= 1;
                if( rankBits[thisBit] )
                    ++this->_myFinalTBoxCoords[nextTDimToCut];
                ++this->_log2FinalTBoxesPerDim[nextTDimToCut];

                ++numTCuts;
                nextTDimToCut = (nextTDimToCut+1) % d;
                nextSDimToMerge = (nextSDimToMerge+1) % d;
            }
            
#ifndef RELEASE
            for( int p=0; p<this->_numProcesses; ++p )
            {
                if( this->_rank == p )
                {
                    std::cout << "  process " << p << "'s mapped rank: "
                              << this->_myMappedRanks[level-1] << std::endl;
                    std::cout << "  process " << p << "'s cluster children: ";
                    const size_t numLocalChildren = 
                        (1u<<(d-log2NumMergingProcesses));
                    for( size_t i=0; i<numLocalChildren; ++i ) 
                        std::cout << this->LocalToClusterSourceIndex( level, i )
                                  << " ";
                    std::cout << std::endl;
                }
                MPI_Barrier( this->_comm );
                usleep( 100000 );
            }
#endif
        }
    }
}

template<size_t d>
inline size_t 
Plan<d>::LocalToClusterSourceIndex( size_t level, size_t cLocal ) const
{
    if( this->_direction == FORWARD )
        return this->ForwardLocalToClusterSourceIndex( level, cLocal );
    else
        return this->AdjointLocalToClusterSourceIndex( level, cLocal );
}

template<size_t d>
inline size_t
Plan<d>::LocalToBootstrapClusterSourceIndex( size_t cLocal ) const
{
    if( this->_direction == FORWARD )
        return this->ForwardLocalToBootstrapClusterSourceIndex( cLocal );
    else
        return this->AdjointLocalToBootstrapClusterSourceIndex( cLocal );
}

} // bfio

#endif // ifndef BFIO_STRUCTURES_PLAN_HPP
