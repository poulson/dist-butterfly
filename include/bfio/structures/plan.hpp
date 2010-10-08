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

#ifndef RELEASE
# include <iostream>
#endif

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
    MPI_Group _group;
    std::size_t _log2NumProcesses;
    std::size_t _log2N;
    Array<std::size_t,d> _myInitialSourceBoxCoords;
    Array<std::size_t,d> _myFinalTargetBoxCoords;
    Array<std::size_t,d> _log2InitialSourceBoxesPerDim;
    Array<std::size_t,d> _log2FinalTargetBoxesPerDim;

    // Depends on the problem size
    std::vector< MPI_Group                > _clusterGroups;
    std::vector< MPI_Comm                 > _clusterComms;
    std::vector< std::size_t              > _log2SubclusterSizes;
    std::vector< std::vector<std::size_t> > _sourceDimsToMerge;
    std::vector< std::vector<std::size_t> > _targetDimsToCut;
    std::vector< std::vector<bool       > > _rightSideOfCut;

    virtual void GeneratePlan() = 0;
    
public:        
    Plan( MPI_Comm comm, std::size_t N );
    virtual ~Plan();

    virtual std::size_t 
    LocalToClusterSourceIndex
    ( std::size_t level, std::size_t cLocal ) const = 0;

    MPI_Comm GetComm() const;
    std::size_t GetN() const;

    template<typename R>
    Box<R,d> GetMyInitialSourceBox( const Box<R,d>& sourceBox ) const;

    template<typename R>
    Box<R,d> GetMyFinalTargetBox( const Box<R,d>& targetBox ) const;

    const Array<std::size_t,d>& GetMyInitialSourceBoxCoords() const;
    const Array<std::size_t,d>& GetMyFinalTargetBoxCoords() const;
    const Array<std::size_t,d>& GetLog2InitialSourceBoxesPerDim() const;
    const Array<std::size_t,d>& GetLog2FinalTargetBoxesPerDim() const;

    MPI_Comm GetClusterComm( std::size_t level ) const;
    std::size_t GetLog2SubclusterSize( std::size_t level ) const;
    std::size_t GetLog2NumMergingProcesses( std::size_t level ) const;

    const std::vector<std::size_t>& 
    GetSourceDimsToMerge( std::size_t level ) const;

    const std::vector<std::size_t>& 
    GetTargetDimsToCut( std::size_t level ) const;

    const std::vector<bool>& 
    GetRightSideOfCut( std::size_t level ) const;
};

template<std::size_t d>
class ForwardPlan : public Plan<d>
{
    std::vector<std::size_t> _myClusterRanks;
    virtual void GeneratePlan();
public:
    ForwardPlan
    ( MPI_Comm comm, std::size_t N );

    virtual std::size_t 
    LocalToClusterSourceIndex
    ( std::size_t level, std::size_t cLocal ) const;
};

template<std::size_t d>
class InversePlan : public Plan<d>
{
    std::vector<std::size_t> _myMappedRanks;
    virtual void GeneratePlan();
public:
    InversePlan
    ( MPI_Comm comm, std::size_t N );
    
    virtual std::size_t 
    LocalToClusterSourceIndex( std::size_t level, std::size_t cLocal ) const;
};

// Implementations
    
template<std::size_t d>
Plan<d>::Plan( MPI_Comm comm, std::size_t N ) 
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

    _clusterGroups.resize( _log2N );
    _clusterComms.resize( _log2N );
    _log2SubclusterSizes.resize( _log2N );
    _sourceDimsToMerge.resize( _log2N );
    _targetDimsToCut.resize( _log2N );
    _rightSideOfCut.resize( _log2N );
}

template<std::size_t d>
Plan<d>::~Plan()
{
    for( std::size_t level=1; level<=_log2N; ++level )
    {
        MPI_Comm_free( &_clusterComms[level-1] );
        MPI_Group_free( &_clusterGroups[level-1] );
    }
    MPI_Group_free( &_group );
}

template<std::size_t d>
inline MPI_Comm Plan<d>::GetComm() const 
{ return _comm; }

template<std::size_t d>
inline std::size_t 
Plan<d>::GetN() const 
{ return _N; }

template<std::size_t d> template<typename R>
Box<R,d> 
Plan<d>::GetMyInitialSourceBox( const Box<R,d>& sourceBox ) const
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

template<std::size_t d> template<typename R>
Box<R,d> 
Plan<d>::GetMyFinalTargetBox( const Box<R,d>& targetBox ) const
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

template<std::size_t d>
inline const Array<std::size_t,d>& 
Plan<d>::GetMyInitialSourceBoxCoords() const
{ return _myInitialSourceBoxCoords; }

template<std::size_t d>
inline const Array<std::size_t,d>& 
Plan<d>::GetMyFinalTargetBoxCoords() const
{ return _myFinalTargetBoxCoords; }

template<std::size_t d>
inline const Array<std::size_t,d>& 
Plan<d>::GetLog2InitialSourceBoxesPerDim() const
{ return _log2InitialSourceBoxesPerDim; }

template<std::size_t d>
inline const Array<std::size_t,d>& 
Plan<d>::GetLog2FinalTargetBoxesPerDim() const
{ return _log2FinalTargetBoxesPerDim; }

template<std::size_t d>
inline MPI_Comm 
Plan<d>::GetClusterComm( std::size_t level ) const
{ return _clusterComms[level-1]; }

template<std::size_t d>
inline std::size_t 
Plan<d>::GetLog2SubclusterSize( std::size_t level ) const
{ return _log2SubclusterSizes[level-1]; }

template<std::size_t d>
inline std::size_t
Plan<d>::GetLog2NumMergingProcesses( std::size_t level ) const
{ return _sourceDimsToMerge[level-1].size(); }

template<std::size_t d>
inline const std::vector<std::size_t>&
Plan<d>::GetSourceDimsToMerge( std::size_t level ) const
{ return _sourceDimsToMerge[level-1]; }

template<std::size_t d>
inline const std::vector<std::size_t>& 
Plan<d>::GetTargetDimsToCut( std::size_t level ) const
{ return _targetDimsToCut[level-1]; }

template<std::size_t d>
inline const std::vector<bool>& 
Plan<d>::GetRightSideOfCut( std::size_t level ) const
{ return _rightSideOfCut[level-1]; }

template<std::size_t d>
ForwardPlan<d>::ForwardPlan
( MPI_Comm comm, std::size_t N )
: Plan<d>( comm, N )
{ 
    _myClusterRanks.resize( this->_log2N );
    GeneratePlan(); 
}

template<std::size_t d>
inline std::size_t
ForwardPlan<d>::LocalToClusterSourceIndex
( std::size_t level, std::size_t cLocal ) const
{
    return (cLocal<<this->_sourceDimsToMerge[level-1].size()) +     
           this->_myClusterRanks[level-1];
}

template<std::size_t d>
void
ForwardPlan<d>::GeneratePlan()
{
    std::bitset<8*sizeof(int)> rankBits(this->_rank);

    // Compute the number of source boxes per dimension and our coordinates
    std::size_t nextSourceDimToCut = 0;
    std::size_t lastSourceDimCut = 0; // initialize to avoid compiler warnings
    for( std::size_t j=0; j<d; ++j )
    {
        this->_myInitialSourceBoxCoords[j] = 0;
        this->_log2InitialSourceBoxesPerDim[j] = 0;
    }
    for( std::size_t m=this->_log2NumProcesses; m>0; --m )
    {
#ifndef RELEASE
        if( this->_rank == 0 )
        {
            std::cout << "Cutting source dimension " << nextSourceDimToCut
                      << std::endl;
        }
#endif
        lastSourceDimCut = nextSourceDimToCut;
        this->_myInitialSourceBoxCoords[nextSourceDimToCut] <<= 1;
        if( rankBits[m-1] )
            ++this->_myInitialSourceBoxCoords[nextSourceDimToCut];
        ++this->_log2InitialSourceBoxesPerDim[nextSourceDimToCut];
        nextSourceDimToCut = (nextSourceDimToCut+1) % d;
    }
#ifndef RELEASE
    for( int p=0; p<this->_numProcesses; ++p )
    {
        if( this->_rank == p )
        {
            std::cout << "Rank " << p << "'s initial source box coords: ";
            for( std::size_t j=0; j<d; ++j )
                std::cout << this->_myInitialSourceBoxCoords[j] << " ";
            std::cout << std::endl;
        }
        MPI_Barrier( this->_comm );
        usleep( 100000 );
    }
#endif

    // Generate subcommunicator vector by walking through the forward process
    std::size_t numTargetCuts = 0;
    std::size_t nextTargetDimToCut = d-1;
    std::size_t nextSourceDimToMerge = lastSourceDimCut;
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

            this->_myClusterRanks[level-1] = 0;
            
            MPI_Group_incl
            ( this->_group, 
              1, 
              &(this->_rank), 
              &(this->_clusterGroups[level-1]) );

            MPI_Comm_create
            ( this->_comm, 
              this->_clusterGroups[level-1], 
              &(this->_clusterComms[level-1]) );

            this->_log2SubclusterSizes[level-1] = 0;

#ifndef RELEASE
            if( this->_rank == 0 )
            {
                std::cout << "No communication at level " << level
                          << ", there are now 2^" << log2LocalSourceBoxes
                          << " local source boxes." << std::endl;
            }
#endif
        }
        else
        {
            const std::size_t log2NumMergingProcesses = d-log2LocalSourceBoxes;
            const std::size_t numMergingProcesses = 1u<<log2NumMergingProcesses;
            log2LocalSourceBoxes = 0;

#ifndef RELEASE
            if( this->_rank == 0 )
            {
                std::cout << "Merging " << log2NumMergingProcesses
                          << " dimension(s), starting with "
                          << nextSourceDimToMerge << " (cutting starting with "
                          << nextTargetDimToCut << ")" << std::endl;
            }
#endif

            // Construct the group and communicator for our current cluster
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
                if( this->_rank == ranks[j] )
                    this->_myClusterRanks[level-1] = j;
            }
#ifndef RELEASE
            for( int p=0; p<this->_numProcesses; ++p )
            {
                if( this->_rank == p )
                {
                    std::cout << "  process " << p << "'s cluster group: ";
                    for( std::size_t j=0; j<numMergingProcesses; ++j )
                        std::cout << ranks[j] << " ";
                    std::cout << std::endl;
                }
                MPI_Barrier( this->_comm );
                usleep( 100000 );
            }
#endif
            MPI_Group_incl
            ( this->_group, 
              numMergingProcesses, 
              &ranks[0], 
              &(this->_clusterGroups[level-1]) );
            MPI_Comm_create
            ( this->_comm, 
              this->_clusterGroups[level-1],
              &(this->_clusterComms[level-1]) );
            this->_log2SubclusterSizes[level-1] = 0;

            this->_sourceDimsToMerge[level-1].resize( log2NumMergingProcesses );
            this->_targetDimsToCut[level-1].resize( log2NumMergingProcesses );
            this->_rightSideOfCut[level-1].resize( log2NumMergingProcesses );
            for( std::size_t j=0; j<log2NumMergingProcesses; ++j )
            {
                const std::size_t thisBit = numTargetCuts;

                this->_sourceDimsToMerge[level-1][j] = nextSourceDimToMerge;
                this->_targetDimsToCut[level-1][j] = nextTargetDimToCut;
                this->_rightSideOfCut[level-1][j] = rankBits[thisBit];

                this->_myFinalTargetBoxCoords[nextTargetDimToCut] <<= 1;
                if( rankBits[thisBit] )
                    ++this->_myFinalTargetBoxCoords[nextTargetDimToCut];
                ++this->_log2FinalTargetBoxesPerDim[nextTargetDimToCut];

                ++numTargetCuts;
                nextTargetDimToCut = (nextTargetDimToCut+d-1) % d;
                nextSourceDimToMerge = (nextSourceDimToMerge+d-1) % d;
            }
#ifndef RELEASE
            for( int p=0; p<this->_numProcesses; ++p )
            {
                if( this->_rank == p )
                {
                    std::cout << "  process " << p << "'s cluster rank: "
                              << this->_myClusterRanks[level-1] << std::endl;
                    std::cout << "  process " << p << "'s cluster children: ";
                    const std::size_t numLocalChildren =
                        (1u<<(d-log2NumMergingProcesses));
                    for( std::size_t i=0; i<numLocalChildren; ++i )
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

template<std::size_t d>
InversePlan<d>::InversePlan
( MPI_Comm comm, std::size_t N )
: Plan<d>( comm, N )
{ 
    _myMappedRanks.resize( this->_log2N );
    GeneratePlan(); 
}

template<std::size_t d>
inline std::size_t
InversePlan<d>::LocalToClusterSourceIndex
( std::size_t level, std::size_t cLocal ) const
{
    return (this->_myMappedRanks[level-1]<<
            (d-this->_sourceDimsToMerge[level-1].size())) + cLocal;
}

template<std::size_t d>
void
InversePlan<d>::GeneratePlan()
{
    std::bitset<8*sizeof(int)> rankBits(this->_rank);

    // Compute the number of source boxes per dimension and our coordinates
    std::size_t nextSourceDimToCut = d-1;
    std::size_t lastSourceDimCut = 0; // initialize to avoid compiler warnings
    for( std::size_t j=0; j<d; ++j )
    {
        this->_myInitialSourceBoxCoords[j] = 0;
        this->_log2InitialSourceBoxesPerDim[j] = 0;
    }
    for( std::size_t m=0; m<this->_log2NumProcesses; ++m )
    {
#ifndef RELEASE
        if( this->_rank == 0 )
        {
            std::cout << "Cutting source dimension " << nextSourceDimToCut
                      << std::endl;
        }
#endif
        lastSourceDimCut = nextSourceDimToCut;
        this->_myInitialSourceBoxCoords[nextSourceDimToCut] <<= 1;
        if( rankBits[m] )
            ++this->_myInitialSourceBoxCoords[nextSourceDimToCut];
        ++this->_log2InitialSourceBoxesPerDim[nextSourceDimToCut];
        nextSourceDimToCut = (nextSourceDimToCut+d-1) % d;
    }
#ifndef RELEASE
    for( int p=0; p<this->_numProcesses; ++p )
    {
        if( this->_rank == p )
        {
            std::cout << "Rank " << p << "'s initial source box coords: ";
            for( std::size_t j=0; j<d; ++j )
                std::cout << this->_myInitialSourceBoxCoords[j] << " ";
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
    std::size_t numTargetCuts = 0;
    std::size_t nextTargetDimToCut = 0;
    std::size_t nextSourceDimToMerge = lastSourceDimCut;
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

            this->_myMappedRanks[level-1] = 0;

            MPI_Group_incl
            ( this->_group, 
              1, 
              &(this->_rank), 
              &(this->_clusterGroups[level-1]) );

            MPI_Comm_create
            ( this->_comm, 
              this->_clusterGroups[level-1], 
              &(this->_clusterComms[level-1]) );
            
            this->_log2SubclusterSizes[level-1] =  0;

#ifndef RELEASE
            if( this->_rank == 0 )
            {
                std::cout << "No communication at level " << level 
                          << ", there are now 2^" << log2LocalSourceBoxes
                          << " local source boxes." << std::endl;
            }
#endif
        }
        else
        {
            const std::size_t log2NumMergingProcesses = d-log2LocalSourceBoxes;
            const std::size_t numMergingProcesses = 1u<<log2NumMergingProcesses;
            log2LocalSourceBoxes = 0;

#ifndef RELEASE
            if( this->_rank == 0 )
            {
                std::cout << "Merging " << log2NumMergingProcesses 
                          << " dimension(s), starting with " 
                          << nextSourceDimToMerge << " (cutting starting with "
                          << nextTargetDimToCut << ")" << std::endl;
            }
#endif

            // Construct the group and communicator for our current cluster
            const std::size_t log2Stride = 
                this->_log2NumProcesses-numTargetCuts-log2NumMergingProcesses;
            const int startRank = 
                this->_rank & ~((numMergingProcesses-1)<<log2Stride);
            std::vector<int> ranks( numMergingProcesses ); 
            {
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
                }
            }
#ifndef RELEASE
            for( int p=0; p<this->_numProcesses; ++p )
            {
                if( this->_rank == p )
                {
                    std::cout << "  process " << p << "'s cluster group: ";
                    for( std::size_t j=0; j<numMergingProcesses; ++j )
                        std::cout << ranks[j] << " ";
                    std::cout << std::endl;
                }
                MPI_Barrier( this->_comm );
                usleep( 100000 );
            }
#endif

            {
                const std::size_t lastBitsetSize = 
                    std::min(d-nextSourceDimToMerge,log2NumMergingProcesses);
                const std::size_t firstBitsetSize = 
                    log2NumMergingProcesses-lastBitsetSize;
                // Apply the same transformation to our shifted rank
                const std::size_t shiftedRank = this->_rank-startRank;
                const std::size_t scaledRank = shiftedRank >> log2Stride;
                std::size_t wrappedRank = 0;
                for( std::size_t k=0; k<firstBitsetSize; ++k )
                {
                    wrappedRank |=
                        ((scaledRank>>k)&1)<<(firstBitsetSize-1-k);
                }
                for( std::size_t k=0; k<lastBitsetSize; ++k )
                {
                    wrappedRank |=
                        ((scaledRank>>(k+firstBitsetSize))&1)<<
                         (lastBitsetSize-1-k+firstBitsetSize);
                }
                this->_myMappedRanks[level-1] = wrappedRank;
            }
            MPI_Group_incl
            ( this->_group,
              numMergingProcesses,
              &ranks[0],
              &(this->_clusterGroups[level-1]) );
            MPI_Comm_create
            ( this->_comm, 
              this->_clusterGroups[level-1],
              &(this->_clusterComms[level-1]) );
            this->_log2SubclusterSizes[level-1] = 
                this->_log2NumProcesses % d;

            this->_sourceDimsToMerge[level-1].resize( log2NumMergingProcesses );
            this->_targetDimsToCut[level-1].resize( log2NumMergingProcesses );
            this->_rightSideOfCut[level-1].resize( log2NumMergingProcesses );
            for( std::size_t j=0; j<log2NumMergingProcesses; ++j )
            {
                const std::size_t thisBit = 
                    (this->_log2NumProcesses-1) - numTargetCuts;

                this->_sourceDimsToMerge[level-1][j] =  nextSourceDimToMerge;
                this->_targetDimsToCut[level-1][j] = nextTargetDimToCut;
                this->_rightSideOfCut[level-1][j] =  rankBits[thisBit];

                this->_myFinalTargetBoxCoords[nextTargetDimToCut] <<= 1;
                if( rankBits[thisBit] )
                    ++this->_myFinalTargetBoxCoords[nextTargetDimToCut];
                ++this->_log2FinalTargetBoxesPerDim[nextTargetDimToCut];

                ++numTargetCuts;
                nextTargetDimToCut = (nextTargetDimToCut+1) % d;
                nextSourceDimToMerge = (nextSourceDimToMerge+1) % d;
            }
            
#ifndef RELEASE
            for( int p=0; p<this->_numProcesses; ++p )
            {
                if( this->_rank == p )
                {
                    std::cout << "  process " << p << "'s mapped rank: "
                              << this->_myMappedRanks[level-1] << std::endl;
                    std::cout << "  process " << p << "'s cluster children: ";
                    const std::size_t numLocalChildren = 
                        (1u<<(d-log2NumMergingProcesses));
                    for( std::size_t i=0; i<numLocalChildren; ++i ) 
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

} // bfio

#endif // BFIO_STRUCTURES_PLAN_HPP

