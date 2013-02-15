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
    MPI_Comm comm_;
    const Direction direction_;
    const size_t N_;

    // Does not depend on the problem size
    int rank_;
    int numProcs_;
    size_t log2NumProcs_;
    size_t log2N_;
    array<size_t,d> myInitialSBoxCoords_;
    array<size_t,d> myFinalTBoxCoords_;
    array<size_t,d> log2InitialSBoxesPerDim_;
    array<size_t,d> log2FinalTBoxesPerDim_;

    // Depends on the problem size
    const size_t bootstrap_;
    MPI_Comm bootstrapClusterComm_;
    vector<size_t> bootstrapSDimsToMerge_;
    vector<size_t> bootstrapTDimsToCut_;
    vector<bool> bootstrapRightSideOfCut_;
    vector<MPI_Comm> clusterComms_;
    vector<size_t> log2SubclusterSizes_;
    vector<vector<size_t>> sDimsToMerge_;
    vector<vector<size_t>> tDimsToCut_;
    vector<vector<bool>> rightSideOfCut_;

    PlanBase( MPI_Comm comm, Direction direction, size_t N, size_t bootstrap );

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
    size_t myBootstrapClusterRank_;
    vector<size_t> myClusterRanks_;
    void GenerateForwardPlan();

    size_t
    ForwardLocalToBootstrapClusterSourceIndex( size_t cLocal ) const;

    size_t 
    ForwardLocalToClusterSourceIndex( size_t level, size_t cLocal ) const;

    //---------------------------------//
    // For adjoint plans               //
    //---------------------------------//
    size_t myBootstrapMappedRank_;
    vector<size_t> myMappedRanks_;
    void GenerateAdjointPlan();

    size_t
    AdjointLocalToBootstrapClusterSourceIndex( size_t cLocal ) const;

    size_t 
    AdjointLocalToClusterSourceIndex( size_t level, size_t cLocal ) const;

public:
    Plan( MPI_Comm comm, Direction direction, size_t N, size_t bootstrap=0 );

    virtual size_t
    LocalToBootstrapClusterSourceIndex( size_t cLocal ) const;

    virtual size_t 
    LocalToClusterSourceIndex( size_t level, size_t cLocal ) const;
};

// Implementations
    
template<size_t d>
PlanBase<d>::PlanBase
( MPI_Comm comm, Direction direction, size_t N, size_t bootstrap ) 
: comm_(comm), direction_(direction), N_(N), bootstrap_(bootstrap)
{ 
    MPI_Comm_rank( comm, &rank_ );
    MPI_Comm_size( comm, &numProcs_ );

    if( ! IsPowerOfTwo(N) )
        throw std::runtime_error("Must use power of 2 problem size");
    if( ! IsPowerOfTwo(numProcs_) )
        throw std::runtime_error("Must use power of 2 number of processes");
    log2N_ = Log2( N );
    log2NumProcs_ = Log2( numProcs_ );
    if( log2NumProcs_ > d*log2N_ )
        throw std::runtime_error("Cannot use more than N^d processes");
    if( bootstrap > log2N_/2 )
        throw std::runtime_error("Cannot bootstrap past the middle switch");

    clusterComms_.resize( log2N_ );
    log2SubclusterSizes_.resize( log2N_ );
    sDimsToMerge_.resize( log2N_ );
    tDimsToCut_.resize( log2N_ );
    rightSideOfCut_.resize( log2N_ );
}

template<size_t d>
PlanBase<d>::~PlanBase()
{
    MPI_Comm_free( &bootstrapClusterComm_ );
    for( size_t level=1; level<=log2N_; ++level )
        MPI_Comm_free( &clusterComms_[level-1] );
}

template<size_t d>
inline MPI_Comm PlanBase<d>::GetComm() const 
{ return comm_; }

template<size_t d>
inline Direction
PlanBase<d>::GetDirection() const
{ return direction_; }

template<size_t d>
inline size_t 
PlanBase<d>::GetN() const 
{ return N_; }

template<size_t d>
inline size_t
PlanBase<d>::GetBootstrapSkip() const
{ return bootstrap_; }

template<size_t d> 
template<typename R>
Box<R,d> 
PlanBase<d>::GetMyInitialSourceBox( const Box<R,d>& sBox ) const
{
    Box<R,d> myInitialSBox;
    for( size_t j=0; j<d; ++j )
    {
        myInitialSBox.widths[j] = 
            sBox.widths[j] / (1u<<log2InitialSBoxesPerDim_[j]);
        myInitialSBox.offsets[j] = 
            sBox.offsets[j] + myInitialSBoxCoords_[j]*myInitialSBox.widths[j];
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
            tBox.widths[j] / (1u<<log2FinalTBoxesPerDim_[j]);
        myFinalTBox.offsets[j] = 
            tBox.offsets[j] + myFinalTBoxCoords_[j]*myFinalTBox.widths[j];
    }
    return myFinalTBox;
}

template<size_t d>
inline const array<size_t,d>& 
PlanBase<d>::GetMyInitialSourceBoxCoords() const
{ return myInitialSBoxCoords_; }

template<size_t d>
inline const array<size_t,d>& 
PlanBase<d>::GetMyFinalTargetBoxCoords() const
{ return myFinalTBoxCoords_; }

template<size_t d>
inline const array<size_t,d>& 
PlanBase<d>::GetLog2InitialSourceBoxesPerDim() const
{ return log2InitialSBoxesPerDim_; }

template<size_t d>
inline const array<size_t,d>& 
PlanBase<d>::GetLog2FinalTargetBoxesPerDim() const
{ return log2FinalTBoxesPerDim_; }

template<size_t d>
inline MPI_Comm 
PlanBase<d>::GetClusterComm( size_t level ) const
{ return clusterComms_[level-1]; }

template<size_t d>
inline MPI_Comm
PlanBase<d>::GetBootstrapClusterComm() const
{ return bootstrapClusterComm_; }

template<size_t d>
inline const vector<size_t>&
PlanBase<d>::GetBootstrapSourceDimsToMerge() const
{ return bootstrapSDimsToMerge_; }

template<size_t d>
inline const vector<size_t>&
PlanBase<d>::GetBootstrapTargetDimsToCut() const
{ return bootstrapTDimsToCut_; }

template<size_t d>
inline const vector<bool>& 
PlanBase<d>::GetBootstrapRightSideOfCut() const
{ return bootstrapRightSideOfCut_; }

template<size_t d>
inline size_t 
PlanBase<d>::GetLog2SubclusterSize( size_t level ) const
{ return log2SubclusterSizes_[level-1]; }

template<size_t d>
inline size_t
PlanBase<d>::GetLog2NumMergingProcesses( size_t level ) const
{ return sDimsToMerge_[level-1].size(); }

template<size_t d>
inline const vector<size_t>&
PlanBase<d>::GetSourceDimsToMerge( size_t level ) const
{ return sDimsToMerge_[level-1]; }

template<size_t d>
inline const vector<size_t>& 
PlanBase<d>::GetTargetDimsToCut( size_t level ) const
{ return tDimsToCut_[level-1]; }

template<size_t d>
inline const vector<bool>& 
PlanBase<d>::GetRightSideOfCut( size_t level ) const
{ return rightSideOfCut_[level-1]; }

template<size_t d>
Plan<d>::Plan
( MPI_Comm comm, Direction direction, size_t N, size_t bootstrap )
: PlanBase<d>( comm, direction, N, bootstrap )
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
    return (cLocal<<this->bootstrapSDimsToMerge_.size()) +
           this->myBootstrapClusterRank_;
}

template<size_t d>
inline size_t
Plan<d>::ForwardLocalToClusterSourceIndex( size_t level, size_t cLocal ) const
{
    return (cLocal<<this->sDimsToMerge_[level-1].size()) +     
           this->myClusterRanks_[level-1];
}

template<size_t d>
void
Plan<d>::GenerateForwardPlan()
{
    std::bitset<8*sizeof(int)> rankBits(this->rank_);
    myClusterRanks_.resize( this->log2N_ );

    // Compute the number of source boxes per dimension and our coordinates
    size_t nextSDimToCut = 0;
    size_t lastSDimCut = 0; // initialize to avoid compiler warnings
    for( size_t j=0; j<d; ++j )
    {
        this->myInitialSBoxCoords_[j] = 0;
        this->log2InitialSBoxesPerDim_[j] = 0;
    }
    for( size_t m=this->log2NumProcs_; m>0; --m )
    {
#ifndef RELEASE
        if( this->rank_ == 0 )
        {
            std::cout << "Cutting source dimension " << nextSDimToCut
                      << std::endl;
        }
#endif
        lastSDimCut = nextSDimToCut;
        this->myInitialSBoxCoords_[nextSDimToCut] <<= 1;
        if( rankBits[m-1] )
            ++this->myInitialSBoxCoords_[nextSDimToCut];
        ++this->log2InitialSBoxesPerDim_[nextSDimToCut];
        nextSDimToCut = (nextSDimToCut+1) % d;
    }
#ifndef RELEASE
    for( int p=0; p<this->numProcs_; ++p )
    {
        if( this->rank_ == p )
        {
            std::cout << "Rank " << p << "'s initial source box coords: ";
            for( size_t j=0; j<d; ++j )
                std::cout << this->myInitialSBoxCoords_[j] << " ";
            std::cout << std::endl;
        }
        MPI_Barrier( this->comm_ );
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
        log2LocalSBoxes += this->log2N_-this->log2InitialSBoxesPerDim_[j];
        this->myFinalTBoxCoords_[j] = 0;
        this->log2FinalTBoxesPerDim_[j] = 0;
    }
    // Generate the bootstrap communicator
    if( log2LocalSBoxes >= d*this->bootstrap_ )
    {
        this->myBootstrapClusterRank_ = 0; 

        MPI_Comm_split
        ( this->comm_, this->rank_, 0, &this->bootstrapClusterComm_ );

#ifndef RELEASE
        if( this->rank_ == 0 )
            std::cout << "No communication during bootstrapping." << std::endl;
#endif
    }
    else
    {
        const size_t log2NumMergingProcs = this->bootstrap_*d - log2LocalSBoxes;
        const size_t numMergingProcs = 1u<<log2NumMergingProcs;

#ifndef RELEASE
        if( this->rank_ == 0 )
        {
            std::cout << "Merging " << log2NumMergingProcs
                      << " dimension(s) during bootstrapping, starting with "
                      << nextSDimToMerge << " (cutting starting with "
                      << nextTDimToCut << ")" << std::endl;
        }
#endif

        // Construct the communicator for the bootstrap cluster
        const int startRank = this->rank_ & ~(numMergingProcs-1);
        vector<int> ranks( numMergingProcs );
        for( size_t j=0; j<numMergingProcs; ++j )
        {
            // We need to reverse the order of the last log2NumMergingProcs
            // bits of j and add the result onto the startRank
            size_t jReversed = 0;
            for( size_t k=0; k<log2NumMergingProcs; ++k )
                jReversed |= ((j>>k)&1)<<(log2NumMergingProcs-1-k);
            ranks[j] = startRank + jReversed;
            if( this->rank_ == ranks[j] )
                this->myBootstrapClusterRank_ = j;
        }

#ifndef RELEASE
        for( int p=0; p<this->numProcs_; ++p )        
        {
            if( this->rank_ == p )
            {
                std::cout << "  process " << p 
                          << "'s bootstrap cluster ranks: ";
                for( size_t j=0; j<numMergingProcs; ++j )
                    std::cout << ranks[j] << " ";
                std::cout << std::endl;
            }
            MPI_Barrier( this->comm_ );
            usleep( 100000 );
        }
#endif

        MPI_Comm_split
        ( this->comm_, ranks[0], this->myBootstrapClusterRank_, 
          &this->bootstrapClusterComm_ );

        this->bootstrapSDimsToMerge_.resize( log2NumMergingProcs );
        this->bootstrapTDimsToCut_.resize( log2NumMergingProcs );
        this->bootstrapRightSideOfCut_.resize( log2NumMergingProcs );
        size_t nextBootstrapSDimToMerge = nextSDimToMerge;
        size_t nextBootstrapTDimToCut = nextTDimToCut;
        for( size_t j=0; j<log2NumMergingProcs; ++j )
        {
            this->bootstrapSDimsToMerge_[j] = nextBootstrapSDimToMerge;
            this->bootstrapTDimsToCut_[j] = nextBootstrapTDimToCut;
            this->bootstrapRightSideOfCut_[j] = rankBits[j];

            nextBootstrapTDimToCut = (nextBootstrapTDimToCut+d-1) % d;
            nextBootstrapSDimToMerge = (nextBootstrapSDimToMerge+d-1) % d;
        }
#ifndef RELEASE
        for( int p=0; p<this->numProcs_; ++p )
        {
            if( this->rank_ == p )
            {
                std::cout << "  process " << p << "'s bootstrap cluster rank: "
                          << this->myBootstrapClusterRank_ << std::endl;
                std::cout << "  process " << p 
                          << "'s bootstrap cluster children: ";
                const size_t numLocalChildren =
                    (1u<<(this->bootstrap_*d-log2NumMergingProcs));
                for( size_t i=0; i<numLocalChildren; ++i )
                    std::cout << this->LocalToBootstrapClusterSourceIndex( i )
                              << " ";
                std::cout << std::endl;
            }
            MPI_Barrier( this->comm_ );
            usleep( 100000 );
        }
#endif
    }
    // Generate the single-level communicators
    for( size_t level=1; level<=this->log2N_; ++level )
    {
        if( log2LocalSBoxes >= d )
        {
            log2LocalSBoxes -= d;

            this->myClusterRanks_[level-1] = 0;
            
            MPI_Comm_split
            ( this->comm_, this->rank_, 0, &this->clusterComms_[level-1] );
            this->log2SubclusterSizes_[level-1] = 0;

#ifndef RELEASE
            if( this->rank_ == 0 )
            {
                std::cout << "No communication at level " << level
                          << ", there are now 2^" << log2LocalSBoxes
                          << " local source boxes." << std::endl;
            }
#endif
        }
        else
        {
            const size_t log2NumMergingProcs = d-log2LocalSBoxes;
            const size_t numMergingProcs = 1u<<log2NumMergingProcs;
            log2LocalSBoxes = 0;

#ifndef RELEASE
            if( this->rank_ == 0 )
            {
                std::cout << "Merging " << log2NumMergingProcs
                          << " dimension(s), starting with "
                          << nextSDimToMerge << " (cutting starting with "
                          << nextTDimToCut << ")" << std::endl;
            }
#endif

            // Construct the communicator for our current cluster
            const size_t log2Stride = numTCuts;
            const int startRank = 
                this->rank_ & ~((numMergingProcs-1)<<log2Stride);
            vector<int> ranks( numMergingProcs );
            for( size_t j=0; j<numMergingProcs; ++j )
            {
                // We need to reverse the order of the last
                // log2NumMergingProcs bits of j and add the result
                // multiplied by the stride onto the startRank
                size_t jReversed = 0;
                for( size_t k=0; k<log2NumMergingProcs; ++k )
                    jReversed |= ((j>>k)&1)<<(log2NumMergingProcs-1-k);
                ranks[j] = startRank+(jReversed<<log2Stride);
                if( this->rank_ == ranks[j] )
                    this->myClusterRanks_[level-1] = j;
            }
#ifndef RELEASE
            for( int p=0; p<this->numProcs_; ++p )
            {
                if( this->rank_ == p )
                {
                    std::cout << "  process " << p << "'s cluster ranks: ";
                    for( size_t j=0; j<numMergingProcs; ++j )
                        std::cout << ranks[j] << " ";
                    std::cout << std::endl;
                }
                MPI_Barrier( this->comm_ );
                usleep( 100000 );
            }
#endif
            MPI_Comm_split
            ( this->comm_, ranks[0], this->myClusterRanks_[level-1],
              &this->clusterComms_[level-1] );

#ifdef BGP
# ifdef BGP_MPIDO_USE_REDUCESCATTER
            MPIX_Set_property
            ( this->clusterComms_[level-1], MPIDO_USE_REDUCESCATTER, 1 );
# else
            MPIX_Set_property
            ( this->clusterComms_[level-1], MPIDO_USE_REDUCESCATTER, 0 );
# endif
#endif

            this->log2SubclusterSizes_[level-1] = 0;

            this->sDimsToMerge_[level-1].resize( log2NumMergingProcs );
            this->tDimsToCut_[level-1].resize( log2NumMergingProcs );
            this->rightSideOfCut_[level-1].resize( log2NumMergingProcs );
            for( size_t j=0; j<log2NumMergingProcs; ++j )
            {
                const size_t thisBit = numTCuts;

                this->sDimsToMerge_[level-1][j] = nextSDimToMerge;
                this->tDimsToCut_[level-1][j] = nextTDimToCut;
                this->rightSideOfCut_[level-1][j] = rankBits[thisBit];

                this->myFinalTBoxCoords_[nextTDimToCut] <<= 1;
                if( rankBits[thisBit] )
                    ++this->myFinalTBoxCoords_[nextTDimToCut];
                ++this->log2FinalTBoxesPerDim_[nextTDimToCut];

                ++numTCuts;
                nextTDimToCut = (nextTDimToCut+d-1) % d;
                nextSDimToMerge = (nextSDimToMerge+d-1) % d;
            }
#ifndef RELEASE
            for( int p=0; p<this->numProcs_; ++p )
            {
                if( this->rank_ == p )
                {
                    std::cout << "  process " << p << "'s cluster rank: "
                              << this->myClusterRanks_[level-1] << std::endl;
                    std::cout << "  process " << p << "'s cluster children: ";
                    const size_t numLocalChildren =
                        (1u<<(d-log2NumMergingProcs));
                    for( size_t i=0; i<numLocalChildren; ++i )
                        std::cout << this->LocalToClusterSourceIndex( level, i )
                                  << " ";
                    std::cout << std::endl;
                }
                MPI_Barrier( this->comm_ );
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
    return (this->myBootstrapMappedRank_<<
            (this->bootstrap_*d-this->bootstrapSDimsToMerge_.size()))
           + cLocal;
}

template<size_t d>
inline size_t
Plan<d>::AdjointLocalToClusterSourceIndex( size_t level, size_t cLocal ) const
{
    return (this->myMappedRanks_[level-1]<<
            (d-this->sDimsToMerge_[level-1].size())) + cLocal;
}

template<size_t d>
void
Plan<d>::GenerateAdjointPlan()
{
    std::bitset<8*sizeof(int)> rankBits(this->rank_);
    myMappedRanks_.resize( this->log2N_ );

    // Compute the number of source boxes per dimension and our coordinates
    size_t nextSDimToCut = d-1;
    size_t lastSDimCut = 0; // initialize to avoid compiler warnings
    for( size_t j=0; j<d; ++j )
    {
        this->myInitialSBoxCoords_[j] = 0;
        this->log2InitialSBoxesPerDim_[j] = 0;
    }
    // HERE
    for( size_t m=0; m<this->log2NumProcs_; ++m )
    {
#ifndef RELEASE
        if( this->rank_ == 0 )
        {
            std::cout << "Cutting source dimension " << nextSDimToCut
                      << std::endl;
        }
#endif
        lastSDimCut = nextSDimToCut;
        this->myInitialSBoxCoords_[nextSDimToCut] <<= 1;
        if( rankBits[m] )
            ++this->myInitialSBoxCoords_[nextSDimToCut];
        ++this->log2InitialSBoxesPerDim_[nextSDimToCut];
        nextSDimToCut = (nextSDimToCut+d-1) % d;
    }
#ifndef RELEASE
    for( int p=0; p<this->numProcs_; ++p )
    {
        if( this->rank_ == p )
        {
            std::cout << "Rank " << p << "'s initial source box coords: ";
            for( size_t j=0; j<d; ++j )
                std::cout << this->myInitialSBoxCoords_[j] << " ";
            std::cout << std::endl;
        }
        MPI_Barrier( this->comm_ );
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
        log2LocalSBoxes += this->log2N_-this->log2InitialSBoxesPerDim_[j];
        this->myFinalTBoxCoords_[j] = 0;
        this->log2FinalTBoxesPerDim_[j] = 0;
    }
    // Generate the bootstrap communicator
    if( log2LocalSBoxes >= d*this->bootstrap_ )
    {
        this->myBootstrapMappedRank_ = 0;

        MPI_Comm_split
        ( this->comm_, this->rank_, 0, &this->bootstrapClusterComm_ );

#ifndef RELEASE
        if( this->rank_ == 0 )
            std::cout << "No communication during bootstrapping." << std::endl;
#endif
    }
    else
    {
        const size_t log2NumMergingProcs = this->bootstrap_*d - log2LocalSBoxes;
        const size_t numMergingProcs = 1u<<log2NumMergingProcs;

#ifndef RELEASE
        if( this->rank_ == 0 )
        {
            std::cout << "Merging " << log2NumMergingProcs
                      << " dimension(s) during bootstrapping, starting with "
                      << nextSDimToMerge << " (cutting starting with "
                      << nextTDimToCut << ")" << std::endl;
        }
#endif

        // Construct the communicator for the bootstrap cluster
        const size_t log2Stride = this->log2NumProcs_-log2NumMergingProcs;
        const int startRank = this->rank_ & ~((numMergingProcs-1)<<log2Stride);
        vector<int> ranks( numMergingProcs );
        for( size_t j=0; j<numMergingProcs; ++j )
        {
            // We need to reverse the order of the last log2NumMergingProcs
            // bits of j and add the result multiplied by the stride onto the 
            // startRank
            size_t jReversed = 0;
            for( size_t k=0; k<log2NumMergingProcs; ++k )
                jReversed |= ((j>>k)&1)<<(log2NumMergingProcs-1-k);
            ranks[j] = startRank + (jReversed<<log2Stride);
            if( this->rank_ == ranks[j] )
                this->myBootstrapMappedRank_ = j;
        }

#ifndef RELEASE
        for( int p=0; p<this->numProcs_; ++p )
        {
            if( this->rank_ == p )
            {
                std::cout << "  process " << p
                          << "'s bootstrap cluster ranks: ";
                for( size_t j=0; j<numMergingProcs; ++j )
                    std::cout << ranks[j] << " ";
                std::cout << std::endl;
            }
            MPI_Barrier( this->comm_ );
            usleep( 100000 );
        }
#endif

        MPI_Comm_split
        ( this->comm_, ranks[0], this->myBootstrapMappedRank_, 
          &this->bootstrapClusterComm_ );

        this->bootstrapSDimsToMerge_.resize( log2NumMergingProcs );
        this->bootstrapTDimsToCut_.resize( log2NumMergingProcs );
        this->bootstrapRightSideOfCut_.resize( log2NumMergingProcs );
        size_t nextBootstrapSDimToMerge = nextSDimToMerge;
        size_t nextBootstrapTDimToCut = nextTDimToCut;
        for( size_t j=0; j<log2NumMergingProcs; ++j )
        {
            const size_t thisBit = (this->log2NumProcs_-1)-j;

            this->bootstrapSDimsToMerge_[j] = nextBootstrapSDimToMerge;
            this->bootstrapTDimsToCut_[j] = nextBootstrapTDimToCut;
            this->bootstrapRightSideOfCut_[j] = rankBits[thisBit];

            nextBootstrapTDimToCut = (nextBootstrapTDimToCut+1) % d;
            nextBootstrapSDimToMerge = (nextBootstrapSDimToMerge+1) % d;
        }
#ifndef RELEASE
        for( int p=0; p<this->numProcs_; ++p )
        {
            if( this->rank_ == p )
            {
                std::cout << "  process " << p << "'s bootstrap cluster rank: "
                          << this->myBootstrapClusterRank_ << std::endl;
                std::cout << "  process " << p
                          << "'s bootstrap cluster children: ";
                const size_t numLocalChildren =
                    (1u<<(this->bootstrap_*d-log2NumMergingProcs));
                for( size_t i=0; i<numLocalChildren; ++i )
                    std::cout << this->LocalToBootstrapClusterSourceIndex( i )
                              << " ";
                std::cout << std::endl;
            }
            MPI_Barrier( this->comm_ );
            usleep( 100000 );
        }
#endif
    }
    // Generate the single-level communicators
    for( size_t level=1; level<=this->log2N_; ++level )
    {
        if( log2LocalSBoxes >= d )
        {

            log2LocalSBoxes -= d;

            this->myMappedRanks_[level-1] = 0;

            MPI_Comm_split
            ( this->comm_, this->rank_, 0, &this->clusterComms_[level-1] );

            this->log2SubclusterSizes_[level-1] =  0;

#ifndef RELEASE
            if( this->rank_ == 0 )
            {
                std::cout << "No communication at level " << level 
                          << ", there are now 2^" << log2LocalSBoxes
                          << " local source boxes." << std::endl;
            }
#endif
        }
        else
        {
            const size_t log2NumMergingProcs = d-log2LocalSBoxes;
            const size_t numMergingProcs = 1u<<log2NumMergingProcs;
            log2LocalSBoxes = 0;

#ifndef RELEASE
            if( this->rank_ == 0 )
            {
                std::cout << "Merging " << log2NumMergingProcs 
                          << " dimension(s), starting with " 
                          << nextSDimToMerge << " (cutting starting with "
                          << nextTDimToCut << ")" << std::endl;
            }
#endif

            // Construct the communicator for our current cluster
            const size_t log2Stride = 
                this->log2NumProcs_-numTCuts-log2NumMergingProcs;
            const int startRank = 
                this->rank_ & ~((numMergingProcs-1)<<log2Stride);
            vector<int> ranks( numMergingProcs ); 
            {
                // The bits of j must be shuffled according to the ordering 
                // on the partition dimensions:
                //  - The last d-nextTDimToCut bits are reversed and last
                //  - The remaining bits are reversed but occur first
                const size_t lastBitsetSize = 
                    std::min(d-nextTDimToCut,log2NumMergingProcs);
                const size_t firstBitsetSize = 
                    log2NumMergingProcs-lastBitsetSize;
                for( size_t j=0; j<numMergingProcs; ++j )
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
            for( int p=0; p<this->numProcs_; ++p )
            {
                if( this->rank_ == p )
                {
                    std::cout << "  process " << p << "'s cluster ranks: ";
                    for( size_t j=0; j<numMergingProcs; ++j )
                        std::cout << ranks[j] << " ";
                    std::cout << std::endl;
                }
                MPI_Barrier( this->comm_ );
                usleep( 100000 );
            }
#endif

            {
                const size_t lastBitsetSize = 
                    std::min(d-nextSDimToMerge,log2NumMergingProcs);
                const size_t firstBitsetSize = 
                    log2NumMergingProcs-lastBitsetSize;
                // Apply the same transformation to our shifted rank
                const size_t shiftedRank = this->rank_-startRank;
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
                this->myMappedRanks_[level-1] = wrappedRank;
            }

            MPI_Comm_split
            ( this->comm_, ranks[0], this->myMappedRanks_[level-1],
              &this->clusterComms_[level-1] );
#ifdef BGP
# ifdef BGP_MPIDO_USE_REDUCESCATTER
            MPIX_Set_property
            ( this->clusterComms_[level-1], MPIDO_USE_REDUCESCATTER, 1 );
# else
            MPIX_Set_property
            ( this->clusterComms_[level-1], MPIDO_USE_REDUCESCATTER, 0 );
# endif
#endif

            this->log2SubclusterSizes_[level-1] = 
                this->log2NumProcs_ % d;

            this->sDimsToMerge_[level-1].resize( log2NumMergingProcs );
            this->tDimsToCut_[level-1].resize( log2NumMergingProcs );
            this->rightSideOfCut_[level-1].resize( log2NumMergingProcs );
            for( size_t j=0; j<log2NumMergingProcs; ++j )
            {
                const size_t thisBit = (this->log2NumProcs_-1) - numTCuts;

                this->sDimsToMerge_[level-1][j] =  nextSDimToMerge;
                this->tDimsToCut_[level-1][j] = nextTDimToCut;
                this->rightSideOfCut_[level-1][j] =  rankBits[thisBit];

                this->myFinalTBoxCoords_[nextTDimToCut] <<= 1;
                if( rankBits[thisBit] )
                    ++this->myFinalTBoxCoords_[nextTDimToCut];
                ++this->log2FinalTBoxesPerDim_[nextTDimToCut];

                ++numTCuts;
                nextTDimToCut = (nextTDimToCut+1) % d;
                nextSDimToMerge = (nextSDimToMerge+1) % d;
            }
            
#ifndef RELEASE
            for( int p=0; p<this->numProcs_; ++p )
            {
                if( this->rank_ == p )
                {
                    std::cout << "  process " << p << "'s mapped rank: "
                              << this->myMappedRanks_[level-1] << std::endl;
                    std::cout << "  process " << p << "'s cluster children: ";
                    const size_t numLocalChildren = 
                        (1u<<(d-log2NumMergingProcs));
                    for( size_t i=0; i<numLocalChildren; ++i ) 
                        std::cout << this->LocalToClusterSourceIndex( level, i )
                                  << " ";
                    std::cout << std::endl;
                }
                MPI_Barrier( this->comm_ );
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
    if( this->direction_ == FORWARD )
        return this->ForwardLocalToClusterSourceIndex( level, cLocal );
    else
        return this->AdjointLocalToClusterSourceIndex( level, cLocal );
}

template<size_t d>
inline size_t
Plan<d>::LocalToBootstrapClusterSourceIndex( size_t cLocal ) const
{
    if( this->direction_ == FORWARD )
        return this->ForwardLocalToBootstrapClusterSourceIndex( cLocal );
    else
        return this->AdjointLocalToBootstrapClusterSourceIndex( cLocal );
}

} // bfio

#endif // ifndef BFIO_STRUCTURES_PLAN_HPP
