/*
   Copyright (C) 2010-2013 Jack Poulson and Lexing Ying
 
   This file is part of ButterflyFIO and is under the GNU General Public 
   License, which can be found in the LICENSE file in the root directory, or at
   <http://www.gnu.org/licenses/>.
*/
#include <ctime>
#include <memory>
#include "bfio.hpp"
using namespace std;
using namespace bfio;

void 
Usage()
{
    cout << "Random3DWaves <N> <M> <bootstrap> <T> <nT>\n" 
         << "  N: power of 2, the source spread in each dimension\n" 
         << "  M: number of random sources to instantiate\n" 
         << "  bootstrapSkip: level to bootstrap to\n"
         << "  T: time to simulate to\n" 
         << "  nT: number of timesteps\n" 
         << endl;
}

// Define the dimension of the problem and the order of interpolation
static const size_t d = 3;
static const size_t q = 5;

template<typename R>
class UpWave : public Phase<R,d>
{
    R _t;
public:
    UpWave();

    virtual UpWave<R>* Clone() const override;

    virtual R
    operator()( const array<R,d>& x, const array<R,d>& p ) const override;

    // We can optionally override the batched application for better efficiency
    virtual void
    BatchEvaluate
    ( const vector<array<R,d>>& xPoints,
      const vector<array<R,d>>& pPoints,
            vector<R         >& results ) const override;

    void SetTime( const R t );
    R GetTime() const;
};

template<typename R>
class DownWave : public Phase<R,d>
{
    R _t;
public:
    DownWave();

    virtual DownWave<R>* Clone() const override;

    virtual R
    operator()( const array<R,d>& x, const array<R,d>& p ) const override;
    
    // We can optionally override the batched application for better efficiency
    virtual void
    BatchEvaluate
    ( const vector<array<R,d>>& xPoints,
      const vector<array<R,d>>& pPoints,
            vector<R         >& results ) const override;

    void SetTime( const R t );
    R GetTime() const;
};

template<typename R>
inline
UpWave<R>::UpWave() 
: _t(0) 
{ }

template<typename R>
inline
DownWave<R>::DownWave() 
: _t(0) 
{ }

template<typename R>
inline UpWave<R>*
UpWave<R>::Clone() const
{ return new UpWave<R>(*this); }

template<typename R>
inline DownWave<R>*
DownWave<R>::Clone() const
{ return new DownWave<R>(*this); }

template<typename R>
inline void 
UpWave<R>::SetTime( const R t ) 
{ _t = t; }

template<typename R>
inline void 
DownWave<R>::SetTime( const R t ) 
{ _t = t; }

template<typename R>
inline R 
UpWave<R>::GetTime() const 
{ return _t; }

template<typename R>
inline R 
DownWave<R>::GetTime() const 
{ return _t; }

template<typename R>
inline R
UpWave<R>::operator()( const array<R,d>& x, const array<R,d>& p ) const
{ 
    return TwoPi*( 
             x[0]*p[0]+x[1]*p[1]+x[2]*p[2] + 
             _t * sqrt(p[0]*p[0]+p[1]*p[1]+p[2]*p[2])
           );
}

template<typename R>
inline R
DownWave<R>::operator()( const array<R,d>& x, const array<R,d>& p ) const
{ 
    return TwoPi*(
             x[0]*p[0]+x[1]*p[1]+x[2]*p[2] -
             _t * sqrt(p[0]*p[0]+p[1]*p[1]+p[2]*p[2])
           );
}

template<typename R>
void
UpWave<R>::BatchEvaluate
( const vector<array<R,d>>& xPoints,
  const vector<array<R,d>>& pPoints,
        vector<R         >& results ) const
{
    const size_t xSize = xPoints.size();
    const size_t pSize = pPoints.size();

    // Set up the square root arguments
    vector<R> sqrtArguments( pSize );
    {
        R* RESTRICT sqrtArgBuffer = &sqrtArguments[0];
        const R* RESTRICT pPointsBuffer = &(pPoints[0][0]);
        for( size_t j=0; j<pSize; ++j )
            sqrtArgBuffer[j] = pPointsBuffer[j*d+0]*pPointsBuffer[j*d+0] +
                               pPointsBuffer[j*d+1]*pPointsBuffer[j*d+1] +
                               pPointsBuffer[j*d+2]*pPointsBuffer[j*d+2];
    }

    // Perform the batched square roots
    vector<R> sqrtResults;
    SqrtBatch( sqrtArguments, sqrtResults );

    // Scale the square roots by _t
    {
        R* sqrtBuffer = &sqrtResults[0];
        for( size_t j=0; j<pSize; ++j )
            sqrtBuffer[j] *= _t;
    }

    // Form the final results
    results.resize( xSize*pSize );
    {
        R* RESTRICT resultsBuffer = &results[0];
        const R* RESTRICT sqrtBuffer = &sqrtResults[0];
        const R* RESTRICT xPointsBuffer = &(xPoints[0][0]);
        const R* RESTRICT pPointsBuffer = &(pPoints[0][0]);
        for( size_t i=0; i<xSize; ++i )
        {
            for( size_t j=0; j<pSize; ++j )
            {
                resultsBuffer[i*pSize+j] = 
                    xPointsBuffer[i*d+0]*pPointsBuffer[j*d+0] + 
                    xPointsBuffer[i*d+1]*pPointsBuffer[j*d+1] +
                    xPointsBuffer[i*d+2]*pPointsBuffer[j*d+2] +
                    sqrtBuffer[j];
                resultsBuffer[i*pSize+j] *= TwoPi;
            }
        }
    }
}

template<typename R>
void
DownWave<R>::BatchEvaluate
( const vector<array<R,d>>& xPoints,
  const vector<array<R,d>>& pPoints,
        vector<R         >& results ) const
{
    const size_t xSize = xPoints.size();
    const size_t pSize = pPoints.size();

    // Set up the square root arguments
    vector<R> sqrtArguments( pSize );
    {
        R* sqrtArgBuffer = &sqrtArguments[0];
        const R* pPointsBuffer = &(pPoints[0][0]);
        for( size_t j=0; j<pSize; ++j )
            sqrtArgBuffer[j] = pPointsBuffer[j*d+0]*pPointsBuffer[j*d+0] +
                               pPointsBuffer[j*d+1]*pPointsBuffer[j*d+1] +
                               pPointsBuffer[j*d+2]*pPointsBuffer[j*d+2];
    }

    // Perform the batched square roots
    vector<R> sqrtResults;
    SqrtBatch( sqrtArguments, sqrtResults );

    // Scale the square roots by _t
    {
        R* sqrtBuffer = &sqrtResults[0];
        for( size_t j=0; j<pSize; ++j )
            sqrtBuffer[j] *= _t;
    }

    // Form the final results
    results.resize( xSize*pSize );
    {
        R* resultsBuffer = &results[0];
        const R* sqrtBuffer = &sqrtResults[0];
        const R* xPointsBuffer = &(xPoints[0][0]);
        const R* pPointsBuffer = &(pPoints[0][0]);
        for( size_t i=0; i<xSize; ++i )
        {
            for( size_t j=0; j<pSize; ++j )
            {
                resultsBuffer[i*pSize+j] = 
                    xPointsBuffer[i*d+0]*pPointsBuffer[j*d+0] + 
                    xPointsBuffer[i*d+1]*pPointsBuffer[j*d+1] +
                    xPointsBuffer[i*d+2]*pPointsBuffer[j*d+2] -
                    sqrtBuffer[j];
                resultsBuffer[i*pSize+j] *= TwoPi;
            }
        }
    }
}

int
main( int argc, char* argv[] )
{
    MPI_Init( &argc, &argv );

    int rank, numProcesses;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_rank( comm, &rank );
    MPI_Comm_size( comm, &numProcesses );

    if( argc != 6 )
    {
        if( rank == 0 )
            Usage();
        MPI_Finalize();
        return 0;
    }
    const size_t N = atoi(argv[1]);
    const size_t M = atoi(argv[2]);
    const size_t bootstrapSkip = atoi(argv[3]);
    const double T = atof(argv[4]);
    const size_t nT = atoi(argv[5]);

    try 
    {
        // Define the source and target boxes
        Box<double,d> sourceBox, targetBox; 
        for( size_t j=0; j<d; ++j )
        {
            sourceBox.offsets[j] = -0.5*N;
            sourceBox.widths[j] = N;
            targetBox.offsets[j] = 0;
            targetBox.widths[j] = 1;
        }

        // Set up the general strategy for the forward transform
        Plan<d> plan( comm, FORWARD, N, bootstrapSkip );
        Box<double,d> mySourceBox = plan.GetMyInitialSourceBox( sourceBox );

        if( rank == 0 )
        {
            ostringstream msg;
            msg << "Will distribute " << M << " random sources over the source "
                << "domain, which will be split into " << N
                << " boxes in each of the " << d << " dimensions and "
                << "distributed amongst " << numProcesses << " processes. "
                << "The simulation will be over " << T << " units of time with "
                << nT << " timesteps.\n";
            cout << msg.str() << endl;
        }

        // Consistently seed all of the processes' PRNGs
        long seed;
        if( rank == 0 )
            seed = time(0);
        MPI_Bcast( &seed, 1, MPI_LONG, 0, comm );
        srand( seed );

        // Now generate random sources in our frequency box
        size_t numLocalSources = 
            ( rank<(int)(M%numProcesses) 
              ? M/numProcesses+1 : M/numProcesses );
        vector<Source<double,d>> mySources( numLocalSources );
        for( size_t i=0; i<numLocalSources; ++i )
        {
            for( size_t j=0; j<d; ++j )
            {
                mySources[i].p[j] = 
                    mySourceBox.offsets[j] +
                    Uniform<double>()*mySourceBox.widths[j];
            }
            mySources[i].magnitude = 200*Uniform<double>()-100;
        }

        // Set up our phase functors
        UpWave<double> upWave;
        DownWave<double> downWave;

        // Create the context 
        if( rank == 0 )
            cout << "Creating context..." << endl;
        rfio::Context<double,d,q> context;

        // Loop over each timestep, computing in parallel, gathering the 
        // results, and then dumping to file
        double deltaT = T/(nT-1);
        for( size_t i=0; i<nT; ++i )
        {
            const double t = i*deltaT;
            upWave.SetTime( t );
            downWave.SetTime( t );

            if( rank == 0 )
            {
                cout << "t=" << t << "\n"
                          << "  Starting upWave transform...";
                cout.flush();
            }
            auto u = RFIO
            ( context, plan, upWave, sourceBox, targetBox, mySources );
            if( rank == 0 )
                cout << "done" << endl;
#ifdef TIMING
            if( rank == 0 )
                rfio::PrintTimings();
#endif

            if( rank == 0 )
            {
                cout << "  Starting downWave transform...";
                cout.flush();
            }
            auto v = RFIO
            ( context, plan, downWave, sourceBox, targetBox, mySources );
            if( rank == 0 )
                cout << "done" << endl;
#ifdef TIMING
            if( rank == 0 )
                rfio::PrintTimings();
#endif

            // Store this timeslice
            ostringstream fileStream;
            fileStream << "randomWaves-" << i;
            rfio::WriteImage( comm, N, targetBox, *u, fileStream.str() );
        }
    }
    catch( const exception& e )
    {
        ostringstream msg;
        msg << "Caught exception on process " << rank << ":\n"
            << "   " << e.what();
        cout << msg.str() << endl;
    }

    MPI_Finalize();
    return 0;
}
