/*
   Copyright (C) 2010-2013 Jack Poulson and Lexing Ying
 
   This file is part of ButterflyFIO and is under the GNU General Public 
   License, which can be found in the LICENSE file in the root directory, or at
   <http://www.gnu.org/licenses/>.
*/
#include <ctime>
#include <fstream>
#include <memory>
#include "bfio.hpp"
using namespace std;

void 
Usage()
{
    cout << "NonUniformFT-2d <N> <M> <bootstrap> <testAccuracy?> <store?>\n"
         << "  N: power of 2, the source spread in each dimension\n" 
         << "  M: number of random sources to instantiate\n" 
         << "  bootstrap: level to bootstrap to\n"
         << "  testAccuracy?: tests accuracy iff 1\n" 
         << "  store?: creates data files iff 1\n" 
         << endl;
}

// Define the dimension of the problem and the order of interpolation
static const size_t d = 2;
static const size_t q = 7;

template<typename R>
class Fourier : public bfio::Phase<R,d>
{
public:
    virtual Fourier<R>* Clone() const;

    virtual R 
    operator()( const array<R,d>& x, const array<R,d>& p ) const;

    // We can optionally override the batched application for better efficiency
    virtual void
    BatchEvaluate
    ( const vector<array<R,d>>& xPoints,
      const vector<array<R,d>>& pPoints,
            vector<R         >& results ) const;
};

template<typename R>
inline Fourier<R>*
Fourier<R>::Clone() const
{ return new Fourier<R>(*this); }

template<typename R>
inline R
Fourier<R>::operator()( const array<R,d>& x, const array<R,d>& p ) const
{ return -bfio::TwoPi*(x[0]*p[0]+x[1]*p[1]); }

// We can optionally override the batched application for better efficiency
template<typename R>
void
Fourier<R>::BatchEvaluate
( const vector<array<R,d>>& xPoints,
  const vector<array<R,d>>& pPoints,
        vector<R         >& results ) const
{
    const size_t xSize = xPoints.size();
    const size_t pSize = pPoints.size();
    results.resize( xSize*pSize );

    R* RESTRICT resultsBuffer = &results[0];
    const R* RESTRICT xPointsBuffer = &(xPoints[0][0]);
    const R* RESTRICT pPointsBuffer = &(pPoints[0][0]);
    for( size_t i=0; i<xSize; ++i )
    {
        for( size_t j=0; j<pSize; ++j )
        {
            resultsBuffer[i*pSize+j] = 
                xPointsBuffer[i*d+0]*pPointsBuffer[j*d+0] + 
                xPointsBuffer[i*d+1]*pPointsBuffer[j*d+1];
            resultsBuffer[i*pSize+j] *= -bfio::TwoPi;
        }
    }
}

int
main
( int argc, char* argv[] )
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
    const bool testAccuracy = atoi(argv[4]);
    const bool store = atoi(argv[5]);

    try 
    {
        // Set our source and target boxes
        bfio::Box<double,d> sourceBox, targetBox;
        for( size_t j=0; j<d; ++j )
        {
            sourceBox.offsets[j] = -0.5*N;
            sourceBox.widths[j] = N;
            targetBox.offsets[j] = 0;
            targetBox.widths[j] = 1;
        }

        // Set up the general strategy for the forward transform
        bfio::Plan<d> plan( comm, bfio::FORWARD, N, bootstrapSkip );
        bfio::Box<double,d> mySourceBox = 
            plan.GetMyInitialSourceBox( sourceBox );

        if( rank == 0 )
        {
            ostringstream msg;
            msg << "Will distribute " << M << " random sources over the source "
                << "domain, which will be split into " << N 
                << " boxes in each of the " << d << " dimensions and "
                << "distributed amongst " << numProcesses << " processes.\n";
            cout << msg.str() << endl;
        }

        // Consistently randomly seed all of the processes' PRNG.
        long seed;
        if( rank == 0 )
            seed = time(0);
        MPI_Bcast( &seed, 1, MPI_LONG, 0, comm );
        srand( seed );

        // Now generate random sources across the domain and store them in 
        // our local list when appropriate
        vector<bfio::Source<double,d>> mySources;
        vector<bfio::Source<double,d>> globalSources;
        if( testAccuracy || store )
        {
            globalSources.resize( M );
            for( size_t i=0; i<M; ++i )
            {
                for( size_t j=0; j<d; ++j )
                {
                    globalSources[i].p[j] = sourceBox.offsets[j] + 
                        sourceBox.widths[j]*bfio::Uniform<double>(); 
                }
                globalSources[i].magnitude = 1.*(2*bfio::Uniform<double>()-1); 

                // Check if we should push this source onto our local list
                bool isMine = true;
                for( size_t j=0; j<d; ++j )
                {
                    double u = globalSources[i].p[j];
                    double start = mySourceBox.offsets[j];
                    double stop = 
                        mySourceBox.offsets[j] + mySourceBox.widths[j];
                    if( u < start || u >= stop )
                        isMine = false;
                }
                if( isMine )
                    mySources.push_back( globalSources[i] );
            }
        }
        else
        {
            size_t numLocalSources = 
                ( rank<(int)(M%numProcesses) 
                  ? M/numProcesses+1 : M/numProcesses );
            mySources.resize( numLocalSources );
            for( size_t i=0; i<numLocalSources; ++i )
            {
                for( size_t j=0; j<d; ++j )
                {
                    mySources[i].p[j] = 
                        mySourceBox.offsets[j] + 
                        bfio::Uniform<double>()*mySourceBox.widths[j];
                }
                mySources[i].magnitude = 1.*(2*bfio::Uniform<double>()-1);
            }
        }

        // Create a context for Interpolative NUFTs
        if( rank == 0 )
            cout << "Creating InterpolativeNUFT context..." << endl;
        bfio::inuft::Context<double,d,q> 
            inuftContext( bfio::FORWARD, N, sourceBox, targetBox );

        // Run with the interpolative NUFT
        if( rank == 0 )
            cout << "Starting InterpolativeNUFT..." << endl;
        MPI_Barrier( comm );
        double startTime = MPI_Wtime();
        auto u = bfio::INUFT
        ( inuftContext, plan, sourceBox, targetBox, mySources );
        MPI_Barrier( comm );
        double stopTime = MPI_Wtime();
        if( rank == 0 )
            cout << "Runtime: " << stopTime-startTime << " seconds.\n" << endl;
#ifdef TIMING
        if( rank == 0 )
            bfio::inuft::PrintTimings();
#endif

        // Create a context for NUFTs with Lagrangian interpolation
        if( rank == 0 )
            cout << "Creating LagrangianNUFT context..." << endl;
        bfio::lnuft::Context<double,d,q> 
            lnuftContext( bfio::FORWARD, N, sourceBox, targetBox );

        // Run with the Lagrangian NUFT
        if( rank == 0 )
            cout << "Starting LagrangianNUFT..." << endl;
        MPI_Barrier( comm );
        startTime = MPI_Wtime();
        auto v = bfio::LNUFT
        ( lnuftContext, plan, sourceBox, targetBox, mySources );
        MPI_Barrier( comm );
        stopTime = MPI_Wtime();
        if( rank == 0 )
            cout << "Runtime: " << stopTime-startTime << " seconds.\n" << endl;
#ifdef TIMING
        if( rank == 0 )
            bfio::lnuft::PrintTimings();
#endif

        // Set up our phase functor
        Fourier<double> fourier;

        // Create a general context 
        if( rank == 0 )
            cout << "Creating ReducedFIO context..." << endl;
        bfio::rfio::Context<double,d,q> rfioContext;

        // Run with the general algorithm
        if( rank == 0 )
            cout << "Starting ReducedFIO transform..." << endl;
        MPI_Barrier( comm );
        startTime = MPI_Wtime();
        auto w = bfio::ReducedFIO
        ( rfioContext, plan, fourier, sourceBox, targetBox, mySources );
        MPI_Barrier( comm );
        stopTime = MPI_Wtime();
        if( rank == 0 )
            cout << "Runtime: " << stopTime-startTime << " seconds.\n" << endl;
#ifdef TIMING
        if( rank == 0 )
            bfio::rfio::PrintTimings();
#endif

        if( testAccuracy )
            bfio::lnuft::PrintErrorEstimates( comm, *v, globalSources );
        
        if( store )
        {
            if( testAccuracy )
            {
                bfio::lnuft::WriteVtkXmlPImageData
                ( comm, N, targetBox, *v, "nuft2d", globalSources );
            }
            else
            {
                bfio::lnuft::WriteVtkXmlPImageData
                ( comm, N, targetBox, *v, "nuft2d" );
            }
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
