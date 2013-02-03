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
using namespace bfio;

void 
Usage()
{
    cout << "VariableUpWave-2d <N> <M> <bootstrap> <testAccuracy?> <store?>\n"
         << "  N: power of 2, the source spread in each dimension\n" 
         << "  M: number of random sources to instantiate\n" 
         << "  bootstrap: level to bootstrap to\n"
         << "  testAccuracy?: test accuracy iff 1\n" 
         << "  store?: create data files iff 1\n" 
         << endl;
}

// Define the dimension of the problem and the order of interpolation
static const size_t d = 2;
static const size_t q = 12;

template<typename R>
class Oscillatory : public Amplitude<R,d>
{
public:
    virtual Oscillatory<R>* Clone() const override;

    virtual complex<R>
    operator()( const array<R,d>& x, const array<R,d>& p ) const override;

    // We can optionally override the batched application for better efficiency
    virtual void
    BatchEvaluate
    ( const vector<array<R,d>>& xPoints,
      const vector<array<R,d>>& pPoints,
            vector<complex<R>>& results ) const override;
};

template<typename R>
class UpWave : public Phase<R,d>
{
public:
    virtual UpWave<R>* Clone() const override;

    virtual R 
    operator()( const array<R,d>& x, const array<R,d>& p ) const override;

    // We can optionally override the batched application for better efficiency
    virtual void
    BatchEvaluate
    ( const vector<array<R,d>>& xPoints,
      const vector<array<R,d>>& pPoints,
            vector<R         >& results ) const override;
};

template<typename R>
inline Oscillatory<R>*
Oscillatory<R>::Clone() const
{ return new Oscillatory<R>(*this); }

template<typename R>
inline complex<R>
Oscillatory<R>::operator()( const array<R,d>& x, const array<R,d>& p ) const
{
    return 1. + 0.5*sin(1*Pi*x[0])*sin(4*Pi*x[1])*
                    cos(3*Pi*p[0])*cos(4*Pi*p[1]);
}

template<typename R>
void
Oscillatory<R>::BatchEvaluate
( const vector<array<R,d>>& xPoints,
  const vector<array<R,d>>& pPoints,
        vector<complex<R>>& results ) const
{
    const size_t xSize = xPoints.size();
    const size_t pSize = pPoints.size();

    // Set up the sin and cos arguments
    vector<R> sinArguments( d*xSize );
    {
        R* RESTRICT sinArgBuffer = &sinArguments[0];
        const R* RESTRICT xPointsBuffer = &(xPoints[0][0]);
        for( size_t i=0; i<xSize; ++i )
        {
            sinArgBuffer[i*d+0] =   Pi*xPointsBuffer[i*d+0];
            sinArgBuffer[i*d+1] = 4*Pi*xPointsBuffer[i*d+1];
        }
    }
    vector<R> cosArguments( d*pPoints.size() );
    {
        R* RESTRICT cosArgBuffer = &cosArguments[0];
        const R* RESTRICT pPointsBuffer = &(pPoints[0][0]);
        for( size_t j=0; j<pSize; ++j )
        {
            cosArgBuffer[j*d+0] = 3*Pi*pPointsBuffer[j*d+0];
            cosArgBuffer[j*d+1] = 4*Pi*pPointsBuffer[j*d+1];
        }
    }

    // Call the vector sin and cos
    vector<R> sinResults, cosResults;
    SinBatch( sinArguments, sinResults );
    CosBatch( cosArguments, cosResults );

    // Form the x and p coefficients
    vector<R> xCoefficients( xSize ); 
    vector<R> pCoefficients( pSize );
    {
        R* RESTRICT xCoefficientsBuffer = &xCoefficients[0];
        const R* RESTRICT sinBuffer = &sinResults[0];
        for( size_t i=0; i<xSize; ++i )
            xCoefficientsBuffer[i] = 0.5*sinBuffer[i*d]*sinBuffer[i*d+1];
    }
    {
        R* RESTRICT pCoefficientsBuffer = &pCoefficients[0];
        const R* RESTRICT cosBuffer = &cosResults[0];
        for( size_t j=0; j<pSize; ++j )
            pCoefficientsBuffer[j] = cosBuffer[j*d]*cosBuffer[j*d+1];
    }

    // Form the answer
    results.resize( xSize*pSize );
    {
        complex<R>* RESTRICT resultsBuffer = &results[0];
        const R* RESTRICT xCoefficientsBuffer = &xCoefficients[0];
        const R* RESTRICT pCoefficientsBuffer = &pCoefficients[0];
        for( size_t i=0; i<xSize; ++i )
            for( size_t j=0; j<pSize; ++j )
                resultsBuffer[i*pSize+j] = 
                    1. + xCoefficientsBuffer[i]*pCoefficientsBuffer[j];
    }
}

template<typename R>
inline UpWave<R>*
UpWave<R>::Clone() const
{ return new UpWave<R>(*this); }

template<typename R>
inline R
UpWave<R>::operator()( const array<R,d>& x, const array<R,d>& p ) const
{
    return TwoPi*(x[0]*p[0]+x[1]*p[1] + 0.5*sqrt(p[0]*p[0]+p[1]*p[1])); 
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
        R* sqrtArgBuffer = &sqrtArguments[0];
        const R* pPointsBuffer = &(pPoints[0][0]);
        for( size_t j=0; j<pSize; ++j )
            sqrtArgBuffer[j] = pPointsBuffer[j*d+0]*pPointsBuffer[j*d+0] +
                               pPointsBuffer[j*d+1]*pPointsBuffer[j*d+1];
    }

    // Perform the batched square roots
    vector<R> sqrtResults;
    SqrtBatch( sqrtArguments, sqrtResults );

    // Scale the square roots by 1/2
    {
        R* sqrtBuffer = &sqrtResults[0];
        for( size_t j=0; j<pSize; ++j )
            sqrtBuffer[j] *= 0.5;
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
    const bool testAccuracy = atoi(argv[4]);
    const bool store = atoi(argv[5]);

    try
    {
        // Set the source and target boxes
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
        vector<Source<double,d>> mySources, globalSources;
        if( testAccuracy || store )
        {
            globalSources.resize( M );
            for( size_t i=0; i<M; ++i )
            {
                for( size_t j=0; j<d; ++j )
                {
                    globalSources[i].p[j] = sourceBox.offsets[j] + 
                        sourceBox.widths[j]*Uniform<double>(); 
                }
                globalSources[i].magnitude = 10*(2*Uniform<double>()-1); 

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
                        mySourceBox.offsets[j]+
                        Uniform<double>()*mySourceBox.widths[j];
                }
                mySources[i].magnitude = 10*(2*Uniform<double>()-1);
            }
        }

        // Set up our amplitude and phase functors
        Oscillatory<double> oscillatory;
        UpWave<double> upWave;

        // Create our context
        if( rank == 0 )
            cout << "Creating context..." << endl;
        rfio::Context<double,d,q> context;

        // Run the algorithm
        if( rank == 0 )
            cout << "Starting transform..." << endl;
        MPI_Barrier( comm );
        double startTime = MPI_Wtime();
        auto u = RFIO
        ( context, plan, oscillatory, upWave, sourceBox, targetBox, mySources );
        MPI_Barrier( comm );
        double stopTime = MPI_Wtime();
        if( rank == 0 )
            cout << "Runtime: " << stopTime-startTime << " seconds.\n" << endl;
#ifdef TIMING
        if( rank == 0 )
            rfio::PrintTimings();
#endif

        if( testAccuracy )
            rfio::PrintErrorEstimates( comm, *u, globalSources );
        
        if( store )
        {
            if( testAccuracy )
                rfio::WriteImage
                ( comm, N, targetBox, *u, "varUpWave2d", globalSources );
            else
                rfio::WriteImage( comm, N, targetBox, *u, "varUpWave2d" );
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
