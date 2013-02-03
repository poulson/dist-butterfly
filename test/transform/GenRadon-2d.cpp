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
    cout << "GenRadon-2d <N> <F> <M> <bootstrap> <testAccuracy?> <store?>\n" 
         << "  N: power of 2, the number of boxes in each dimension\n" 
         << "  F: power of 2, boxes per unit length in each source dim\n"
         << "  M: number of random sources to instantiate\n" 
         << "  bootstrap: level to bootstrap to\n"
         << "  testAccuracy?: tests accuracy iff 1\n" 
         << "  store?: creates data files iff 1\n" 
         << endl;
}

// Define the dimension of the problem and the order of interpolation
static const size_t d = 2;
static const size_t q = 8;

template<typename R>    
class GenRadon : public Phase<R,d>
{
public:
    virtual GenRadon<R>* Clone() const override;

    virtual R operator()
    ( const array<R,d>& x, const array<R,d>& p ) const override;

    // We can optionally override the batched application for better efficiency.
    virtual void BatchEvaluate
    ( const vector<array<R,d>>& xPoints,
      const vector<array<R,d>>& pPoints,
            vector<R         >& results ) const override;
};

template<typename R>
inline GenRadon<R>* 
GenRadon<R>::Clone() const
{ return new GenRadon<R>(*this); }

template<typename R>
inline R GenRadon<R>::operator()
( const array<R,d>& x, const array<R,d>& p ) const
{
    R a = p[0]*(2+sin(TwoPi*x[0])*sin(TwoPi*x[1]))/3.;
    R b = p[1]*(2+cos(TwoPi*x[0])*cos(TwoPi*x[1]))/3.;
    return TwoPi*(x[0]*p[0]+x[1]*p[1] + sqrt(a*a+b*b));
}

template<typename R>
void GenRadon<R>::BatchEvaluate
( const vector<array<R,d>>& xPoints,
  const vector<array<R,d>>& pPoints,
        vector<R         >& results ) const
{
    const size_t xSize = xPoints.size();
    const size_t pSize = pPoints.size();

    // Compute all of the sin's and cos's of the x indices times TwoPi 
    vector<R> sinCosArguments( d*xSize );
    {
        R* RESTRICT sinCosArgBuffer = &sinCosArguments[0];
        const R* RESTRICT xPointsBuffer = 
            static_cast<const R*>(&(xPoints[0][0]));
        for( size_t i=0; i<d*xSize; ++i )
            sinCosArgBuffer[i] = TwoPi*xPointsBuffer[i];
    }
    vector<R> sinResults, cosResults;
    SinCosBatch( sinCosArguments, sinResults, cosResults );

    // Compute the the c1(x) and c2(x) results for every x vector
    vector<R> c1( xSize ), c2( xSize );
    {
        R* RESTRICT c1Buffer = &c1[0];
        const R* RESTRICT sinBuffer = &sinResults[0];
        for( size_t i=0; i<xSize; ++i )
            c1Buffer[i] = (2+sinBuffer[i*d]*sinBuffer[i*d+1])/3;
    }
    {
        R* RESTRICT c2Buffer = &c2[0];
        const R* RESTRICT cosBuffer = &cosResults[0];
        for( size_t i=0; i<xSize; ++i )
            c2Buffer[i] = (2+cosBuffer[i*d]*cosBuffer[i*d+1])/3;
    }

    // Form the set of sqrt arguments
    vector<R> sqrtArguments( xSize*pSize );
    {
        R* RESTRICT sqrtArgBuffer = &sqrtArguments[0];
        const R* RESTRICT c1Buffer = &c1[0];
        const R* RESTRICT c2Buffer = &c2[0];
        const R* RESTRICT pPointsBuffer = &(pPoints[0][0]);
        for( size_t i=0; i<xSize; ++i )
        {
            for( size_t j=0; j<pSize; ++j )
            {
                const R a = c1Buffer[i]*pPointsBuffer[j*d+0];
                const R b = c2Buffer[i]*pPointsBuffer[j*d+1];
                sqrtArgBuffer[i*pSize+j] = a*a+b*b;
            }
        }
    }

    // Perform the batched square roots
    vector<R> sqrtResults;
    SqrtBatch( sqrtArguments, sqrtResults );

    // Form the answer
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
                    sqrtBuffer[i*pSize+j];
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

    if( argc != 7 )
    {
        if( rank == 0 )
            Usage();
        MPI_Finalize();
        return 0;
    }
    int argNum = 0;
    const size_t N = atoi(argv[++argNum]);
    const size_t F = atoi(argv[++argNum]);
    const size_t M = atoi(argv[++argNum]);
    const size_t bootstrapSkip = atoi(argv[++argNum]);
    const bool testAccuracy = atoi(argv[++argNum]);
    const bool store = atoi(argv[++argNum]);

    try 
    {
        // Set our source and target boxes
        Box<float,d> sBox, tBox;
        for( size_t j=0; j<d; ++j )
        {
            sBox.offsets[j] = -0.5*(N/F);
            sBox.widths[j] = (N/F);
            tBox.offsets[j] = 0;
            tBox.widths[j] = 1;
        }

        // Set up the general strategy for the forward transform
        Plan<d> plan( comm, FORWARD, N, bootstrapSkip );
        Box<float,d> mySBox = plan.GetMyInitialSourceBox( sBox );;

        if( rank == 0 )
        {
            ostringstream msg;
            msg << "Will distribute " << M << " random sources over the "
                << "source domain, which will be split into " << N 
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
        vector<Source<float,d>> mySources, sources;
        if( testAccuracy || store )
        {
            sources.resize( M );
            for( size_t i=0; i<M; ++i )
            {
                for( size_t j=0; j<d; ++j )
                {
                    const float relPos = Uniform<float>();
                    sources[i].p[j] = sBox.offsets[j] + sBox.widths[j]*relPos;
                }
                sources[i].magnitude = 1.*(2*Uniform<float>()-1); 

                // Check if we should push this source onto our local list
                bool isMine = true;
                for( size_t j=0; j<d; ++j )
                {
                    float u = sources[i].p[j];
                    float start = mySBox.offsets[j];
                    float stop = mySBox.offsets[j] + mySBox.widths[j];
                    if( u < start || u >= stop )
                        isMine = false;
                }
                if( isMine )
                    mySources.push_back( sources[i] );
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
                    mySources[i].p[j] = 
                        mySBox.offsets[j] + Uniform<float>()*mySBox.widths[j];
                mySources[i].magnitude = 1.*(2*Uniform<float>()-1);
            }
        }

        // Create the phase functor and the context that takes care of all of 
        // the precomputation
        if( rank == 0 )
        {
            cout << "Creating context...";
            cout.flush();
        }
        GenRadon<float> genRadon;
        rfio::Context<float,d,q> context;
        if( rank == 0 )
            cout << "done." << endl;

        // Run the algorithm to generate the potential field
        if( rank == 0 )
            cout << "Launching transform..." << endl;
        MPI_Barrier( comm );
        double startTime = MPI_Wtime();
        auto u = RFIO( context, plan, genRadon, sBox, tBox, mySources );
        MPI_Barrier( comm );
        double stopTime = MPI_Wtime();
        if( rank == 0 )
            cout << "Runtime: " << stopTime-startTime << " seconds.\n" << endl;
#ifdef TIMING
        if( rank == 0 )
            rfio::PrintTimings();
#endif

        if( testAccuracy )
            rfio::PrintErrorEstimates( comm, *u, sources );
        
        if( store )
        {
            if( testAccuracy )
                rfio::WriteImage( comm, N, tBox, *u, "genRadon2d", sources );
            else
                rfio::WriteImage( comm, N, tBox, *u, "genRadon2d" );
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

