/*
   Copyright (C) 2010-2013 Jack Poulson and Lexing Ying
 
   This file is part of DistButterfly and is under the GNU General Public 
   License, which can be found in the LICENSE file in the root directory, or at
   <http://www.gnu.org/licenses/>.
*/
#include "dist-butterfly.hpp"
using namespace std;
using namespace dbf;

void 
Usage()
{
    cout << "Backproj-2d <N> <F> <M> <bootstrap> <testAccuracy?> <store?>\n" 
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
class Backproj : public Phase<R,d>
{
public:
    virtual Backproj<R>* Clone() const override;

    virtual R operator()
    ( const array<R,d>& x, const array<R,d>& p ) const override;

    // We can optionally override the batched application for better efficiency.
    virtual void BatchEvaluate
    ( const vector<array<R,d>>& xPoints,
      const vector<array<R,d>>& pPoints,
            vector<R         >& results ) const override;
};

template<typename R>
inline Backproj<R>* 
Backproj<R>::Clone() const
{ return new Backproj<R>(*this); }

template<typename R>
inline R Backproj<R>::operator()
( const array<R,d>& x, const array<R,d>& p ) const
{
    return TwoPi<R>()*p[1]*(cos(p[0])*x[0]+sin(p[0])*x[1]);
}

template<typename R>
void Backproj<R>::BatchEvaluate
( const vector<array<R,d>>& xPoints,
  const vector<array<R,d>>& pPoints,
        vector<R         >& results ) const
{
    const R twoPi = TwoPi<R>();
    const size_t xSize = xPoints.size();
    const size_t pSize = pPoints.size();

    // Compute all of the sin's and cos's of the x indices times TwoPi 
    static vector<R> sinCosArguments;
    sinCosArguments.resize( pSize );
    for( size_t j=0; j<pSize; ++j )
        sinCosArguments[j] = pPoints[j][0];
    static vector<R> sinResults, cosResults;
    SinCosBatch( sinCosArguments, sinResults, cosResults );

    // Form the answer
    results.resize( xSize*pSize );
    for( size_t i=0; i<xSize; ++i )
    {
        const R x0 = xPoints[i][0];
        const R x1 = xPoints[i][1];
        for( size_t j=0; j<pSize; ++j )
        {
            results[i*pSize+j] = cosResults[j]*x0+sinResults[j]*x1;
            results[i*pSize+j] *= twoPi*pPoints[j][1];
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
    const size_t bootstrap = atoi(argv[++argNum]);
    const bool testAccuracy = atoi(argv[++argNum]);
    const bool store = atoi(argv[++argNum]);

    try 
    {
        // Set our source and target boxes
        Box<float,d> sBox, tBox;
        sBox.offsets[0] = 0;
        sBox.widths[0] = TwoPi<float>();
        sBox.offsets[1] = -0.5*(N/F);
        sBox.widths[1] = N/F;
        for( size_t j=0; j<d; ++j )
        {
            tBox.offsets[j] = 0;
            tBox.widths[j] = 1;
        }

        // Set up the general strategy for the forward transform
        Plan<d> plan( comm, FORWARD, N, bootstrap );
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
        unsigned seed;
        if( rank == 0 )
        {
            random_device rd;
            seed = rd();
        }
        MPI_Bcast( &seed, 1, MPI_UNSIGNED, 0, comm );
        default_random_engine engine( seed );
        uniform_real_distribution<float> uniform_dist(0.f,1.f);
        auto uniform = bind( uniform_dist, ref(engine) );

        // Now generate random sources across the domain and store them in 
        // our local list when appropriate
        vector<Source<float,d>> mySources, sources;
        if( testAccuracy || store )
        {
            sources.resize( M );
            for( size_t i=0; i<M; ++i )
            {
                for( size_t j=0; j<d; ++j )
                    sources[i].p[j] = sBox.offsets[j]+sBox.widths[j]*uniform();
                sources[i].magnitude = 1.*(2*uniform()-1); 

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
                ( rank<int(M%numProcesses) 
                  ? M/numProcesses+1 : M/numProcesses );
            mySources.resize( numLocalSources );
            for( size_t i=0; i<numLocalSources; ++i )
            {
                for( size_t j=0; j<d; ++j )
                    mySources[i].p[j] = 
                        mySBox.offsets[j] + mySBox.widths[j]*uniform();
                mySources[i].magnitude = 1.*(2*uniform()-1);
            }
        }

        // Create the phase functor and the context that takes care of all of 
        // the precomputation
        if( rank == 0 )
        {
            cout << "Creating context...";
            cout.flush();
        }
        Backproj<float> backproj;
        bfly::Context<float,d,q> context;
        if( rank == 0 )
            cout << "done." << endl;

        // Run the algorithm to generate the potential field
        if( rank == 0 )
            cout << "Launching transform..." << endl;
        MPI_Barrier( comm );
        double startTime = MPI_Wtime();
        auto u = Butterfly( context, plan, backproj, sBox, tBox, mySources );
        MPI_Barrier( comm );
        double stopTime = MPI_Wtime();
        if( rank == 0 )
            cout << "Runtime: " << stopTime-startTime << " seconds.\n" << endl;
#ifdef TIMING
        if( rank == 0 )
            bfly::PrintTimings();
#endif

        if( testAccuracy )
            bfly::PrintErrorEstimates( comm, *u, sources );
        
        if( store )
        {
            if( testAccuracy )
                bfly::WriteImage( comm, N, tBox, *u, "backproj2d", sources );
            else
                bfly::WriteImage( comm, N, tBox, *u, "backproj2d" );
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
