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
    cout << "HypRadon-2d <N> <F> <M> <bootstrap> <testAccuracy?> <store?>\n" 
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
static const size_t q = 4;

template<typename R>    
class HypRadon : public Phase<R,d>
{
public:
    virtual HypRadon<R>* Clone() const override;

    virtual R operator()
    ( const array<R,d>& x, const array<R,d>& p ) const override;

    // We can optionally override the batched application for better efficiency.
    virtual void BatchEvaluate
    ( const vector<array<R,d>>& xPoints,
      const vector<array<R,d>>& pPoints,
            vector<R         >& results ) const override;
};

template<typename R>
inline HypRadon<R>* 
HypRadon<R>::Clone() const
{ return new HypRadon<R>; }

template<typename R>
inline R HypRadon<R>::operator()
( const array<R,d>& x, const array<R,d>& p ) const
{
    return TwoPi<R>()*p[0]*sqrt(x[0]*x[0]+x[1]*x[1]*p[1]*p[1]);
}

template<typename R>
void HypRadon<R>::BatchEvaluate
( const vector<array<R,d>>& xPoints,
  const vector<array<R,d>>& pPoints,
        vector<R         >& results ) const
{
    const int xSize = xPoints.size();
    const int pSize = pPoints.size();

    // Compute all of the sin's and cos's of the x indices times TwoPi 
    static vector<R> sqrtArguments;
    sqrtArguments.resize( xSize*pSize );
    for( int i=0; i<xSize; ++i )
    {
        const R x0Squared = xPoints[i][0]*xPoints[i][0];
        const R x1Squared = xPoints[i][1]*xPoints[i][1];
        for( int j=0; j<pSize; ++j )
        {
            const R p1Squared = pPoints[j][1]*pPoints[j][1];
            sqrtArguments[i*pSize+j] = x0Squared+x1Squared*p1Squared;
        }
    }
    SqrtBatch( sqrtArguments, results );

    // Scale result (x,p) by p(0)
    const R twoPi = TwoPi<R>();
    for( int i=0; i<xSize; ++i )
        for( int j=0; j<pSize; ++j )
            results[i*pSize+j] *= twoPi*pPoints[j][0];
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
        Box<double,d> sBox, tBox;
        for( size_t j=0; j<d; ++j )
        {
            sBox.offsets[j] = -0.5*(sqrt(N)/F);
            sBox.widths[j] = sqrt(N)/F;
            tBox.offsets[j] = 0;
            tBox.widths[j] = 1;
        }

        // Set up the general strategy for the forward transform
        Plan<d> plan( comm, FORWARD, N, bootstrap );
        Box<double,d> mySBox = plan.GetMyInitialSourceBox( sBox );;

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
        uniform_real_distribution<double> uniform_dist(0.,1.);
        auto uniform = bind( uniform_dist, ref(engine) );

        // Now generate random sources across the domain and store them in 
        // our local list when appropriate
        vector<Source<double,d>> mySources, sources;
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
                    double u = sources[i].p[j];
                    double start = mySBox.offsets[j];
                    double stop = mySBox.offsets[j] + mySBox.widths[j];
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
        HypRadon<double> hypRadon;
        bfly::Context<double,d,q> context;
        if( rank == 0 )
            cout << "done." << endl;

        // Run the algorithm to generate the potential field
        if( rank == 0 )
            cout << "Launching transform..." << endl;
        MPI_Barrier( comm );
        double startTime = MPI_Wtime();
        auto u = Butterfly( context, plan, hypRadon, sBox, tBox, mySources );
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
                bfly::WriteImage( comm, N, tBox, *u, "hypRadon2d", sources );
            else
                bfly::WriteImage( comm, N, tBox, *u, "hypRadon2d" );
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
