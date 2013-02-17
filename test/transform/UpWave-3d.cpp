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
    cout << "UpWave-3d <N> <M> <bootstrap> <testAccuracy?> <store?>\n" 
         << "  N: power of 2, the source spread in each dimension\n" 
         << "  M: number of random sources to instantiate\n" 
         << "  bootstrap: level to bootstrap to\n"
         << "  testAccuracy?: test accuracy iff 1\n" 
         << "  store?: create data files iff 1\n" 
         << endl;
}

// Define the dimension of the problem and the order of interpolation
static const size_t d = 3;
static const size_t q = 8;

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
inline UpWave<R>*
UpWave<R>::Clone() const
{ return new UpWave<R>(*this); }

template<typename R>
inline R
UpWave<R>::operator()( const array<R,d>& x, const array<R,d>& p ) const
{
    return TwoPi<R>()*( 
             x[0]*p[0]+x[1]*p[1]+x[2]*p[2] + 
             sqrt(p[0]*p[0]+p[1]*p[1]+p[2]*p[2])/R(2)
           ); 
}

template<typename R>
void
UpWave<R>::BatchEvaluate
( const vector<array<R,d>>& xPoints,
  const vector<array<R,d>>& pPoints,
        vector<R         >& results ) const
{
    const R twoPi = TwoPi<R>();
    const int xSize = xPoints.size();
    const int pSize = pPoints.size();

    // Set up the square root arguments 
    vector<R> sqrtArguments( pSize );
    for( int j=0; j<pSize; ++j )
        sqrtArguments[j] = pPoints[j][0]*pPoints[j][0] +
                           pPoints[j][1]*pPoints[j][1] +
                           pPoints[j][2]*pPoints[j][2];

    // Perform the batched square roots
    vector<R> sqrtResults;
    SqrtBatch( sqrtArguments, sqrtResults );

    // Scale the square roots by 1/2
    for( int j=0; j<pSize; ++j )
        sqrtResults[j] /= R(2);

    // Form the final results
    results.resize( xSize*pSize );
    for( int i=0; i<xSize; ++i )
    {
        const R x0 = xPoints[i][0];
        const R x1 = xPoints[i][1];
        const R x2 = xPoints[i][2];
        for( int j=0; j<pSize; ++j )
            results[i*pSize+j] =
                twoPi*(x0*pPoints[j][0]+x1*pPoints[j][1]+x2*pPoints[j][2] + 
                       sqrtResults[j]);
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
    const size_t bootstrap = atoi(argv[3]);
    const bool testAccuracy = atoi(argv[4]);
    const bool store = atoi(argv[5]);

    try
    {
        // Set the source and target boxes
        Box<double,d> sBox, tBox;
        for( size_t j=0; j<d; ++j )
        {
            sBox.offsets[j] = -0.5*N;
            sBox.widths[j] = N;
            tBox.offsets[j] = 0;
            tBox.widths[j] = 1;
        }

        // Set up the general strategy for the forward transform
        Plan<d> plan( comm, FORWARD, N, bootstrap );
        Box<double,d> mySBox = plan.GetMyInitialSourceBox( sBox );

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
                sources[i].magnitude = 10*(2*uniform()-1); 

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
                ( rank<(int)(M%numProcesses) 
                  ? M/numProcesses+1 : M/numProcesses );
            mySources.resize( numLocalSources ); 
            for( size_t i=0; i<numLocalSources; ++i )
            {
                for( size_t j=0; j<d; ++j )
                    mySources[i].p[j] = 
                        mySBox.offsets[j] + mySBox.widths[j]*uniform();
                mySources[i].magnitude = 10*(2*uniform()-1);
            }
        }

        // Set up our phase functor
        UpWave<double> upWave;

        // Create the context 
        if( rank == 0 )
            cout << "Creating context..." << endl;
        bfly::Context<double,d,q> context;

        // Run the algorithm
        if( rank == 0 )
            cout << "Starting transform..." << endl;
        MPI_Barrier( comm );
        double startTime = MPI_Wtime();
        auto u = Butterfly( context, plan, upWave, sBox, tBox, mySources );
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
                bfly::WriteImage( comm, N, tBox, *u, "upWave3d", sources );
            else
                bfly::WriteImage( comm, N, tBox, *u, "upWave3d" );
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
