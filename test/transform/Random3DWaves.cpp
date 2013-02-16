/*
   Copyright (C) 2010-2013 Jack Poulson and Lexing Ying
 
   This file is part of ButterflyFIO and is under the GNU General Public 
   License, which can be found in the LICENSE file in the root directory, or at
   <http://www.gnu.org/licenses/>.
*/
#include "bfio.hpp"
using namespace std;
using namespace bfio;

void 
Usage()
{
    cout << "Random3DWaves <N> <M> <bootstrap> <T> <nT>\n" 
         << "  N: power of 2, the source spread in each dimension\n" 
         << "  M: number of random sources to instantiate\n" 
         << "  bootstrap: level to bootstrap to\n"
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
    R t_;
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
    R t_;
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
: t_(0) 
{ }

template<typename R>
inline
DownWave<R>::DownWave() 
: t_(0) 
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
{ t_ = t; }

template<typename R>
inline void 
DownWave<R>::SetTime( const R t ) 
{ t_ = t; }

template<typename R>
inline R 
UpWave<R>::GetTime() const 
{ return t_; }

template<typename R>
inline R 
DownWave<R>::GetTime() const 
{ return t_; }

template<typename R>
inline R
UpWave<R>::operator()( const array<R,d>& x, const array<R,d>& p ) const
{ 
    return TwoPi<R>()*( 
             x[0]*p[0]+x[1]*p[1]+x[2]*p[2] + 
             t_*sqrt(p[0]*p[0]+p[1]*p[1]+p[2]*p[2])
           );
}

template<typename R>
inline R
DownWave<R>::operator()( const array<R,d>& x, const array<R,d>& p ) const
{ 
    return TwoPi<R>()*(
             x[0]*p[0]+x[1]*p[1]+x[2]*p[2] -
             t_*sqrt(p[0]*p[0]+p[1]*p[1]+p[2]*p[2])
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

    // Scale the square roots by t_
    for( int j=0; j<pSize; ++j )
        sqrtResults[j] *= t_;

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

template<typename R>
void
DownWave<R>::BatchEvaluate
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

    // Scale the square roots by -t_
    for( int j=0; j<pSize; ++j )
        sqrtResults[j] *= -t_;

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
    const double T = atof(argv[4]);
    const size_t nT = atoi(argv[5]);

    try 
    {
        // Define the source and target boxes
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
                << "distributed amongst " << numProcesses << " processes. "
                << "The simulation will be over " << T << " units of time with "
                << nT << " timesteps.\n";
            cout << msg.str() << endl;
        }

        // Consistently seed all of the processes' PRNGs
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

        // Now generate random sources in our frequency box
        size_t numLocalSources = 
            ( rank<(int)(M%numProcesses) 
              ? M/numProcesses+1 : M/numProcesses );
        vector<Source<double,d>> mySources( numLocalSources );
        for( size_t i=0; i<numLocalSources; ++i )
        {
            for( size_t j=0; j<d; ++j )
                mySources[i].p[j]=mySBox.offsets[j]+mySBox.widths[j]*uniform();
            mySources[i].magnitude = 200*uniform()-100;
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
            auto u = RFIO( context, plan, upWave, sBox, tBox, mySources );
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
            auto v = RFIO( context, plan, downWave, sBox, tBox, mySources );
            if( rank == 0 )
                cout << "done" << endl;
#ifdef TIMING
            if( rank == 0 )
                rfio::PrintTimings();
#endif

            // Store this timeslice
            ostringstream fileStream;
            fileStream << "randomWaves-" << i;
            rfio::WriteImage( comm, N, tBox, *u, fileStream.str() );
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
