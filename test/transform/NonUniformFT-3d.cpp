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

void 
Usage()
{
    std::cout << "NonUniformFT-3d <N> <M> <bootstrap> <testAccuracy?> <store?>"
              << "\n" 
              << "  N: power of 2, the source spread in each dimension\n" 
              << "  M: number of random sources to instantiate\n" 
              << "  bootstrap: level to bootstrap to\n"
              << "  testAccuracy?: tests accuracy iff 1\n" 
              << "  store?: creates data files iff 1\n" 
              << std::endl;
}

// Define the dimension of the problem and the order of interpolation
static const std::size_t d = 3;
static const std::size_t q = 5;

template<typename R>
class Fourier : public bfio::Phase<R,d>
{
public:
    virtual Fourier<R>* Clone() const;

    virtual R 
    operator()
    ( const bfio::Array<R,d>& x, const bfio::Array<R,d>& p ) const;

    // We can optionally override the batched application for better efficiency
    virtual void
    BatchEvaluate
    ( const std::vector< bfio::Array<R,d> >& xPoints,
      const std::vector< bfio::Array<R,d> >& pPoints,
            std::vector< R                >& results ) const;
};

template<typename R>
inline Fourier<R>*
Fourier<R>::Clone() const
{ return new Fourier<R>(*this); }

template<typename R>
inline R
Fourier<R>::operator() 
( const bfio::Array<R,d>& x, const bfio::Array<R,d>& p ) const
{ return -bfio::TwoPi*(x[0]*p[0]+x[1]*p[1]+x[2]*p[2]); }

// We can optionally override the batched application for better efficiency
template<typename R>
void
Fourier<R>::BatchEvaluate
( const std::vector< bfio::Array<R,d> >& xPoints,
  const std::vector< bfio::Array<R,d> >& pPoints,
        std::vector< R                >& results ) const
{
    const std::size_t xSize = xPoints.size();
    const std::size_t pSize = pPoints.size();
    results.resize( xSize*pSize );

    R* RESTRICT resultsBuffer = &results[0];
    const R* RESTRICT xPointsBuffer = &(xPoints[0][0]);
    const R* RESTRICT pPointsBuffer = &(pPoints[0][0]);
    for( std::size_t i=0; i<xSize; ++i )
    {
        for( std::size_t j=0; j<pSize; ++j )
        {
            resultsBuffer[i*pSize+j] = 
                xPointsBuffer[i*d+0]*pPointsBuffer[j*d+0] + 
                xPointsBuffer[i*d+1]*pPointsBuffer[j*d+1] + 
                xPointsBuffer[i*d+2]*pPointsBuffer[j*d+2];
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
    const std::size_t N = atoi(argv[1]);
    const std::size_t M = atoi(argv[2]);
    const std::size_t bootstrapSkip = atoi(argv[3]);
    const bool testAccuracy = atoi(argv[4]);
    const bool store = atoi(argv[5]);

    try 
    {
        // Set our source and target boxes
        bfio::Box<double,d> sourceBox, targetBox;
        for( std::size_t j=0; j<d; ++j )
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
            std::ostringstream msg;
            msg << "Will distribute " << M << " random sources over the source "
                << "domain, which will be split into " << N 
                << " boxes in each of the " << d << " dimensions and "
                << "distributed amongst " << numProcesses << " processes.\n";
            std::cout << msg.str() << std::endl;
        }

        // Consistently randomly seed all of the processes' PRNG.
        long seed;
        if( rank == 0 )
            seed = time(0);
        MPI_Bcast( &seed, 1, MPI_LONG, 0, comm );
        srand( seed );

        // Now generate random sources across the domain and store them in 
        // our local list when appropriate
        std::vector< bfio::Source<double,d> > mySources;
        std::vector< bfio::Source<double,d> > globalSources;
        if( testAccuracy || store )
        {
            globalSources.resize( M );
            for( std::size_t i=0; i<M; ++i )
            {
                for( std::size_t j=0; j<d; ++j )
                {
                    globalSources[i].p[j] = sourceBox.offsets[j] + 
                        sourceBox.widths[j]*bfio::Uniform<double>(); 
                }
                globalSources[i].magnitude = 1.*(2*bfio::Uniform<double>()-1); 

                // Check if we should push this source onto our local list
                bool isMine = true;
                for( std::size_t j=0; j<d; ++j )
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
            std::size_t numLocalSources = 
                ( rank<(int)(M%numProcesses) 
                  ? M/numProcesses+1 : M/numProcesses );
            mySources.resize( numLocalSources );
            for( std::size_t i=0; i<numLocalSources; ++i )
            {
                for( std::size_t j=0; j<d; ++j )
                {
                    mySources[i].p[j] = 
                        mySourceBox.offsets[j] + 
                        bfio::Uniform<double>()*mySourceBox.widths[j];
                }
                mySources[i].magnitude = 1.*(2*bfio::Uniform<double>()-1);
            }
        }

        /*
        // Create a context for Interpolative NUFTs
        if( rank == 0 )
            std::cout << "Creating InterpolativeNUFT context..." << std::endl;
        bfio::interpolative_nuft::Context<double,d,q> 
            interpolativeNuftContext( bfio::FORWARD, N, sourceBox, targetBox );

        // Run with the interpolative NUFT
        std::auto_ptr< 
            const bfio::interpolative_nuft::PotentialField<double,d,q> > u;
        if( rank == 0 )
            std::cout << "Starting InterpolativeNUFT..." << std::endl;
        MPI_Barrier( comm );
        double startTime = MPI_Wtime();
        u = bfio::InterpolativeNUFT
        ( interpolativeNuftContext, plan, sourceBox, targetBox, mySources );
        MPI_Barrier( comm );
        double stopTime = MPI_Wtime();
        if( rank == 0 )
        {
            std::cout << "Runtime: " << stopTime-startTime << " seconds.\n"
                      << std::endl;
        }
#ifdef TIMING
	if( rank == 0 )
	    bfio::interpolative_nuft::PrintTimings();
#endif
        */

        // Create a context for NUFTs with Lagrangian interpolation
        if( rank == 0 )
            std::cout << "Creating LagrangianNUFT context..." << std::endl;
        bfio::lagrangian_nuft::Context<double,d,q> 
            lagrangianNuftContext( bfio::FORWARD, N, sourceBox, targetBox );

        // Run with the Lagrangian NUFT
        std::auto_ptr< const bfio::lagrangian_nuft::PotentialField<double,d,q> >
            v;
        if( rank == 0 )
            std::cout << "Starting LagrangianNUFT..." << std::endl;
        MPI_Barrier( comm );
        //startTime = MPI_Wtime();
        double startTime = MPI_Wtime();
        v = bfio::LagrangianNUFT
        ( lagrangianNuftContext, plan, sourceBox, targetBox, mySources );
        MPI_Barrier( comm );
        //stopTime = MPI_Wtime();
        double stopTime = MPI_Wtime();
        if( rank == 0 )
        {
            std::cout << "Runtime: " << stopTime-startTime << " seconds.\n"
                      << std::endl;
        }
#ifdef TIMING
	if( rank == 0 )
	    bfio::lagrangian_nuft::PrintTimings();
#endif

        /*
        // Set up our phase functor
        Fourier<double> fourier;

        // Create an FIO context 
        if( rank == 0 )
            std::cout << "Creating ReducedFIO context..." << std::endl;
        bfio::rfio::Context<double,d,q> rfioContext;

        // Run the general algorithm
        std::auto_ptr< const bfio::rfio::PotentialField<double,d,q> > w;
        if( rank == 0 )
            std::cout << "Starting ReducedFIO transform..." << std::endl;
        MPI_Barrier( comm );
        startTime = MPI_Wtime();
        w = bfio::ReducedFIO
        ( rfioContext, plan, fourier, sourceBox, targetBox, mySources );
        MPI_Barrier( comm );
        stopTime = MPI_Wtime();
        if( rank == 0 )
        {
            std::cout << "Runtime: " << stopTime-startTime << " seconds.\n" 
                      << std::endl;
        }
#ifdef TIMING
	if( rank == 0 )
	    bfio::rfio::PrintTimings();
#endif
        */

        if( testAccuracy )
        {
            bfio::lagrangian_nuft::PrintErrorEstimates
            ( comm, *v, globalSources );
        }
        
        if( store )
        {
            if( testAccuracy )
            {
                bfio::lagrangian_nuft::WriteVtkXmlPImageData
                ( comm, N, targetBox, *v, "nuft3d", globalSources );
            }
            else
            {
                bfio::lagrangian_nuft::WriteVtkXmlPImageData
                ( comm, N, targetBox, *v, "nuft3d" );
            }
        }
    }
    catch( const std::exception& e )
    {
        std::ostringstream msg;
        msg << "Caught exception on process " << rank << ":\n"
            << "   " << e.what();
        std::cout << msg.str() << std::endl;
    }

    MPI_Finalize();
    return 0;
}

