/*
   ButterflyFIO: a distributed-memory fast algorithm for applying FIOs.
   Copyright (C) 2010-2011 Jack Poulson <jack.poulson@gmail.com>
 
   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.
 
   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.
 
   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/
#include <ctime>
#include <fstream>
#include <memory>
#include "bfio.hpp"
using namespace std;

void 
Usage()
{
    cout << "GenRadon-3d <N> <F> <M> <testAccuracy?> <store?>\n" 
         << "  N: power of 2, the number of boxes in each dimension\n" 
         << "  F: power of 2, boxes per unit length in each source dim\n"
         << "  M: number of random sources to instantiate\n" 
         << "  bootstrap: level to bootstrap to\n"
         << "  testAccuracy?: tests accuracy iff 1\n" 
         << "  store?: creates data files iff 1\n" 
         << endl;
}

// Define the dimension of the problem and the order of interpolation
static const size_t d = 3;
static const size_t q = 8;

template<typename R>    
class GenRadon : public bfio::Phase<R,d>
{
public:
    virtual GenRadon<R>* Clone() const;

    virtual R operator()
    ( const bfio::Array<R,d>& x, const bfio::Array<R,d>& p ) const;

    // We can optionally override the batched application for better efficiency.
    virtual void BatchEvaluate
    ( const vector< bfio::Array<R,d> >& xPoints,
      const vector< bfio::Array<R,d> >& pPoints,
            vector< R                >& results ) const;
};

template<typename R>
inline GenRadon<R>*
GenRadon<R>::Clone() const
{ return new GenRadon<R>(*this); }

template<typename R>
inline R GenRadon<R>::operator()
( const bfio::Array<R,d>& x, const bfio::Array<R,d>& p ) const
{
    R a = p[0]*(2+sin(bfio::TwoPi*x[0])*sin(bfio::TwoPi*x[1]))/3;
    R b = p[1]*(2+cos(bfio::TwoPi*x[0])*cos(bfio::TwoPi*x[1]))/3;
    return bfio::TwoPi*(x[0]*p[0]+x[1]*p[1]+x[2]*p[2] + sqrt(a*a+b*b));
}

template<typename R>
void GenRadon<R>::BatchEvaluate
( const vector< bfio::Array<R,d> >& xPoints,
  const vector< bfio::Array<R,d> >& pPoints,
        vector< R                >& results ) const
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
            sinCosArgBuffer[i] = bfio::TwoPi*xPointsBuffer[i];
    }
    vector<R> sinResults;
    vector<R> cosResults;
    bfio::SinCosBatch( sinCosArguments, sinResults, cosResults );

    // Compute the the c1(x) and c2(x) results for every x vector
    vector<R> c1( xSize );
    vector<R> c2( xSize );
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
    bfio::SqrtBatch( sqrtArguments, sqrtResults );

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
                    xPointsBuffer[i*d+2]*pPointsBuffer[j*d+2] +
                    sqrtBuffer[i*pSize+j];
                resultsBuffer[i*pSize+j] *= bfio::TwoPi;
            }
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
        bfio::Box<double,d> sourceBox, targetBox;
        for( size_t j=0; j<d; ++j )
        {
            sourceBox.offsets[j] = -0.5*(N/F);
            sourceBox.widths[j] = (N/F);
            targetBox.offsets[j] = 0;
            targetBox.widths[j] = 1;
        }

        // Set up the general strategy for the forward transform
        bfio::Plan<d> plan( comm, bfio::FORWARD, N, bootstrapSkip );
        bfio::Box<double,d> mySourceBox = 
            plan.GetMyInitialSourceBox( sourceBox );;

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
        vector< bfio::Source<double,d> > mySources;
        vector< bfio::Source<double,d> > globalSources;
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

        // Create our phase functor
        GenRadon<double> genRadon;

        // Create the context that takes care of all of the precomputation
        if( rank == 0 )
        {
            cout << "Creating context...";
            cout.flush();
        }
        bfio::rfio::Context<double,d,q> context;
        if( rank == 0 )
            cout << "done." << endl;

        // Run the algorithm to generate the potential field
        auto_ptr< const bfio::rfio::PotentialField<double,d,q> > u;
        if( rank == 0 )
            cout << "Launching transform..." << endl;
        MPI_Barrier( comm );
        double startTime = MPI_Wtime();
        u = bfio::ReducedFIO
        ( context, plan, genRadon, sourceBox, targetBox, mySources );
        MPI_Barrier( comm );
        double stopTime = MPI_Wtime();
        if( rank == 0 )
            cout << "Runtime: " << stopTime-startTime << " seconds.\n" << endl;
#ifdef TIMING
        if( rank == 0 )
            bfio::rfio::PrintTimings();
#endif

        if( testAccuracy )
            bfio::rfio::PrintErrorEstimates( comm, *u, globalSources );
        
        if( store )
        {
            if( testAccuracy )
            {
                bfio::rfio::WriteVtkXmlPImageData
                ( comm, N, targetBox, *u, "genRadon3d", globalSources );
            }
            else
            {
                bfio::rfio::WriteVtkXmlPImageData
                ( comm, N, targetBox, *u, "genRadon3d" );
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

