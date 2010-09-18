/*
   ButterflyFIO: a distributed-memory fast algorithm for applying FIOs.
   Copyright (C) 2010 Jack Poulson <jack.poulson@gmail.com>
 
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
#include <algorithm>
#include <ctime>
#include <fstream>
#include <memory>
#include "bfio.hpp"

namespace {
void 
Usage()
{
    std::cout << "UpWave <N> <M> <testAccuracy?>" 
              << std::endl 
              << "  N: power of 2, the frequency spread in each dimension" 
              << std::endl
              << "  M: number of random sources to instantiate" 
              << std::endl
              << "  testAccuracy?: test accuracy iff 1" 
              << std::endl
              << "  visualize?: create data files iff 1" 
              << std::endl << std::endl;
}
} // anonymous namespace

// Define the dimension of the problem and the order of interpolation
static const std::size_t d = 3;
static const std::size_t q = 8;

// Define the number of samples to take from each box if testing accuracy
static const std::size_t numAccuracyTestsPerBox = 1;

// If we visualize the results, define the number of samples per box per dim.
static const std::size_t numVizSamplesPerBoxDim = 3;
static const std::size_t numVizSamplesPerBox = 
    bfio::Pow<numVizSamplesPerBoxDim,d>::val;

template<typename R>
class Unity : public bfio::AmplitudeFunctor<R,d>
{
public:
    std::complex<R>
    operator() 
    ( const std::tr1::array<R,d>& x, const std::tr1::array<R,d>& p ) const
    { return std::complex<R>(1); }
};

template<typename R>
class UpWave : public bfio::PhaseFunctor<R,d>
{
public:
    R
    operator() 
    ( const std::tr1::array<R,d>& x, const std::tr1::array<R,d>& p ) const
    {
        return x[0]*p[0]+x[1]*p[1]+x[2]*p[2] + 
               0.5*sqrt(p[0]*p[0]+p[1]*p[1]+p[2]*p[2]); 
    }
};

int
main
( int argc, char* argv[] )
{
    MPI_Init( &argc, &argv );

    int rank, numProcesses;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_rank( comm, &rank );
    MPI_Comm_size( comm, &numProcesses );

    if( !bfio::IsPowerOfTwo(numProcesses) )
    {
        if( rank == 0 )
        {
            std::cout << "Must run with a power of two number of cores." 
                      << std::endl;
        }
        MPI_Finalize();
        return 0;
    }

    if( argc != 5 )
    {
        if( rank == 0 )
            Usage();
        MPI_Finalize();
        return 0;
    }
    const std::size_t N = atoi(argv[1]);
    const std::size_t M = atoi(argv[2]);
    const bool testAccuracy = atoi(argv[3]);
    const bool visualize = atoi(argv[4]);

    // Set the frequency and spatial boxes
    bfio::Box<double,d> freqBox, spatialBox;
    for( std::size_t j=0; j<d; ++j )
    {
        freqBox.offsets[j] = -0.5*N;
        freqBox.widths[j] = N;
        spatialBox.offsets[j] = 0;
        spatialBox.widths[j] = 1;
    }

    if( rank == 0 )
    {
        std::ostringstream msg;
        msg << "Will distribute " << M << " random sources over the frequency "
            << "domain, which will be split into " << N
            << " boxes in each of the " << d << " dimensions and distributed "
            << "amongst " << numProcesses << " processes." 
            << std::endl << std::endl;
        std::cout << msg.str();
    }

    try 
    {
        // Consistently randomly seed all of the processes' PRNG.
        long seed;
        if( rank == 0 )
            seed = time(0);
        MPI_Bcast( &seed, 1, MPI_LONG, 0, comm );
        srand( seed );

        // Compute the box that our process owns
        bfio::Box<double,d> myFreqBox;
        bfio::LocalFreqPartitionData( freqBox, myFreqBox, comm );

        // Now generate random sources across the domain and store them in 
        // our local list when appropriate
        double L1Sources = 0;
        std::vector< bfio::Source<double,d> > mySources;
        std::vector< bfio::Source<double,d> > globalSources;
        if( testAccuracy || visualize )
        {
            globalSources.resize( M );
            for( std::size_t i=0; i<M; ++i )
            {
                for( std::size_t j=0; j<d; ++j )
                {
                    globalSources[i].p[j] = freqBox.offsets[j] + 
                        freqBox.widths[j]*bfio::Uniform<double>(); 
                }
                globalSources[i].magnitude = 10*(2*bfio::Uniform<double>()-1); 
                L1Sources += std::abs(globalSources[i].magnitude);

                // Check if we should push this source onto our local list
                bool isMine = true;
                for( std::size_t j=0; j<d; ++j )
                {
                    double u = globalSources[i].p[j];
                    double start = myFreqBox.offsets[j];
                    double stop = myFreqBox.offsets[j] + myFreqBox.widths[j];
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
                        myFreqBox.offsets[j]+
                        bfio::Uniform<double>()*myFreqBox.widths[j];
                }
                mySources[i].magnitude = 10*(2*bfio::Uniform<double>()-1);
                L1Sources += std::abs(mySources[i].magnitude);
            }
        }

        // Set up our amplitude and phase functors
        Unity<double> unity;
        UpWave<double> upWave;

        // Create a context, which includes all of the precomputation
        if( rank == 0 )
            std::cout << "Creating context..." << std::endl;
        bfio::Context<double,d,q> context;

        // Run the algorithm
        std::auto_ptr< const bfio::PotentialField<double,d,q> > u;
        if( rank == 0 )
            std::cout << "Starting transform..." << std::endl;
        MPI_Barrier( comm );
        double startTime = MPI_Wtime();
        u = bfio::FreqToSpatial
        ( N, freqBox, spatialBox, unity, upWave, context, mySources, comm );
        MPI_Barrier( comm );
        double stopTime = MPI_Wtime();
        if( rank == 0 )
        {
            std::cout << "Runtime: " << stopTime-startTime << " seconds." 
                      << std::endl << std::endl;
        }

        if( testAccuracy )
        {
            const bfio::Box<double,d>& myBox = u->GetBox();
            const std::size_t numSubboxes = u->GetNumSubboxes();
            const std::size_t numTests = numSubboxes*numAccuracyTestsPerBox;

            // Compute error estimates using a constant number of samples within
            // each box in the resulting approximation of the transform.
            if( rank == 0 )
            {
                std::cout << "Test accuracy with O(N^d) samples..." 
                          << std::endl;
            }
            double myL2ErrorSquared = 0.;
            double myL2TruthSquared = 0.;
            double myLinfError = 0.;
            for( std::size_t k=0; k<numTests; ++k )
            {
                // Compute a random point in our process's spatial box
                std::tr1::array<double,d> x;
                for( std::size_t j=0; j<d; ++j )
                    x[j] = myBox.offsets[j] + 
                           bfio::Uniform<double>()*myBox.widths[j];

                // Evaluate our potential field at x and compare against truth
                std::complex<double> approx = u->Evaluate( x );
                std::complex<double> truth(0.,0.);
                for( std::size_t m=0; m<globalSources.size(); ++m )
                {
                    std::complex<double> beta = 
                        bfio::ImagExp
                        ( bfio::TwoPi*upWave(x,globalSources[m].p) );
                    truth += beta * globalSources[m].magnitude;
                }
                double absError = std::abs(approx-truth);
                double absTruth = std::abs(truth);
                myL2ErrorSquared += absError*absError;
                myL2TruthSquared += absTruth*absTruth;
                myLinfError = std::max( myLinfError, absError );
            }
            double L2ErrorSquared;
            double L2TruthSquared;
            double LinfError;
            MPI_Reduce
            ( &myL2ErrorSquared, &L2ErrorSquared, 1, MPI_DOUBLE, MPI_SUM, 0, 
              comm );
            MPI_Reduce
            ( &myL2TruthSquared, &L2TruthSquared, 1, MPI_DOUBLE, MPI_SUM, 0,
              comm );
            MPI_Reduce
            ( &myLinfError, &LinfError, 1, MPI_DOUBLE, MPI_MAX, 0, comm );
            if( rank == 0 )
            {   
                std::cout << "---------------------------------------------" 
                          << std::endl
                          << "Estimate of relative ||e||_2:    " 
                          << sqrt(L2ErrorSquared/L2TruthSquared) << std::endl
                          << "Estimate of ||e||_inf:           " 
                          << LinfError << std::endl
                          << "||f||_1:                         " 
                          << L1Sources << std::endl
                          << "Estimate of ||e||_inf / ||f||_1: "  
                          << LinfError/L1Sources << std::endl << std::endl;
            }
        }

        if( visualize )
        {
            std::ostringstream basenameStream;
            basenameStream << "upWave-N=" << N << "-q=" << q 
                           << "-rank=" << rank;
            std::string basename = basenameStream.str();

            // Columns 0-(d-1) contain the coordinates of the sources, 
            // and columns d and d+1 contain the real and complex components of
            // the magnitudes of the sources.
            if( rank == 0 )
                std::cout << "Creating sources file..." << std::endl;
            std::ofstream file;
            file.open( (basename+"-sources.dat").c_str() );
            for( std::size_t i=0; i<globalSources.size(); ++i )
            {
                for( std::size_t j=0; j<d; ++j )
                    file << globalSources[i].p[j] << " ";
                file << std::real(globalSources[i].magnitude) << " "
                     << std::imag(globalSources[i].magnitude) << std::endl;
            }
            file.close();

            // Columns 0-(d-1) contain the coordinates of the samples, 
            // columns d and d+1 contain the real and complex components of 
            // the true solution, d+2 and d+3 contain the real and complex 
            // components of the approximate solution, and columns d+4 and d+5
            // contain the real and complex parts of the error, truth-approx.
            if( rank == 0 )
                std::cout << "Creating results file..." << std::endl;
            file.open( (basename+"-results.dat").c_str() );
            const bfio::Box<double,d>& myBox = u->GetBox();
            const std::tr1::array<double,d>& wA = u->GetSubboxWidths();
            const std::tr1::array<std::size_t,d>& log2SubboxesPerDim =
                u->GetLog2SubboxesPerDim();
            const std::size_t numSubboxes = u->GetNumSubboxes();
            const std::size_t numVizSamples = numVizSamplesPerBox*numSubboxes;

            std::tr1::array<std::size_t,d> numSamplesUpToDim;
            for( std::size_t j=0; j<d; ++j )
            {
                numSamplesUpToDim[j] = 1;
                for( std::size_t i=0; i<j; ++i )
                {
                    numSamplesUpToDim[j] *=
                        numVizSamplesPerBoxDim << log2SubboxesPerDim[i];
                }
            }

            for( std::size_t k=0; k<numVizSamples; ++k )
            {
                // Extract our indices in each dimension
                std::tr1::array<std::size_t,d> coords;
                for( std::size_t j=0; j<d; ++j )
                    coords[j] = (k/numSamplesUpToDim[j]) %
                                (numVizSamplesPerBoxDim<<log2SubboxesPerDim[j]);

                // Compute the location of our sample
                std::tr1::array<double,d> x;
                for( std::size_t j=0; j<d; ++j )
                {
                    x[j] = myBox.offsets[j] +
                           coords[j]*wA[j]/numVizSamplesPerBoxDim;
                }

                std::complex<double> truth(0,0);
                for( std::size_t m=0; m<globalSources.size(); ++m )
                {
                    std::complex<double> beta =
                        bfio::ImagExp
                        ( bfio::TwoPi*upWave(x,globalSources[m].p) );
                    truth += beta * globalSources[m].magnitude;
                }
                std::complex<double> approx = u->Evaluate( x );
                std::complex<double> error = truth - approx;

                // Write out this sample
                for( std::size_t j=0; j<d; ++j )
                    file << x[j] << " ";
                file << std::real(truth)  << " " << std::imag(truth)  << " "
                     << std::real(approx) << " " << std::imag(approx) << " "
                     << std::real(error)  << " " << std::imag(error)  
                     << std::endl;
            }
            file.close();
        }
    }
    catch( const std::exception& e )
    {
        std::ostringstream msg;
        msg << "Caught exception on process " << rank << ":" << std::endl
            << "   " << e.what() << std::endl;
        std::cout << msg.str();
    }

    MPI_Finalize();
    return 0;
}

