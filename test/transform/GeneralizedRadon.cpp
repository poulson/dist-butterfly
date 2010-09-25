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
using namespace std;
using namespace bfio;

void 
Usage()
{
    cout << "GeneralizedRadon <N> <M> <testAccuracy?> <visualize?>" << endl;
    cout << "  N: power of 2, the frequency spread in each dimension" << endl;
    cout << "  M: number of random sources to instantiate" << endl;
    cout << "  testAccuracy?: tests accuracy iff 1" << endl;
    cout << "  visualize?: creates data files iff 1" << endl;
    cout << endl;
}

// Define the dimension of the problem and the order of interpolation
static const size_t d = 2;
static const size_t q = 8;

// If we test the accuracy, define the number of tests to perform per box
static const size_t numAccuracyTestsPerBox = 1;

// If we visualize the results, define the number of samples per box per dim.
static const size_t numVizSamplesPerBoxDim = 3;
static const size_t numVizSamplesPerBox = Pow<numVizSamplesPerBoxDim,d>::val;

template<typename R>
class Unity : public AmplitudeFunctor<R,d>
{
public:
    complex<R>
    operator() 
    ( const tr1::array<R,d>& x, const tr1::array<R,d>& p ) const
    { return complex<R>(1); }
};

template<typename R>
class GenRadon : public PhaseFunctor<R,d>
{
    R c1( const tr1::array<R,d>& x ) const
    { return (2+std::sin(TwoPi*x[0])*std::sin(TwoPi*x[1]))/3.; }

    R c2( const tr1::array<R,d>& x ) const
    { return (2+std::cos(TwoPi*x[0])*std::cos(TwoPi*x[1]))/3.; }
public:
    R
    operator() ( const tr1::array<R,d>& x, const tr1::array<R,d>& p ) const
    {
        R a = c1(x)*p[0];
        R b = c2(x)*p[1];
        return x[0]*p[0]+x[1]*p[1] + sqrt(a*a+b*b);
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

    if( !IsPowerOfTwo(numProcesses) )
    {
        if( rank == 0 )
            cout << "Must run with a power of two number of cores." << endl;
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
    const size_t N = atoi(argv[1]);
    const size_t M = atoi(argv[2]);
    const bool testAccuracy = atoi(argv[3]);
    const bool visualize = atoi(argv[4]);

    const size_t log2N = Log2( N );
    const size_t log2NumProcesses = Log2( numProcesses );
    if( log2NumProcesses > d*log2N )
    {
        if( rank == 0 )
            cout << "Cannot run with more than N^d processes." << endl;
        MPI_Finalize();
        return 0;
    }

    // Set our spatial and frequency boxes
    Box<double,d> freqBox, spatialBox;
    for( size_t j=0; j<d; ++j )
    {
        freqBox.offsets[j] = -0.5*N;
        freqBox.widths[j] = N;
        spatialBox.offsets[j] = 0;
        spatialBox.widths[j] = 1;
    }

    if( rank == 0 )
    {
        ostringstream msg;
        msg << "Will distribute " << M << " random sources over the frequency " 
            << "domain, which will be split into " << N 
            << " boxes in each of the " << d << " dimensions and distributed "
            << "amongst " << numProcesses << " processes." << endl << endl;
        cout << msg.str();
    }

    try 
    {
        // Consistently randomly seed all of the processes' PRNG.
        long seed;
        if( rank == 0 )
            seed = time(0);
        MPI_Bcast( &seed, 1, MPI_LONG, 0, comm );
        srand( seed );

        // Compute the box that our process owns within the frequency box
        Box<double,d> myFreqBox;
        LocalFreqPartitionData( freqBox, myFreqBox, comm );

        // Now generate random sources across the domain and store them in 
        // our local list when appropriate
        double L1Sources = 0;
        vector< Source<double,d> > mySources;
        vector< Source<double,d> > globalSources;
        if( testAccuracy || visualize )
        {
            globalSources.resize( M );
            for( size_t i=0; i<M; ++i )
            {
                for( size_t j=0; j<d; ++j )
                {
                    globalSources[i].p[j] = freqBox.offsets[j] + 
                        freqBox.widths[j]*Uniform<double>(); 
                }
                globalSources[i].magnitude = 1.*(2*Uniform<double>()-1); 
                L1Sources += abs(globalSources[i].magnitude);

                // Check if we should push this source onto our local list
                bool isMine = true;
                for( size_t j=0; j<d; ++j )
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
            size_t numLocalSources = 
                ( rank<(int)(M%numProcesses) 
                  ? M/numProcesses+1 : M/numProcesses );
            mySources.resize( numLocalSources );
            for( size_t i=0; i<numLocalSources; ++i )
            {
                for( size_t j=0; j<d; ++j )
                {
                    mySources[i].p[j] = myFreqBox.offsets[j] + 
                                        Uniform<double>()*myFreqBox.widths[j];
                }
                mySources[i].magnitude = 1.*(2*Uniform<double>()-1);
                L1Sources += abs(mySources[i].magnitude);
            }
        }

        // Set up our amplitude and phase functors
        Unity<double> unity;
        GenRadon<double> genRadon;

        // Create a context, which includes all of the precomputation
        if( rank == 0 )
        {
            cout << "Creating context...";
            cout.flush();
        }
        Context<double,d,q> context;
        if( rank == 0 )
            cout << "done." << endl;

        // Run the algorithm to generate the potential field
        auto_ptr< const PotentialField<double,d,q> > u;
        if( rank == 0 )
            cout << "Launching transform..." << endl;
        MPI_Barrier( comm );
        double startTime = MPI_Wtime();
        u = FreqToSpatial
        ( N, freqBox, spatialBox, unity, genRadon, context, mySources, comm );
        MPI_Barrier( comm );
        double stopTime = MPI_Wtime();
        if( rank == 0 )
        {
            cout << "Runtime: " << stopTime-startTime << " seconds." << endl;
            cout << endl;
        }

        if( testAccuracy )
        {
            const Box<double,d>& myBox = u->GetBox();
            const size_t numSubboxes = u->GetNumSubboxes();
            const size_t numTests = numSubboxes*numAccuracyTestsPerBox;

            // Compute error estimates using a constant number of samples within
            // each box in the resulting approximation of the transform.
            if( rank == 0 )
                cout << "Testing accuracy with O(N^d) samples..." << endl;
            double myL2ErrorSquared = 0;
            double myL2TruthSquared = 0;
            double myLinfError = 0;
            for( size_t k=0; k<numTests; ++k )
            {
                // Compute a random point in our process's spatial box
                tr1::array<double,d> x;
                for( size_t j=0; j<d; ++j )
                    x[j] = myBox.offsets[j] + Uniform<double>()*myBox.widths[j];

                // Evaluate our potential field at x and compare against truth
                complex<double> approx = u->Evaluate( x );
                complex<double> truth(0.,0.);
                for( size_t m=0; m<globalSources.size(); ++m )
                {
                    complex<double> beta = 
                        ImagExp<double>( TwoPi*genRadon(x,globalSources[m].p) );
                    truth += beta * globalSources[m].magnitude;
                }
                double absError = abs(approx-truth);
                double absTruth = abs(truth);
                myL2ErrorSquared += absError*absError;
                myL2TruthSquared += absTruth*absTruth;
                myLinfError = max( myLinfError, absError );
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
                cout << "---------------------------------------------" << endl;
                cout << "Estimate of relative ||e||_2:    "
                     << sqrt(L2ErrorSquared/L2TruthSquared) << endl;
                cout << "Estimate of ||e||_inf:           "                     
                     << LinfError << endl;
                cout << "||f||_1:                         "
                     << L1Sources << endl;
                cout << "Estimate of ||e||_inf / ||f||_1: "
                     << LinfError/L1Sources << endl;
                cout << endl;
            }
        }

        if( visualize )
        {
            ostringstream basenameStream;
            basenameStream << "genRadon-N=" << N << "-q=" << q 
                           << "-rank=" << rank;
            string basename = basenameStream.str();

            // Columns 0-(d-1) contain the coordinates of the sources, 
            // and columns d and d+1 contain the real and complex components of
            // the magnitudes of the sources.
            if( rank == 0 )
                cout << "Creating sources file..." << endl;
            ofstream file;
            file.open( (basename+"-sources.dat").c_str() );
            for( size_t i=0; i<globalSources.size(); ++i )
            {
                for( size_t j=0; j<d; ++j )
                    file << globalSources[i].p[j] << " ";
                file << real(globalSources[i].magnitude) << " "
                     << imag(globalSources[i].magnitude) << endl;
            }
            file.close();

            // Columns 0-(d-1) contain the coordinates of the samples, 
            // columns d and d+1 contain the real and complex components of 
            // the true solution, d+2 and d+3 contain the real and complex 
            // components of the approximate solution, and columns d+4 and d+5
            // contain the real and complex parts of the error, truth-approx.
            if( rank == 0 )
                cout << "Creating results file..." << endl;
            file.open( (basename+"-results.dat").c_str() );
            const Box<double,d>& myBox = u->GetBox();
            const tr1::array<double,d>& wA = u->GetSubboxWidths();
            const tr1::array<size_t,d>& log2SubboxesPerDim = 
                u->GetLog2SubboxesPerDim();
            const size_t numSubboxes = u->GetNumSubboxes();
            const size_t numVizSamples = numVizSamplesPerBox*numSubboxes;

            tr1::array<size_t,d> numSamplesUpToDim;
            for( size_t j=0; j<d; ++j )
            {
                numSamplesUpToDim[j] = 1;
                for( size_t i=0; i<j; ++i )
                {
                    numSamplesUpToDim[j] *= 
                        numVizSamplesPerBoxDim << log2SubboxesPerDim[i];
                }
            }

            for( size_t k=0; k<numVizSamples; ++k )
            {
                // Extract our indices in each dimension
                tr1::array<size_t,d> coords;
                for( size_t j=0; j<d; ++j )
                    coords[j] = (k/numSamplesUpToDim[j]) % 
                                (numVizSamplesPerBoxDim<<log2SubboxesPerDim[j]);

                // Compute the location of our sample
                tr1::array<double,d> x;
                for( size_t j=0; j<d; ++j )
                {
                    x[j] = myBox.offsets[j] + 
                           coords[j]*wA[j]/numVizSamplesPerBoxDim;
                }

                complex<double> truth(0,0);
                for( size_t m=0; m<globalSources.size(); ++m )
                {
                    complex<double> beta = 
                        ImagExp<double>(TwoPi*genRadon(x,globalSources[m].p));
                    truth += beta * globalSources[m].magnitude;
                }
                complex<double> approx = u->Evaluate( x );
                complex<double> error = truth - approx;

                // Write out this sample
                for( size_t j=0; j<d; ++j )
                    file << x[j] << " ";
                file << real(truth)  << " " << imag(truth)  << " "
                     << real(approx) << " " << imag(approx) << " "
                     << real(error)  << " " << imag(error)  << endl;
            }
            file.close();
        }
    }
    catch( const exception& e )
    {
        ostringstream msg;
        msg << "Caught exception on process " << rank << ":" << endl;
        msg << "   " << e.what() << endl;
        cout << msg.str();
    }

    MPI_Finalize();
    return 0;
}

