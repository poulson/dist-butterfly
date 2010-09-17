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
    cout << "NonUniformFT <N> <M> <testAccuracy?>" << endl;
    cout << "  N: power of 2, the frequency spread in each dimension" << endl;
    cout << "  M: number of random sources to instantiate" << endl;
    cout << "  testAccuracy?: tests accuracy iff 1" << endl;
    cout << "  visualize?: creates data files iff 1" << endl;
    cout << endl;
}

// Define the dimension of the problem and the order of interpolation
static const unsigned d = 2;
static const unsigned q = 8;

// If we test the accuracy, define the number of tests to perform per box
static const unsigned numAccuracyTestsPerBox = 10;

// If we visualize the results, define the number of samples per box per dim.
static const unsigned numVizSamplesPerBoxDim = 3;

struct VizSample {
    Array<double,d> point;
    complex<double> truth;
    complex<double> approx;
    complex<double> error;
};

// This must be modified if d != 2
bool
VizSampleSort( const VizSample& a, const VizSample& b )
{ return a.point[1]<b.point[1] ||
         (a.point[1]==b.point[1] && a.point[0]<b.point[0]); }

template<typename R>
class Unity : public AmplitudeFunctor<R,d>
{
public:
    complex<R>
    operator() ( const Array<R,d>& x, const Array<R,d>& p ) const
    { return complex<R>(1); }
};

template<typename R>
class Fourier : public PhaseFunctor<R,d>
{
public:
    R
    operator() ( const Array<R,d>& x, const Array<R,d>& p ) const
    {
        return x[0]*p[0]+x[1]*p[1];
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
    const unsigned N = atoi(argv[1]);
    const unsigned M = atoi(argv[2]);
    const bool testAccuracy = atoi(argv[3]);
    const bool visualize = atoi(argv[4]);

    const unsigned log2N = Log2( N );
    const unsigned log2NumProcesses = Log2( numProcesses );
    if( log2NumProcesses > d*log2N )
    {
        if( rank == 0 )
            cout << "Cannot run with more than N^d processes." << endl;
        MPI_Finalize();
        return 0;
    }

    // Set our spatial and frequency boxes
    Box<double,d> freqBox, spatialBox;
    for( unsigned j=0; j<d; ++j )
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
            for( unsigned i=0; i<M; ++i )
            {
                for( unsigned j=0; j<d; ++j )
                {
                    globalSources[i].p[j] = freqBox.offsets[j] + 
                        freqBox.widths[j]*Uniform<double>(); 
                }
                globalSources[i].magnitude = 1.*(2*Uniform<double>()-1); 
                L1Sources += abs(globalSources[i].magnitude);

                // Check if we should push this source onto our local list
                bool isMine = true;
                for( unsigned j=0; j<d; ++j )
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
            unsigned numLocalSources = 
                ( rank<(int)(M%numProcesses) 
                  ? M/numProcesses+1 : M/numProcesses );
            mySources.resize( numLocalSources );
            for( unsigned i=0; i<numLocalSources; ++i )
            {
                for( unsigned j=0; j<d; ++j )
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
        Fourier<double> fourier;

        // Create a context, which includes all of the precomputation
        if( rank == 0 )
            cout << "Creating context..." << endl;
        Context<double,d,q> context;

        // Create vectors for storing the results
        unsigned numLocalLRPs = NumLocalBoxes<d>( N, comm );
        vector< LowRankPotential<double,d,q> > myLRPs
        ( numLocalLRPs, LowRankPotential<double,d,q>(unity,fourier,context) );

        // Run the algorithm
        if( rank == 0 )
            cout << "Starting transform..." << endl;
        MPI_Barrier( comm );
        double startTime = MPI_Wtime();
        FreqToSpatial( N, freqBox, spatialBox, mySources, myLRPs, comm );
        MPI_Barrier( comm );
        double stopTime = MPI_Wtime();
        if( rank == 0 )
        {
            cout << "Runtime: " << stopTime-startTime << " seconds." << endl;
            cout << endl;
        }

        if( testAccuracy )
        {
            // Compute the relative error using 256 random samples on the grid 
            // as in Candes et al.'s ButterflyFIO paper.
            if( rank == 0 )
                cout << "Testing accuracy with 256 samples..." << endl;
            double myL2ErrorSquared256 = 0;
            double myL2TruthSquared256 = 0;
            double myLinfError256 = 0;
            for( unsigned s=0; s<256; ++s )
            {
                unsigned k = rand() % myLRPs.size();

                // Retrieve the spatial center of LRP k
                Array<double,d> x0 = myLRPs[k].GetSpatialCenter();
                
                // Evaluate our LRP at x0 and compare against truth
                complex<double> u = myLRPs[k]( x0 );
                complex<double> uTruth(0,0);
                for( unsigned m=0; m<globalSources.size(); ++m )
                {
                    complex<double> beta = 
                        ImagExp( TwoPi*fourier(x0,globalSources[m].p) );
                    uTruth += beta * globalSources[m].magnitude;
                }
                double absError = abs(u-uTruth);
                double absTruth = abs(uTruth);
                myL2ErrorSquared256 += absError*absError;
                myL2TruthSquared256 += absTruth*absTruth;
                myLinfError256 = max( myLinfError256, absError );
            }

            double L2ErrorSquared256;
            double L2TruthSquared256;
            double LinfError256;
            MPI_Reduce
            ( &myL2ErrorSquared256, &L2ErrorSquared256, 1, MPI_DOUBLE, 
              MPI_SUM, 0, comm );
            MPI_Reduce
            ( &myL2TruthSquared256, &L2TruthSquared256, 1, MPI_DOUBLE, 
              MPI_SUM, 0, comm );
            MPI_Reduce
            ( &myLinfError256, &LinfError256, 1, MPI_DOUBLE, MPI_MAX, 0, comm );
            if( rank == 0 )
            {   
                cout << "---------------------------------------------" << endl;
                cout << "Estimate of relative ||e||_2:    "
                     << sqrt(L2ErrorSquared256/L2TruthSquared256) << endl;
                cout << "Estimate of ||e||_inf:           "                     
                     << LinfError256 << endl;
                cout << "||f||_1:                         "
                     << L1Sources << endl;
                cout << "Estimate of ||e||_inf / ||f||_1: "
                     << LinfError256/L1Sources << endl;
                cout << endl;
            }

            // Compute error estimates using a constant number of samples within
            // each box in the resulting approximation of the transform.
            if( rank == 0 )
                cout << "Testing accuracy with O(N^d) samples..." << endl;
            double myL2ErrorSquared = 0;
            double myL2TruthSquared = 0;
            double myLinfError = 0;
            for( unsigned k=0; k<myLRPs.size(); ++k )
            {
                // Retrieve the spatial center of LRP k
                Array<double,d> x0 = 
                    myLRPs[k].GetSpatialCenter();

                for( unsigned s=0; s<numAccuracyTestsPerBox; ++s )
                {
                    // Find a random point in that box
                    Array<double,d> x;
                    for( unsigned j=0; j<d; ++j )
                    {
                        x[j] = x0[j] + spatialBox.widths[j] /
                                       (2*N)*(2*Uniform<double>()-1.);
                    }

                    // Evaluate our LRP at x  and compare against truth
                    complex<double> u = myLRPs[k]( x );
                    complex<double> uTruth(0.,0.);
                    for( unsigned m=0; m<globalSources.size(); ++m )
                    {
                        complex<double> beta = 
                            ImagExp( TwoPi*fourier(x,globalSources[m].p) );
                        uTruth += beta * globalSources[m].magnitude;
                    }
                    double absError = abs(u-uTruth);
                    double absTruth = abs(uTruth);
                    myL2ErrorSquared += absError*absError;
                    myL2TruthSquared += absTruth*absTruth;
                    myLinfError = max( myLinfError, absError );
                }
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
            // Each process creates a sorted Cartesian grid of the true 
            // solution, the approximate solution, and the error.
            if( rank == 0 )
                cout << "Sampling to create files for visualization..." << endl;
            const unsigned numVizSamplesPerBox =
                Pow<numVizSamplesPerBoxDim,d>::val;
            const unsigned numVizSamples = numVizSamplesPerBox*myLRPs.size();
            vector<VizSample> vizSamples( numVizSamples );

            const Array<double,d> wA = myLRPs[0].GetSpatialWidths();

            // Fill the unsorted vector
            for( unsigned k=0; k<myLRPs.size(); ++k )
            {
                // Retrieve the bottom-left corner of LRP k
                const Array<double,d> x0 = myLRPs[k].GetSpatialCenter();
                Array<double,d> xBL;
                for( unsigned j=0; j<d; ++j )
                    xBL[j] = x0[j] - wA[j]/2.;

                for( unsigned s=0; s<numVizSamplesPerBox; ++s )
                {
                    unsigned pow = 1;
                    VizSample& v =
                        vizSamples[numVizSamplesPerBox*k+s];
                    for( unsigned j=0; j<d; ++j )
                    {
                        unsigned i = (s/pow) % numVizSamplesPerBoxDim;
                        v.point[j] = xBL[j] + i*wA[j]/numVizSamplesPerBoxDim;
                        pow *= numVizSamplesPerBoxDim;
                    }

                    v.truth = complex<double>(0,0);
                    for( unsigned m=0; m<globalSources.size(); ++m )
                    {
                        complex<double> beta =
                            ImagExp(TwoPi*fourier(v.point,globalSources[m].p));
                        v.truth += beta * globalSources[m].magnitude;
                    }
                    v.approx = myLRPs[k]( v.point );
                    v.error = v.truth-v.approx;
                }
            }

            // Sort the vectors from the highest to lowest dimensions
            if( rank == 0 )
                cout << "Sorting samples..." << endl;
            sort( vizSamples.begin(), vizSamples.end(), VizSampleSort );

            if( rank == 0 )
                cout << "Writing out sorted data..." << endl;
            ostringstream basenameStream;
            basenameStream << "nuft-N=" << N << "-" << "q=" << q
                << "-rank=" << rank;
            string basename = basenameStream.str();

            // Columns 0-(d-1) contain the coordinates of the sources, 
            // and columns d and d+1 contain the real and complex components of
            // the magnitudes of the sources.
            ofstream file;
            file.open( (basename+"-sources.dat").c_str() );
            for( unsigned i=0; i<globalSources.size(); ++i )
            {
                for( unsigned j=0; j<d; ++j )
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
            file.open( (basename+"-results.dat").c_str() );
            for( unsigned i=0; i<vizSamples.size(); ++i )
            {
                for( unsigned j=0; j<d; ++j )
                    file << vizSamples[i].point[j] << " ";
                file << real(vizSamples[i].truth) << " "
                     << imag(vizSamples[i].truth) << " "
                     << real(vizSamples[i].approx) << " "
                     << imag(vizSamples[i].approx) << " "
                     << real(vizSamples[i].error) << " "
                     << imag(vizSamples[i].error) << endl;
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

