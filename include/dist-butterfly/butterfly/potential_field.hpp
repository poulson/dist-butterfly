/*
   Copyright (C) 2010-2013 Jack Poulson and Lexing Ying
 
   This file is part of DistButterfly and is under the GNU General Public 
   License, which can be found in the LICENSE file in the root directory, or at
   <http://www.gnu.org/licenses/>.
*/
#pragma once
#ifndef DBF_BFLY_POTENTIAL_FIELD_HPP
#define DBF_BFLY_POTENTIAL_FIELD_HPP

#include <fstream>
#include <iostream>
#include <random>

#include "dist-butterfly/structures/box.hpp"
#include "dist-butterfly/structures/constrained_htree_walker.hpp"
#include "dist-butterfly/structures/low_rank_potential.hpp"
#include "dist-butterfly/structures/weight_grid.hpp"
#include "dist-butterfly/structures/weight_grid_list.hpp"

#include "dist-butterfly/butterfly/context.hpp"

#include "dist-butterfly/functors/amplitude.hpp"
#include "dist-butterfly/functors/phase.hpp"
#include "dist-butterfly/tools/special_functions.hpp"

namespace dbf {

using std::array;
using std::complex;
using std::size_t;
using std::string;
using std::vector;

namespace bfly {

template<typename R,size_t d,size_t q>
class PotentialField
{
    const Context<R,d,q>& context_;
    const Amplitude<R,d>* amplitude_;
    const Phase<R,d>* phase_;
    const Box<R,d> sBox_;
    const Box<R,d> myTBox_;
    const array<size_t,d> myTBoxCoords_;
    const array<size_t,d> log2TSubboxesPerDim_;

    array<R,d> wA_;
    array<R,d> p0_;
    array<size_t,d> log2TSubboxesUpToDim_;
    vector<LRP<R,d,q>> LRPs_;

public:
    PotentialField
    ( const Context<R,d,q>& context,
      const Amplitude<R,d>& amplitude,
      const Phase<R,d>& phase,
      const Box<R,d>& sBox,
      const Box<R,d>& myTBox,
      const array<size_t,d>& myTBoxCoords,
      const array<size_t,d>& log2TSubboxesPerDim,
      const WeightGridList<R,d,q>& weightGridList );

    ~PotentialField();

    complex<R> Evaluate( const array<R,d>& x ) const;
    // TODO: BatchEvaluate? SafeEvaluate?

    const Amplitude<R,d>& GetAmplitude() const;
    const Phase<R,d>& GetPhase() const;
    const Box<R,d>& GetMyTargetBox() const;
    size_t GetNumSubboxes() const;
    const array<R,d>& GetSubboxWidths() const;
    const array<size_t,d>& GetMyTargetBoxCoords() const;
    const array<size_t,d>& GetLog2SubboxesPerDim() const;
    const array<size_t,d>& GetLog2SubboxesUpToDim() const;
};

template<typename R,size_t d,size_t q>
void PrintErrorEstimates
( MPI_Comm comm,
  const PotentialField<R,d,q>& u,
  const vector<Source<R,d>>& sources );

template<typename R,size_t d,size_t q>
void WriteImage
( MPI_Comm comm, 
  const size_t N,
  const Box<R,d>& tBox,
  const PotentialField<R,d,q>& u,
  const string& basename );

template<typename R,size_t d,size_t q>
void WriteImage
( MPI_Comm comm, 
  const size_t N,
  const Box<R,d>& tBox,
  const PotentialField<R,d,q>& u,
  const string& basename,
  const vector<Source<R,d>>& sources );

// Implementations

template<typename R,size_t d,size_t q>
inline
PotentialField<R,d,q>::PotentialField
( const Context<R,d,q>& context,
  const Amplitude<R,d>& amplitude,
  const Phase<R,d>& phase,
  const Box<R,d>& sBox,
  const Box<R,d>& myTBox,
  const array<size_t,d>& myTBoxCoords,
  const array<size_t,d>& log2TSubboxesPerDim,
  const WeightGridList<R,d,q>& weightGridList )
: context_(context), amplitude_(amplitude.Clone()), phase_(phase.Clone()), 
  sBox_(sBox), myTBox_(myTBox), myTBoxCoords_(myTBoxCoords),
  log2TSubboxesPerDim_(log2TSubboxesPerDim)
{ 
    // Compute the widths of the target subboxes and the source center
    for( size_t j=0; j<d; ++j )
        wA_[j] = myTBox.widths[j] / (1<<log2TSubboxesPerDim[j]);
    for( size_t j=0; j<d; ++j )
        p0_[j] = sBox.offsets[j] + sBox.widths[j]/2;

    // Compute the array of the partial sums
    log2TSubboxesUpToDim_[0] = 0;
    for( size_t j=1; j<d; ++j )
    {
        log2TSubboxesUpToDim_[j] = 
            log2TSubboxesUpToDim_[j-1] + log2TSubboxesPerDim[j-1];
    }

    // Figure out the size of our LRP vector by summing log2TargetSubboxesPerDim
    size_t log2TSubboxes = 0;
    for( size_t j=0; j<d; ++j )
        log2TSubboxes += log2TSubboxesPerDim[j];
    LRPs_.resize( 1<<log2TSubboxes );

    // The weightGridList is assumed to be ordered by the constrained 
    // HTree described by log2TSubboxesPerDim. We will unroll it 
    // lexographically into the LRP vector.
    ConstrainedHTreeWalker<d> AWalker( log2TSubboxesPerDim );
    for( size_t tIndex=0; tIndex<LRPs_.size(); ++tIndex, AWalker.Walk() )
    {
        const array<size_t,d>& A = AWalker.State();

        // Unroll the indices of A into its lexographic position
        size_t k=0; 
        for( size_t j=0; j<d; ++j )
            k += A[j] << log2TSubboxesUpToDim_[j];

        // Now fill the k'th LRP index
        for( size_t j=0; j<d; ++j )
            LRPs_[k].x0[j] = myTBox.offsets[j] + (A[j]+R(1)/R(2))*wA_[j];
        LRPs_[k].weightGrid = weightGridList[tIndex];
    }
}

template<typename R,size_t d,size_t q>
inline
PotentialField<R,d,q>::~PotentialField()
{
    delete amplitude_;
    delete phase_;
}

template<typename R,size_t d,size_t q>
inline complex<R>
PotentialField<R,d,q>::Evaluate( const array<R,d>& x ) const
{
    typedef complex<R> C;

#ifndef RELEASE
    for( size_t j=0; j<d; ++j )
    {
        if( x[j] < myTBox_.offsets[j] || 
            x[j] > myTBox_.offsets[j]+myTBox_.widths[j] )
            throw std::runtime_error
            ("Tried to evaluate outside of potential range");
    }
#endif

    // Compute the lexographic position of the LRP to use for evaluation
    size_t k = 0;
    for( size_t j=0; j<d; ++j )
    {
        size_t owningIndex = size_t((x[j]-myTBox_.offsets[j])/wA_[j]);
        const size_t maxIndex = size_t((1u<<log2TSubboxesPerDim_[j])-1);
        owningIndex = std::min(owningIndex,maxIndex);
        k += owningIndex << log2TSubboxesUpToDim_[j];
    }

    // Convert x to the reference domain of [-1/2,+1/2]^d for box k
    const LRP<R,d,q>& lrp = LRPs_[k];
    array<R,d> xRef;
    for( size_t j=0; j<d; ++j )
        xRef[j] = (x[j]-lrp.x0[j])/wA_[j];

    const vector<array<R,d>>& chebyshevGrid = context_.GetChebyshevGrid();
    R realValue(0), imagValue(0);
    for( size_t t=0; t<Pow<q,d>::val; ++t )
    {
        // Construct the t'th translated Chebyshev gridpoint
        array<R,d> xt;
        for( size_t j=0; j<d; ++j )
            xt[j] = lrp.x0[j] + wA_[j]*chebyshevGrid[t][j];

        const C beta = ImagExp<R>( -phase_->operator()(xt,p0_) );
        const R lambda = context_.Lagrange(t,xRef);
        const R realWeight = lrp.weightGrid.RealWeight(t);
        const R imagWeight = lrp.weightGrid.ImagWeight(t);
        realValue += lambda*(realWeight*beta.real()-imagWeight*beta.imag());
        imagValue += lambda*(imagWeight*beta.real()+realWeight*beta.imag());
    }
    const C beta = ImagExp<R>( phase_->operator()(x,p0_) );
    const R realPotential = realValue*beta.real()-imagValue*beta.imag();
    const R imagPotential = imagValue*beta.real()+realValue*beta.imag();
    return C( realPotential, imagPotential );
}

template<typename R,size_t d,size_t q>
inline const Amplitude<R,d>&
PotentialField<R,d,q>::GetAmplitude() const
{ return *amplitude_; }

template<typename R,size_t d,size_t q>
inline const Phase<R,d>&
PotentialField<R,d,q>::GetPhase() const
{ return *phase_; }

template<typename R,size_t d,size_t q>
inline const Box<R,d>&
PotentialField<R,d,q>::GetMyTargetBox() const
{ return myTBox_; }

template<typename R,size_t d,size_t q>
inline size_t
PotentialField<R,d,q>::GetNumSubboxes() const
{ return LRPs_.size(); }

template<typename R,size_t d,size_t q>
inline const array<R,d>&
PotentialField<R,d,q>::GetSubboxWidths() const
{ return wA_; }

template<typename R,size_t d,size_t q>
inline const array<size_t,d>&
PotentialField<R,d,q>::GetMyTargetBoxCoords() const
{ return myTBoxCoords_; }

template<typename R,size_t d,size_t q>
inline const array<size_t,d>&
PotentialField<R,d,q>::GetLog2SubboxesPerDim() const
{ return log2TSubboxesPerDim_; }

template<typename R,size_t d,size_t q>
inline const array<size_t,d>&
PotentialField<R,d,q>::GetLog2SubboxesUpToDim() const
{ return log2TSubboxesUpToDim_; }

template<typename R,size_t d,size_t q>
void PrintErrorEstimates
( MPI_Comm comm,
  const PotentialField<R,d,q>& u,
  const vector<Source<R,d>>& sources )
{
#ifdef TIMING
    MPI_Barrier( comm );
    Timer accTimer;
    accTimer.Reset();
    accTimer.Start();
#endif // ifdef TIMING
    const size_t numAccuracyTestsPerBox = 10;

    int rank;
    MPI_Comm_rank( comm, &rank );

    const Amplitude<R,d>& amplitude = u.GetAmplitude();
    const Phase<R,d>& phase = u.GetPhase();
    const Box<R,d>& myTBox = u.GetMyTargetBox();
    const size_t numSubboxes = u.GetNumSubboxes();
    const size_t numTests = numSubboxes*numAccuracyTestsPerBox;

    // Compute error estimates using a constant number of samples within
    // each box in the resulting approximation of the transform.
    //
    // Double precision should be perfectly fine for our purposes.
    //
    if( rank == 0 )
    {
        std::cout << "Testing accuracy with " << numAccuracyTestsPerBox 
                  << " N^d = " << numTests << " samples..."
                  << std::endl;
    }
    // Build an RNG for uniformly sampling (0,1)
    std::random_device rd;
    std::default_random_engine engine( rd() );
    std::uniform_real_distribution<R> uniform_dist(R(0),R(1));
    auto uniform = std::bind( uniform_dist, std::ref(engine) );
    // Compute the L1 norm of the sources
    double L1Sources = 0.;
    const size_t numSources = sources.size();
    for( size_t m=0; m<numSources; ++m )
        L1Sources += abs(sources[m].magnitude);
    double myL2ErrorSquared = 0.;
    double myL2TruthSquared = 0.;
    double myLinfError = 0.;
    for( size_t k=0; k<numTests; ++k )
    {
        // Compute a random point in our process's target box
        array<R,d> x;
        for( size_t j=0; j<d; ++j )
            x[j] = myTBox.offsets[j] + myTBox.widths[j]*uniform();

        // Evaluate our potential field at x and compare against truth
        complex<R> approx = u.Evaluate( x );
        complex<R> truth(R(0),R(0));
        for( size_t m=0; m<numSources; ++m )
        {
            complex<R> beta =
                amplitude( x, sources[m].p ) * ImagExp( phase(x,sources[m].p) );
            truth += beta * sources[m].magnitude;
        }
        const double absError = std::abs(approx-truth);
        const double absTruth = std::abs(truth);
        myL2ErrorSquared += absError*absError;
        myL2TruthSquared += absTruth*absTruth;
        myLinfError = std::max( myLinfError, absError );
    }
    double L2ErrorSquared, L2TruthSquared, LinfError;
    MPI_Reduce
    ( &myL2ErrorSquared, &L2ErrorSquared, 1, MPI_DOUBLE, MPI_SUM, 0, comm );
    MPI_Reduce
    ( &myL2TruthSquared, &L2TruthSquared, 1, MPI_DOUBLE, MPI_SUM, 0, comm );
    MPI_Reduce( &myLinfError, &LinfError, 1, MPI_DOUBLE, MPI_MAX, 0, comm );
#ifdef TIMING
    accTimer.Stop();
#endif // ifdef TIMING
    if( rank == 0 )
    {
        std::cout << "---------------------------------------------\n"
                  << "Estimate of relative ||e||_2:    "
                  << sqrt(L2ErrorSquared/L2TruthSquared) << "\n"
                  << "Estimate of ||e||_inf:           "
                  << LinfError << "\n"
                  << "||f||_1:                         "
                  << L1Sources << "\n"
                  << "Estimate of ||e||_inf / ||f||_1: "
                  << LinfError/L1Sources << "\n" 
                  << "---------------------------------------------\n";
#ifdef TIMING
        std::cout << "Time for accuracy test: " << accTimer.Total() 
                  << " seconds\n";
#endif // ifdef TIMING
        std::cout << std::endl;
    }
}

// Just write out the real and imag components of the approximation
template<typename R,size_t d,size_t q>
inline void
WriteImage
( MPI_Comm comm,
  const size_t N,
  const Box<R,d>& tBox,
  const PotentialField<R,d,q>& u,
  const string& basename )
{
    using namespace std;

    const size_t numSamplesPerBoxDim = 4;
    const size_t numSamplesPerBox = Pow<numSamplesPerBoxDim,d>::val;

    int rank, numProcesses;
    MPI_Comm_rank( comm, &rank );
    MPI_Comm_size( comm, &numProcesses );

    if( d <= 3 )
    {
        const Box<R,d>& myTBox = u.GetMyTargetBox();
        const array<R,d>& wA = u.GetSubboxWidths();
        const array<size_t,d>& log2SubboxesPerDim = u.GetLog2SubboxesPerDim();
        const size_t numSubboxes = u.GetNumSubboxes();
        const size_t numSamples = numSamplesPerBox*numSubboxes;

        // Gather the target box coordinates to the root to write the 
        // Piece Extent data.
        const array<size_t,d>& myCoordsArray = u.GetMyTargetBoxCoords();
        vector<int> myCoords(d);
        for( size_t j=0; j<d; ++j )
            myCoords[j] = myCoordsArray[j]; // convert size_t -> int
        vector<int> coords(1);
        if( rank == 0 )
            coords.resize(d*numProcesses);
        MPI_Gather( &myCoords[0], d, MPI_INT, &coords[0], d, MPI_INT, 0, comm );

        // Have the root create the parallel file
        if( rank == 0 )
        {
            cout << "Creating parallel files...";
            cout.flush();
            ofstream realFile, imagFile;
            ostringstream os;
            os << basename << "_real.pvti";
            realFile.open( os.str().c_str() );
            os.clear(); os.str("");
            os << basename << "_imag.pvti";
            imagFile.open( os.str().c_str() );
            os.clear(); os.str("");
            os << "<?xml version=\"1.0\"?>\n"
               << "<VTKFile type=\"PImageData\" version=\"0.1\">\n"
               << " <PImageData WholeExtent=\"";
            for( size_t j=0; j<d; ++j )
            {
                os << "0 " << N*numSamplesPerBoxDim;
                if( j != d-1 )
                    os << " ";
            }
            if( d == 1 )
                os << " 0 1 0 1";
            else if( d == 2 )
                os << " 0 1";
            os << "\" Origin=\"";
            for( size_t j=0; j<d; ++j )
            {
                os << tBox.offsets[j];
                if( j != d-1 )
                    os << " ";
            }
            if( d == 1 )
                os << " 0 0";
            else if( d == 2 )
                os << " 0";
            os << "\" Spacing=\"";
            for( size_t j=0; j<d; ++j )
            {
                os << tBox.widths[j]/(N*numSamplesPerBoxDim);
                if( j != d-1 )
                    os << " ";
            }
            if( d == 1 )
                os << " 1 1";
            else if( d == 2 )
                os << " 1";
            os << "\" GhostLevel=\"0\">\n"
               << "  <PCellData Scalars=\"cell_scalars\">\n"
               << "   <PDataArray type=\"Float32\" Name=\"cell_scalars\"/>\n"
               << "  </PCellData>\n";
            for( int i=0; i<numProcesses; ++i )
            {
                os << "  <Piece Extent=\"";
                for( size_t j=0; j<d; ++j )
                {
                    size_t width = numSamplesPerBoxDim << log2SubboxesPerDim[j];
                    os << coords[i*d+j]*width << " " << (coords[i*d+j]+1)*width;
                    if( j != d-1 )
                        os << " ";
                }
                if( d == 1 )
                    os << " 0 1 0 1";
                else if( d == 2 )
                    os << " 0 1";
                realFile << os.str();
                imagFile << os.str();
                os.clear(); os.str("");
                realFile << "\" Source=\"" << basename << "_real_" << i 
                         << ".vti\"/>\n";
                imagFile << "\" Source=\"" << basename << "_imag_" << i 
                         << ".vti\"/>\n";
            }
            os << " </PImageData>\n"
               << "</VTKFile>" << endl;
            realFile << os.str();
            imagFile << os.str();
            realFile.close();
            imagFile.close();
            cout << "done" << endl;
        }

        // Have each process write its serial image data
        if( rank == 0 )
        {
            cout << "Creating serial vti files...";
            cout.flush();
        }
        ofstream realFile, imagFile;
        ostringstream os;
        os << basename << "_real_" << rank << ".vti";
        realFile.open( os.str().c_str() );
        os.clear(); os.str("");
        os << basename << "_imag_" << rank << ".vti";
        imagFile.open( os.str().c_str() );
        os.clear(); os.str("");
        os << "<?xml version=\"1.0\"?>\n"
           << "<VTKFile type=\"ImageData\" version=\"0.1\">\n"
           << " <ImageData WholeExtent=\"";
        for( size_t j=0; j<d; ++j )
        {
            os << "0 " << N*numSamplesPerBoxDim;
            if( j != d-1 )
                os << " ";
        }
        if( d == 1 )
            os << " 0 1 0 1";
        else if( d == 2 )
            os << " 0 1";
        os << "\" Origin=\"";
        for( size_t j=0; j<d; ++j )
        {
            os << tBox.offsets[j];
            if( j != d-1 )
                os << " ";
        }
        if( d == 1 )
            os << " 0 0";
        else if( d == 2 )
            os << " 0";
        os << "\" Spacing=\"";
        for( size_t j=0; j<d; ++j )
        {
            os << tBox.widths[j]/(N*numSamplesPerBoxDim);
            if( j != d-1 )
                os << " ";
        }
        if( d == 1 )
            os << " 1 1";
        else if( d == 2 )
            os << " 1";
        os << "\">\n"
           << "  <Piece Extent=\"";
        for( size_t j=0; j<d; ++j )
        {
            size_t width = numSamplesPerBoxDim << log2SubboxesPerDim[j];
            os << myCoords[j]*width << " " << (myCoords[j]+1)*width;
            if( j != d-1 )
                os << " ";
        }
        if( d == 1 )
            os << " 0 1 0 1";
        else if( d == 2 )
            os << " 0 1";
        os << "\">\n"
           << "   <CellData Scalars=\"cell_scalars\">\n"
           << "    <DataArray type=\"Float32\" Name=\"cell_scalars\""
           << " format=\"ascii\">\n";
        realFile << os.str();
        imagFile << os.str();
        os.clear(); os.str("");
        array<size_t,d> numSamplesUpToDim;
        for( size_t j=0; j<d; ++j )
        {
            numSamplesUpToDim[j] = 1;
            for( size_t i=0; i<j; ++i )
            {
                numSamplesUpToDim[j] *=
                    numSamplesPerBoxDim << log2SubboxesPerDim[i];
            }
        }
        for( size_t k=0; k<numSamples; ++k )
        {
            // Extract our indices in each dimension
            array<size_t,d> coords;
            for( size_t j=0; j<d; ++j )
                coords[j] = (k/numSamplesUpToDim[j]) %
                            (numSamplesPerBoxDim<<log2SubboxesPerDim[j]);

            // Compute the location of our sample
            array<R,d> x;
            for( size_t j=0; j<d; ++j )
                x[j] = myTBox.offsets[j] + coords[j]*wA[j]/numSamplesPerBoxDim;
            complex<R> approx = u.Evaluate( x );
            realFile << float(real(approx)) << " ";
            imagFile << float(imag(approx)) << " ";
            if( (k+1) % numSamplesPerBox == 0 )
            {
                realFile << "\n";
                imagFile << "\n";
            }
        }
        os << "    </DataArray>\n"
           << "   </CellData>\n"
           << "  </Piece>\n"
           << " </ImageData>\n"
           << "</VTKFile>" << endl;
        realFile << os.str();
        imagFile << os.str();
        realFile.close();
        imagFile.close();
        if( rank == 0 )
            cout << "done" << endl;
    }
    else
    {
        throw logic_error("VTK only supports visualizing up to 3d.");
    }
}

// Write out the real and imag components of the truth, the approximation,
// and the error.
template<typename R,size_t d,size_t q>
inline void
WriteImage
( MPI_Comm comm,
  const size_t N,
  const Box<R,d>& tBox,
  const PotentialField<R,d,q>& u,
  const string& basename,
  const vector<Source<R,d>>& sources )
{
    using namespace std;

    const size_t numSamplesPerBoxDim = 4;
    const size_t numSamplesPerBox = Pow<numSamplesPerBoxDim,d>::val;

    const Amplitude<R,d>& amplitude = u.GetAmplitude();
    const Phase<R,d>& phase = u.GetPhase();

    int rank, numProcesses;
    MPI_Comm_rank( comm, &rank );
    MPI_Comm_size( comm, &numProcesses );

    if( d <= 3 )
    {
        const Box<R,d>& myTBox = u.GetMyTargetBox();
        const array<R,d>& wA = u.GetSubboxWidths();
        const array<size_t,d>& log2SubboxesPerDim = u.GetLog2SubboxesPerDim();
        const size_t numSubboxes = u.GetNumSubboxes();
        const size_t numSamples = numSamplesPerBox*numSubboxes;

        // Gather the target box coordinates to the root to write the 
        // Piece Extent data.
        const array<size_t,d>& myCoordsArray = u.GetMyTargetBoxCoords();
        vector<int> myCoords(d);
        for( size_t j=0; j<d; ++j )
            myCoords[j] = myCoordsArray[j]; // convert size_t -> int
        vector<int> coords(1);
        if( rank == 0 )
            coords.resize(d*numProcesses);
        MPI_Gather( &myCoords[0], d, MPI_INT, &coords[0], d, MPI_INT, 0, comm );

        // Have the root create the parallel file
        if( rank == 0 )
        {
            cout << "Creating parallel files...";
            cout.flush();
            ofstream realTruthFile, imagTruthFile,
                     realApproxFile, imagApproxFile,
                     realErrorFile, imagErrorFile;
            ostringstream os;
            os << basename << "_realTruth.pvti";
            realTruthFile.open( os.str().c_str() );
            os.clear(); os.str("");
            os << basename << "_imagTruth.pvti";
            imagTruthFile.open( os.str().c_str() );
            os.clear(); os.str("");
            os << basename << "_realApprox.pvti";
            realApproxFile.open( os.str().c_str() );
            os.clear(); os.str("");
            os << basename << "_imagApprox.pvti";
            imagApproxFile.open( os.str().c_str() );
            os.clear(); os.str("");
            os << basename << "_realError.pvti";
            realErrorFile.open( os.str().c_str() );
            os.clear(); os.str("");
            os << basename << "_imagError.pvti";
            imagErrorFile.open( os.str().c_str() );
            os.clear(); os.str("");
            os << "<?xml version=\"1.0\"?>\n"
               << "<VTKFile type=\"PImageData\" version=\"0.1\">\n"
               << " <PImageData WholeExtent=\"";
            for( size_t j=0; j<d; ++j )
            {
                os << "0 " << N*numSamplesPerBoxDim;
                if( j != d-1 )
                    os << " ";
            }
            if( d == 1 )
                os << " 0 1 0 1";
            else if( d == 2 )
                os << " 0 1";
            os << "\" Origin=\"";
            for( size_t j=0; j<d; ++j )
            {
                os << tBox.offsets[j];
                if( j != d-1 )
                    os << " ";
            }
            if( d == 1 )
                os << " 0 0";
            else if( d == 2 )
                os << " 0";
            os << "\" Spacing=\"";
            for( size_t j=0; j<d; ++j )
            {
                os << tBox.widths[j]/(N*numSamplesPerBoxDim);
                if( j != d-1 )
                    os << " ";
            }
            if( d == 1 )
                os << " 1 1";
            else if( d == 2 )
                os << " 1";
            os << "\" GhostLevel=\"0\">\n"
               << "  <PCellData Scalars=\"cell_scalars\">\n"
               << "   <PDataArray type=\"Float32\" Name=\"cell_scalars\"/>\n"
               << "  </PCellData>\n";
            for( int i=0; i<numProcesses; ++i )
            {
                os << "  <Piece Extent=\"";
                for( size_t j=0; j<d; ++j )
                {
                    size_t width = numSamplesPerBoxDim << log2SubboxesPerDim[j];
                    os << coords[i*d+j]*width << " " << (coords[i*d+j]+1)*width;
                    if( j != d-1 )
                        os << " ";
                }
                if( d == 1 )
                    os << " 0 1 0 1";
                else if( d == 2 )
                    os << " 0 1";
                realTruthFile << os.str();
                imagTruthFile << os.str();
                realApproxFile << os.str();
                imagApproxFile << os.str();
                realErrorFile << os.str();
                imagErrorFile << os.str();
                os.clear(); os.str("");
                realTruthFile 
                    << "\" Source=\"" << basename << "_realTruth_" << i 
                    << ".vti\"/>\n";
                imagTruthFile 
                    << "\" Source=\"" << basename << "_imagTruth_" << i 
                    << ".vti\"/>\n";
                realApproxFile 
                    << "\" Source=\"" << basename << "_realApprox_" << i 
                    << ".vti\"/>\n";
                imagApproxFile 
                    << "\" Source=\"" << basename << "_imagApprox_" << i 
                    << ".vti\"/>\n";
                realErrorFile 
                    << "\" Source=\"" << basename << "_realError_" << i 
                    << ".vti\"/>\n";
                imagErrorFile 
                    << "\" Source=\"" << basename << "_imagError_" << i 
                    << ".vti\"/>\n";
            }
            os << " </PImageData>\n"
               << "</VTKFile>" << endl;
            realTruthFile << os.str();
            imagTruthFile << os.str();
            realApproxFile << os.str();
            imagApproxFile << os.str();
            realErrorFile << os.str();
            imagErrorFile << os.str();
            realTruthFile.close();
            imagTruthFile.close();
            realApproxFile.close();
            imagApproxFile.close();
            realErrorFile.close();
            imagErrorFile.close();
            cout << "done" << endl;
        }

        // Have each process write its serial image data
        if( rank == 0 )
        {
            cout << "Creating serial vti files...";
            cout.flush();
        }
        ofstream realTruthFile, imagTruthFile,
                 realApproxFile, imagApproxFile,
                 realErrorFile, imagErrorFile;
        ostringstream os;
        os << basename << "_realTruth_" << rank << ".vti";
        realTruthFile.open( os.str().c_str() );
        os.clear(); os.str("");
        os << basename << "_imagTruth_" << rank << ".vti";
        imagTruthFile.open( os.str().c_str() );
        os.clear(); os.str("");
        os << basename << "_realApprox_" << rank << ".vti";
        realApproxFile.open( os.str().c_str() );
        os.clear(); os.str("");
        os << basename << "_imagApprox_" << rank << ".vti";
        imagApproxFile.open( os.str().c_str() );
        os.clear(); os.str("");
        os << basename << "_realError_" << rank << ".vti";
        realErrorFile.open( os.str().c_str() );
        os.clear(); os.str("");
        os << basename << "_imagError_" << rank << ".vti";
        imagErrorFile.open( os.str().c_str() );
        os.clear(); os.str("");
        os << "<?xml version=\"1.0\"?>\n"
           << "<VTKFile type=\"ImageData\" version=\"0.1\">\n"
           << " <ImageData WholeExtent=\"";
        for( size_t j=0; j<d; ++j )
        {
            os << "0 " << N*numSamplesPerBoxDim;
            if( j != d-1 )
                os << " ";
        }
        if( d == 1 )
            os << " 0 1 0 1";
        else if( d == 2 )
            os << " 0 1";
        os << "\" Origin=\"";
        for( size_t j=0; j<d; ++j )
        {
            os << tBox.offsets[j];
            if( j != d-1 )
                os << " ";
        }
        if( d == 1 )
            os << " 0 0";
        else if( d == 2 )
            os << " 0";
        os << "\" Spacing=\"";
        for( size_t j=0; j<d; ++j )
        {
            os << tBox.widths[j]/(N*numSamplesPerBoxDim);
            if( j != d-1 )
                os << " ";
        }
        if( d == 1 )
            os << " 1 1";
        else if( d == 2 )
            os << " 1";
        os << "\">\n"
           << "  <Piece Extent=\"";
        for( size_t j=0; j<d; ++j )
        {
            size_t width = numSamplesPerBoxDim << log2SubboxesPerDim[j];
            os << myCoords[j]*width << " " << (myCoords[j]+1)*width;
            if( j != d-1 )
                os << " ";
        }
        if( d == 1 )
            os << " 0 1 0 1";
        else if( d == 2 )
            os << " 0 1";
        os << "\">\n"
           << "   <CellData Scalars=\"cell_scalars\">\n"
           << "    <DataArray type=\"Float32\" Name=\"cell_scalars\""
           << " format=\"ascii\">\n";
        realTruthFile << os.str();
        imagTruthFile << os.str();
        realApproxFile << os.str();
        imagApproxFile << os.str();
        realErrorFile << os.str();
        imagErrorFile << os.str();
        os.clear(); os.str("");
        array<size_t,d> numSamplesUpToDim;
        for( size_t j=0; j<d; ++j )
        {
            numSamplesUpToDim[j] = 1;
            for( size_t i=0; i<j; ++i )
            {
                numSamplesUpToDim[j] *=
                    numSamplesPerBoxDim << log2SubboxesPerDim[i];
            }
        }
        const size_t numSources = sources.size();
        for( size_t k=0; k<numSamples; ++k )
        {
            // Extract our indices in each dimension
            array<size_t,d> coords;
            for( size_t j=0; j<d; ++j )
                coords[j] = (k/numSamplesUpToDim[j]) %
                            (numSamplesPerBoxDim<<log2SubboxesPerDim[j]);

            // Compute the location of our sample
            array<R,d> x;
            for( size_t j=0; j<d; ++j )
                x[j] = myTBox.offsets[j] + coords[j]*wA[j]/numSamplesPerBoxDim;

            // Compute the approximation
            complex<R> approx = u.Evaluate( x );

            // Compute the 'exact' answer
            complex<R> truth(0,0);
            for( size_t m=0; m<numSources; ++m )
            {
                complex<R> beta = ImagExp<R>( phase(x,sources[m].p) );
                truth += amplitude(x,sources[m].p) * beta*sources[m].magnitude;
            }
            const complex<R> error = approx-truth;

            realTruthFile << float(real(truth)) << " ";
            imagTruthFile << float(imag(truth)) << " ";
            realApproxFile << float(real(approx)) << " ";
            imagApproxFile << float(imag(approx)) << " ";
            realErrorFile << float(abs(real(error))) << " ";
            imagErrorFile << float(abs(imag(error))) << " ";
            if( (k+1) % numSamplesPerBox == 0 )
            {
                realTruthFile << "\n";
                imagTruthFile << "\n";
                realApproxFile << "\n";
                imagApproxFile << "\n";
                realErrorFile << "\n";
                imagErrorFile << "\n";
            }
        }
        os << "    </DataArray>\n"
           << "   </CellData>\n"
           << "  </Piece>\n"
           << " </ImageData>\n"
           << "</VTKFile>" << endl;
        realTruthFile << os.str();
        imagTruthFile << os.str();
        realApproxFile << os.str();
        imagApproxFile << os.str();
        realErrorFile << os.str();
        imagErrorFile << os.str();
        realTruthFile.close();
        imagTruthFile.close();
        realApproxFile.close();
        imagApproxFile.close();
        realErrorFile.close();
        imagErrorFile.close();
        if( rank == 0 )
            cout << "done" << endl;
    }
    else
    {
        throw logic_error("VTK only supports visualizing up to 3d.");
    }
}

} // bfly
} // dbf

#endif // ifndef DBF_BFLY_POTENTIAL_FIELD_HPP
