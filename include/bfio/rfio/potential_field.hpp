/*
   Copyright (C) 2010-2013 Jack Poulson and Lexing Ying
 
   This file is part of ButterflyFIO and is under the GNU General Public 
   License, which can be found in the LICENSE file in the root directory, or at
   <http://www.gnu.org/licenses/>.
*/
#pragma once
#ifndef BFIO_RFIO_POTENTIAL_FIELD_HPP
#define BFIO_RFIO_POTENTIAL_FIELD_HPP

#include <array>
#include <complex>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "bfio/structures/box.hpp"
#include "bfio/structures/constrained_htree_walker.hpp"
#include "bfio/structures/low_rank_potential.hpp"
#include "bfio/structures/weight_grid.hpp"
#include "bfio/structures/weight_grid_list.hpp"

#include "bfio/rfio/context.hpp"

#include "bfio/functors/amplitude.hpp"
#include "bfio/functors/phase.hpp"
#include "bfio/tools/special_functions.hpp"

namespace bfio {

using std::array;
using std::complex;
using std::size_t;
using std::string;
using std::vector;

namespace rfio {

template<typename R,size_t d,size_t q>
class PotentialField
{
    const Context<R,d,q>& _context;
    const Amplitude<R,d>* _amplitude;
    const Phase<R,d>* _phase;
    const Box<R,d> _sBox;
    const Box<R,d> _myTBox;
    const array<size_t,d> _myTBoxCoords;
    const array<size_t,d> _log2TSubboxesPerDim;

    array<R,d> _wA;
    array<R,d> _p0;
    array<size_t,d> _log2TSubboxesUpToDim;
    vector<LRP<R,d,q>> _LRPs;

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
: _context(context), _amplitude(amplitude.Clone()), _phase(phase.Clone()), 
  _sBox(sBox), _myTBox(myTBox), _myTBoxCoords(myTBoxCoords),
  _log2TSubboxesPerDim(log2TSubboxesPerDim)
{ 
    // Compute the widths of the target subboxes and the source center
    for( size_t j=0; j<d; ++j )
        _wA[j] = myTBox.widths[j] / (1<<log2TSubboxesPerDim[j]);
    for( size_t j=0; j<d; ++j )
        _p0[j] = sBox.offsets[j] + sBox.widths[j]/2;

    // Compute the array of the partial sums
    _log2TSubboxesUpToDim[0] = 0;
    for( size_t j=1; j<d; ++j )
    {
        _log2TSubboxesUpToDim[j] = 
            _log2TSubboxesUpToDim[j-1] + log2TSubboxesPerDim[j-1];
    }

    // Figure out the size of our LRP vector by summing log2TargetSubboxesPerDim
    size_t log2TSubboxes = 0;
    for( size_t j=0; j<d; ++j )
        log2TSubboxes += log2TSubboxesPerDim[j];
    _LRPs.resize( 1<<log2TSubboxes );

    // The weightGridList is assumed to be ordered by the constrained 
    // HTree described by log2TSubboxesPerDim. We will unroll it 
    // lexographically into the LRP vector.
    ConstrainedHTreeWalker<d> AWalker( log2TSubboxesPerDim );
    for( size_t tIndex=0; tIndex<_LRPs.size(); ++tIndex, AWalker.Walk() )
    {
        const array<size_t,d> A = AWalker.State();

        // Unroll the indices of A into its lexographic position
        size_t k=0; 
        for( size_t j=0; j<d; ++j )
            k += A[j] << _log2TSubboxesUpToDim[j];

        // Now fill the k'th LRP index
        for( size_t j=0; j<d; ++j )
            _LRPs[k].x0[j] = myTBox.offsets[j] + (A[j]+0.5)*_wA[j];
        _LRPs[k].weightGrid = weightGridList[tIndex];
    }
}

template<typename R,size_t d,size_t q>
inline
PotentialField<R,d,q>::~PotentialField()
{
    delete _amplitude;
    delete _phase;
}

template<typename R,size_t d,size_t q>
inline complex<R>
PotentialField<R,d,q>::Evaluate( const array<R,d>& x ) const
{
    typedef complex<R> C;

#ifndef RELEASE
    for( size_t j=0; j<d; ++j )
    {
        if( x[j] < _myTBox.offsets[j] || 
            x[j] > _myTBox.offsets[j]+_myTBox.widths[j] )
            throw std::runtime_error
            ("Tried to evaluate outside of potential range");
    }
#endif

    // Compute the lexographic position of the LRP to use for evaluation
    size_t k = 0;
    for( size_t j=0; j<d; ++j )
    {
        size_t owningIndex = size_t((x[j]-_myTBox.offsets[j])/_wA[j]);
        k += owningIndex << _log2TSubboxesUpToDim[j];
    }

    // Convert x to the reference domain of [-1/2,+1/2]^d for box k
    const LRP<R,d,q>& lrp = _LRPs[k];
    array<R,d> xRef;
    for( size_t j=0; j<d; ++j )
        xRef[j] = (x[j]-lrp.x0[j])/_wA[j];

    const vector<array<R,d>>& chebyshevGrid = 
        _context.GetChebyshevGrid();
    R realValue = 0;
    R imagValue = 0;
    for( size_t t=0; t<Pow<q,d>::val; ++t )
    {
        // Construct the t'th translated Chebyshev gridpoint
        array<R,d> xt;
        for( size_t j=0; j<d; ++j )
            xt[j] = lrp.x0[j] + _wA[j]*chebyshevGrid[t][j];

        const C beta = ImagExp<R>( -_phase->operator()(xt,_p0) );
        const R lambda = _context.Lagrange(t,xRef);
        const R realWeight = lrp.weightGrid.RealWeight(t);
        const R imagWeight = lrp.weightGrid.ImagWeight(t);
        realValue += lambda*
            (realWeight*std::real(beta)-imagWeight*std::imag(beta));
        imagValue += lambda*
            (imagWeight*std::real(beta)+realWeight*std::imag(beta));
    }
    const C beta = ImagExp<R>( _phase->operator()(x,_p0) );
    const R realPotential = realValue*std::real(beta)-imagValue*std::imag(beta);
    const R imagPotential = imagValue*std::real(beta)+realValue*std::imag(beta);
    return C( realPotential, imagPotential );
}

template<typename R,size_t d,size_t q>
inline const Amplitude<R,d>&
PotentialField<R,d,q>::GetAmplitude() const
{ return *_amplitude; }

template<typename R,size_t d,size_t q>
inline const Phase<R,d>&
PotentialField<R,d,q>::GetPhase() const
{ return *_phase; }

template<typename R,size_t d,size_t q>
inline const Box<R,d>&
PotentialField<R,d,q>::GetMyTargetBox() const
{ return _myTBox; }

template<typename R,size_t d,size_t q>
inline size_t
PotentialField<R,d,q>::GetNumSubboxes() const
{ return _LRPs.size(); }

template<typename R,size_t d,size_t q>
inline const array<R,d>&
PotentialField<R,d,q>::GetSubboxWidths() const
{ return _wA; }

template<typename R,size_t d,size_t q>
inline const array<size_t,d>&
PotentialField<R,d,q>::GetMyTargetBoxCoords() const
{ return _myTBoxCoords; }

template<typename R,size_t d,size_t q>
inline const array<size_t,d>&
PotentialField<R,d,q>::GetLog2SubboxesPerDim() const
{ return _log2TSubboxesPerDim; }

template<typename R,size_t d,size_t q>
inline const array<size_t,d>&
PotentialField<R,d,q>::GetLog2SubboxesUpToDim() const
{ return _log2TSubboxesUpToDim; }

template<typename R,size_t d,size_t q>
void PrintErrorEstimates
( MPI_Comm comm,
  const PotentialField<R,d,q>& u,
  const vector<Source<R,d>>& sources )
{
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
            x[j] = myTBox.offsets[j] + myTBox.widths[j]*Uniform<R>();

        // Evaluate our potential field at x and compare against truth
        complex<R> approx = u.Evaluate( x );
        complex<R> truth(0.,0.);
        for( size_t m=0; m<numSources; ++m )
        {
            complex<R> beta =
                amplitude( x, sources[m].p ) * ImagExp( phase(x,sources[m].p) );
            truth += beta * sources[m].magnitude;
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
        std::cout << "---------------------------------------------\n"
                  << "Estimate of relative ||e||_2:    "
                  << sqrt(L2ErrorSquared/L2TruthSquared) << "\n"
                  << "Estimate of ||e||_inf:           "
                  << LinfError << "\n"
                  << "||f||_1:                         "
                  << L1Sources << "\n"
                  << "Estimate of ||e||_inf / ||f||_1: "
                  << LinfError/L1Sources << "\n" 
                  << "---------------------------------------------\n"
                  << std::endl;
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
        array<size_t,d> myCoordsarray = u.GetMyTargetBoxCoords();
        vector<int> myCoords(d);
        for( size_t j=0; j<d; ++j )
            myCoords[j] = myCoordsarray[j]; // convert size_t -> int
        vector<int> coords(1);
        if( rank == 0 )
            coords.resize(d*numProcesses);
        MPI_Gather
        ( &myCoords[0], d, MPI_INT, &coords[0], d, MPI_INT, 0, comm );

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
        array<size_t,d> myCoordsarray = u.GetMyTargetBoxCoords();
        vector<int> myCoords(d);
        for( size_t j=0; j<d; ++j )
            myCoords[j] = myCoordsarray[j]; // convert size_t -> int
        vector<int> coords(1);
        if( rank == 0 )
            coords.resize(d*numProcesses);
        MPI_Gather
        ( &myCoords[0], d, MPI_INT, &coords[0], d, MPI_INT, 0, comm );

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

} // rfio
} // bfio

#endif // ifndef BFIO_RFIO_POTENTIAL_FIELD_HPP
