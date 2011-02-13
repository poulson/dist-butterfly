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
#ifndef BFIO_FIO_FROM_FT_POTENTIAL_FIELD_HPP
#define BFIO_FIO_FROM_FT_POTENTIAL_FIELD_HPP 1

#include <stdexcept>
#include <complex>
#include <fstream>
#include <vector>

#include "bfio/structures/array.hpp"
#include "bfio/structures/box.hpp"
#include "bfio/structures/constrained_htree_walker.hpp"
#include "bfio/structures/low_rank_potential.hpp"
#include "bfio/structures/weight_grid.hpp"
#include "bfio/structures/weight_grid_list.hpp"

#include "bfio/fio_from_ft/context.hpp"

#include "bfio/functors/amplitude_functor.hpp"
#include "bfio/functors/phase_functor.hpp"
#include "bfio/tools/special_functions.hpp"

namespace bfio {

namespace fio_from_ft {
template<typename R,std::size_t d,std::size_t q>
class PotentialField
{
    const fio_from_ft::Context<R,d,q>& _context;
    const PhaseFunctor<R,d>& _Phi;
    const Box<R,d> _sourceBox;
    const Box<R,d> _myTargetBox;
    const Array<std::size_t,d> _myTargetBoxCoords;
    const Array<std::size_t,d> _log2TargetSubboxesPerDim;

    Array<R,d> _wA;
    Array<R,d> _p0;
    Array<std::size_t,d> _log2TargetSubboxesUpToDim;
    std::vector< LRP<R,d,q> > _LRPs;

public:
    PotentialField
    ( const fio_from_ft::Context<R,d,q>& context,
      const PhaseFunctor<R,d>& Phi,
      const Box<R,d>& sourceBox,
      const Box<R,d>& myTargetBox,
      const Array<std::size_t,d>& myTargetBoxCoords,
      const Array<std::size_t,d>& log2TargetSubboxesPerDim,
      const WeightGridList<R,d,q>& weightGridList );

    std::complex<R> Evaluate( const Array<R,d>& x ) const;

    const Box<R,d>& GetMyTargetBox() const;
    std::size_t GetNumSubboxes() const;
    const Array<R,d>& GetSubboxWidths() const;
    const Array<std::size_t,d>& GetMyTargetBoxCoords() const;
    const Array<std::size_t,d>& GetLog2SubboxesPerDim() const;
    const Array<std::size_t,d>& GetLog2SubboxesUpToDim() const;
};

template<typename R,std::size_t d,std::size_t q>
void WriteVtkXmlPImageData
( MPI_Comm comm, 
  const std::size_t N,
  const Box<R,d>& targetBox,
  const PotentialField<R,d,q>& u,
  const std::string& basename );

} // fio_from_ft

// Implementations

template<typename R,std::size_t d,std::size_t q>
fio_from_ft::PotentialField<R,d,q>::PotentialField
( const fio_from_ft::Context<R,d,q>& context,
  const PhaseFunctor<R,d>& Phi,
  const Box<R,d>& sourceBox,
  const Box<R,d>& myTargetBox,
  const Array<std::size_t,d>& myTargetBoxCoords,
  const Array<std::size_t,d>& log2TargetSubboxesPerDim,
  const WeightGridList<R,d,q>& weightGridList )
: _context(context), _Phi(Phi), 
  _sourceBox(sourceBox), _myTargetBox(myTargetBox),
  _myTargetBoxCoords(myTargetBoxCoords),
  _log2TargetSubboxesPerDim(log2TargetSubboxesPerDim)
{ 
    // Compute the widths of the target subboxes and the source center
    for( std::size_t j=0; j<d; ++j )
        _wA[j] = myTargetBox.widths[j] / (1<<log2TargetSubboxesPerDim[j]);
    for( std::size_t j=0; j<d; ++j )
        _p0[j] = sourceBox.offsets[j] + sourceBox.widths[j]/2;

    // Compute the array of the partial sums
    _log2TargetSubboxesUpToDim[0] = 0;
    for( std::size_t j=1; j<d; ++j )
    {
        _log2TargetSubboxesUpToDim[j] = 
            _log2TargetSubboxesUpToDim[j-1] + log2TargetSubboxesPerDim[j-1];
    }

    // Figure out the size of our LRP vector by summing log2TargetSubboxesPerDim
    std::size_t log2TargetSubboxes = 0;
    for( std::size_t j=0; j<d; ++j )
        log2TargetSubboxes += log2TargetSubboxesPerDim[j];
    _LRPs.resize( 1<<log2TargetSubboxes );

    // The weightGridList is assumed to be ordered by the constrained 
    // HTree described by log2TargetSubboxesPerDim. We will unroll it 
    // lexographically into the LRP vector.
    ConstrainedHTreeWalker<d> AWalker( log2TargetSubboxesPerDim );
    for( std::size_t targetIndex=0; 
         targetIndex<_LRPs.size(); 
         ++targetIndex, AWalker.Walk() )
    {
        const Array<std::size_t,d> A = AWalker.State();

        // Unroll the indices of A into its lexographic position
        std::size_t k=0; 
        for( std::size_t j=0; j<d; ++j )
            k += A[j] << _log2TargetSubboxesUpToDim[j];

        // Now fill the k'th LRP index
        for( std::size_t j=0; j<d; ++j )
            _LRPs[k].x0[j] = myTargetBox.offsets[j] + (A[j]+0.5)*_wA[j];
        _LRPs[k].weightGrid = weightGridList[targetIndex];
    }
}

template<typename R,std::size_t d,std::size_t q>
std::complex<R>
fio_from_ft::PotentialField<R,d,q>::Evaluate( const Array<R,d>& x ) const
{
    typedef std::complex<R> C;

#ifndef RELEASE
    for( std::size_t j=0; j<d; ++j )
    {
        if( x[j] < _myTargetBox.offsets[j] || 
            x[j] > _myTargetBox.offsets[j]+_myTargetBox.widths[j] )
        {
            throw std::runtime_error
                  ( "Tried to evaluate outside of potential range." );
        }
    }
#endif

    // Compute the lexographic position of the LRP to use for evaluation
    std::size_t k = 0;
    for( std::size_t j=0; j<d; ++j )
    {
        std::size_t owningIndex = 
            static_cast<std::size_t>((x[j]-_myTargetBox.offsets[j])/_wA[j]);
        k += owningIndex << _log2TargetSubboxesUpToDim[j];
    }

    // Convert x to the reference domain of [-1/2,+1/2]^d for box k
    const LRP<R,d,q>& lrp = _LRPs[k];
    Array<R,d> xRef;
    for( std::size_t j=0; j<d; ++j )
        xRef[j] = (x[j]-lrp.x0[j])/_wA[j];

    const std::vector< Array<R,d> >& chebyshevGrid = 
        _context.GetChebyshevGrid();
    R realValue = 0;
    R imagValue = 0;
    for( std::size_t t=0; t<Pow<q,d>::val; ++t )
    {
        // Construct the t'th translated Chebyshev gridpoint
        Array<R,d> xt;
        for( std::size_t j=0; j<d; ++j )
            xt[j] = lrp.x0[j] + _wA[j]*chebyshevGrid[t][j];

        const C beta = ImagExp<R>( -TwoPi*_Phi(xt,_p0) );
        const R lambda = _context.Lagrange(t,xRef);
        const R realWeight = lrp.weightGrid.RealWeight(t);
        const R imagWeight = lrp.weightGrid.ImagWeight(t);
        realValue += lambda*
            (realWeight*std::real(beta)-imagWeight*std::imag(beta));
        imagValue += lambda*
            (imagWeight*std::real(beta)+realWeight*std::imag(beta));
    }
    const C beta = ImagExp<R>( TwoPi*_Phi(x,_p0) );
    const R realPotential = realValue*std::real(beta)-imagValue*std::imag(beta);
    const R imagPotential = imagValue*std::real(beta)+realValue*std::imag(beta);
    return C( realPotential, imagPotential );
}

template<typename R,std::size_t d,std::size_t q>
inline const Box<R,d>&
fio_from_ft::PotentialField<R,d,q>::GetMyTargetBox() const
{ return _myTargetBox; }

template<typename R,std::size_t d,std::size_t q>
inline std::size_t
fio_from_ft::PotentialField<R,d,q>::GetNumSubboxes() const
{ return _LRPs.size(); }

template<typename R,std::size_t d,std::size_t q>
inline const Array<R,d>&
fio_from_ft::PotentialField<R,d,q>::GetSubboxWidths() const
{ return _wA; }

template<typename R,std::size_t d,std::size_t q>
inline const Array<std::size_t,d>&
fio_from_ft::PotentialField<R,d,q>::GetMyTargetBoxCoords() const
{ return _myTargetBoxCoords; }

template<typename R,std::size_t d,std::size_t q>
inline const Array<std::size_t,d>&
fio_from_ft::PotentialField<R,d,q>::GetLog2SubboxesPerDim() const
{ return _log2TargetSubboxesPerDim; }

template<typename R,std::size_t d,std::size_t q>
inline const Array<std::size_t,d>&
fio_from_ft::PotentialField<R,d,q>::GetLog2SubboxesUpToDim() const
{ return _log2TargetSubboxesUpToDim; }

template<typename R,std::size_t d,std::size_t q>
inline void
fio_from_ft::WriteVtkXmlPImageData
( MPI_Comm comm,
  const std::size_t N,
  const Box<R,d>& targetBox,
  const fio_from_ft::PotentialField<R,d,q>& u,
  const std::string& basename )
{
    using namespace std;

    const std::size_t numSamplesPerBoxDim = 4;
    const std::size_t numSamplesPerBox = Pow<numSamplesPerBoxDim,d>::val;

    int rank, numProcesses;
    MPI_Comm_rank( comm, &rank );
    MPI_Comm_size( comm, &numProcesses );

    if( d <= 3 )
    {
        const Box<R,d>& myTargetBox = u.GetMyTargetBox();
        const Array<R,d>& wA = u.GetSubboxWidths();
        const Array<size_t,d>& log2SubboxesPerDim = u.GetLog2SubboxesPerDim();
        const size_t numSubboxes = u.GetNumSubboxes();
        const size_t numSamples = numSamplesPerBox*numSubboxes;

        // Gather the target box coordinates to the root to write the 
        // Piece Extent data.
        Array<size_t,d> myCoordsArray = u.GetMyTargetBoxCoords();
        vector<int> myCoords(d);
        for( size_t j=0; j<d; ++j )
            myCoords[j] = myCoordsArray[j]; // convert size_t -> int
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
            os.clear();
            os.str("");
            os << basename << "_imag.pvti";
            imagFile.open( os.str().c_str() );
            os.clear();
            os.str("");
            os << "<?xml version=\"1.0\"?>\n"
               << "<VTKFile type=\"PImageData\" version=\"0.1\">\n"
               << " <PImageData WholeExtent=\"";
            for( size_t j=0; j<d; ++j )
                os << "0 " << N*numSamplesPerBoxDim << " ";
            for( size_t j=d; j<3; ++j )
                os << "0 1 ";
            os << "\" Origin=\"";
            for( size_t j=0; j<d; ++j )
                os << targetBox.offsets[j] << " ";
            for( size_t j=d; j<3; ++j )
                os << "0 ";
            os << "\" Spacing=\"";
            for( size_t j=0; j<d; ++j )
                os << targetBox.widths[j]/(N*numSamplesPerBoxDim) << " ";
            for( size_t j=d; j<3; ++j )
                os << "1 ";
            os << "\" GhostLevel=\"0\">\n"
               << "  <PCellData Scalars=\"cell_scalars\">\n"
               << "   <PDataArray type=\"Float32\" Name=\"cell_scalars\"/>\n"
               << "  </PCellData>\n";
            for( size_t i=0; i<numProcesses; ++i )
            {
                os << "  <Piece Extent=\"";
                for( size_t j=0; j<d; ++j )
                {
                    size_t width = 
                        numSamplesPerBoxDim << log2SubboxesPerDim[j];
                    os << coords[i*d+j]*width << " "
                       << (coords[i*d+j]+1)*width << " ";
                }
                for( size_t j=d; j<3; ++j )
                    os << "0 1 ";
                realFile << os.str();
                imagFile << os.str();
                os.clear();
                os.str("");
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
        os.clear();
        os.str("");
        os << basename << "_imag_" << rank << ".vti";
        imagFile.open( os.str().c_str() );
        os.clear();
        os.str("");
        os << "<?xml version=\"1.0\"?>\n"
           << "<VTKFile type=\"ImageData\" version=\"0.1\">\n"
           << " <ImageData WholeExtent=\"";
        for( size_t j=0; j<d; ++j )
            os << "0 " << N*numSamplesPerBoxDim << " ";
        for( size_t j=d; j<3; ++j )
            os << "0 1 ";
        os << "\" Origin=\"";
        for( size_t j=0; j<d; ++j )
            os << targetBox.offsets[j] << " ";
        for( size_t j=d; j<3; ++j )
            os << "0 ";
        os << "\" Spacing=\"";
        for( size_t j=0; j<d; ++j )
            os << targetBox.widths[j]/(N*numSamplesPerBoxDim) << " ";
        for( size_t j=d; j<3; ++j )
            os << "1 ";
        os << "\">\n"
           << "  <Piece Extent=\"";
        for( size_t j=0; j<d; ++j )
        {
            size_t width =
                numSamplesPerBoxDim << log2SubboxesPerDim[j];
            os << myCoords[j]*width << " " << (myCoords[j]+1)*width << " ";
        }
        for( size_t j=d; j<3; ++j )
            os << "0 1 ";
        os << "\">\n"
           << "   <CellData Scalars=\"cell_scalars\">\n"
           << "    <DataArray type=\"Float32\" Name=\"cell_scalars\""
           << " format=\"ascii\">\n";
        realFile << os.str();
        imagFile << os.str();
        os.clear();
        os.str("");
        Array<size_t,d> numSamplesUpToDim;
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
            Array<size_t,d> coords;
            for( size_t j=0; j<d; ++j )
                coords[j] = (k/numSamplesUpToDim[j]) %
                            (numSamplesPerBoxDim<<log2SubboxesPerDim[j]);

            // Compute the location of our sample
            Array<R,d> x;
            for( size_t j=0; j<d; ++j )
                x[j] = myTargetBox.offsets[j] +
                       coords[j]*wA[j]/numSamplesPerBoxDim;
            complex<R> approx = u.Evaluate( x );
            realFile << (float)real(approx) << " ";
            imagFile << (float)imag(approx) << " ";
            if( k % numSamplesPerBox == 0 )
            {
                realFile << "\n";
                imagFile << "\n";
            }
        }
        os << "\n"
           << "    </DataArray>\n"
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

} // bfio

#endif // BFIO_FIO_FROM_FT_POTENTIAL_FIELD_HPP

