class WeightGrid
---------------

.. cpp:class:: WeightGrid<R,d,q>

   A class for manipulating a set of weights over a Cartesian product of 
   ``q`` interpolation points in each of the ``d`` dimensions. ``R`` represents 
   the datatype for the underlying real field.

   .. cpp:function:: WeightGrid()

      The only constructor for the class. 

   .. cpp:function:: std::complex<R>& operator[]( unsigned i )
      
      Returns a modifiable reference to the ``j`` th index of the array of grid 
      points.

   .. cpp:function:: const std::complex<R>& operator[]( unsigned i ) const

      Returns an immutable reference to the ``j`` th index of the array of grid 
      points.

   .. cpp:function:: const WeightGrid<R,d,q>& operator=( const WeightGrid<R,d,q>& weightGrid )

      Copies the contents of one weight grid into another. 

   **Example usage:**

   .. code-block:: cpp

      // Double a set of weights
      bfio::WeightGrid<R,d,q> weightGrid;
      const unsigned q_to_d = bfio::Pow<q,d>::val;
      for( unsigned t=0; t<q_to_d; ++t )
          weightGrid[t] = 2*oldWeightGrid[t];

