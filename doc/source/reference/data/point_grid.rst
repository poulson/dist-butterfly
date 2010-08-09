class PointGrid
---------------

.. cpp:class:: PointGrid<R,d,q>

   
   A class for manipulating a set of points representing a Cartesian product of 
   ``q`` interpolation points in each of the ``d`` dimensions. ``R`` represents 
   the datatype for the underlying real field.

   .. cpp:function:: PointGrid()

      The only constructor for the class. 

   .. cpp:function:: Array<R,d>& operator[]( unsigned i )
      
      Returns a modifiable reference to the ``j`` th index of the array of grid 
      points.

   .. cpp:function:: const Array<R,d>& operator[]( unsigned i ) const

      Returns an immutable reference to the ``j`` th index of the array of grid 
      points.

   .. cpp:function:: const PointGrid<R,d,q>& operator=( const PointGrid<R,d,q>& pointGrid )

      Copies the contents of one point grid into another. 

   **Example usage:**

   .. code-block:: cpp

      // Given a Cartesian product of Chebyshev grids, chebyGrid, translate the 
      // points by x0 and scale by alpha
      butterflyfio::PointGrid<R,d,q> pointGrid;
      const unsigned q_to_d = butterflyfio::Pow<q,d>::val;
      for( unsigned t=0; t<q_to_d; ++t )
          for( unsigned j=0; j<d; ++j )
              pointGrid[t][j] = x0[j] + alpha*chebyGrid[t][j];

