struct Potential
----------------

.. cpp:type:: Potential<R,d>

   A struct containing the location and magnitude of a potential. ``R`` denotes 
   the datatype of the underlying real field, and ``d`` denots the dimension of
   the domain.

   .. cpp:member:: Array<R,d> x

      The location of the potential.

   .. cpp:member:: std::complex<R> magnitude

      The complex magnitude of the potential.

   **Example usage:**

   .. code-block:: cpp

      bfio::Potential<double,3> potential;
      potential.x[0] = 0.;
      potential.x[1] = 2.;
      potential.x[2] = -1.;
      potential.magnitude = std::complex<double>( 0., 1. );

