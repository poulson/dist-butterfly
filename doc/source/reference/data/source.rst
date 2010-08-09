struct Source
-------------

.. cpp:type:: Source<R,d>

   A struct containing the location and magnitude of a source. ``R`` denotes 
   the datatype of the underlying real field, and ``d`` denotes the dimension of
   the domain.

   .. cpp:member:: Array<R,d> p

      The location of the source in the frequency domain.

   .. cpp:member:: std::complex<R> magnitude

      The complex magnitude of the source.

   **Example usage:**

   .. code-block:: cpp

      butterflyfio::Source<double,3> source;
      source.p[0] = 0.;
      source.p[1] = 2.;
      source.p[2] = -1.;
      source.magnitude = std::complex<double>( 0., 1. );

