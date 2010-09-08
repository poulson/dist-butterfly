class Array
-----------

.. cpp:class:: Array<T,d>

   A simple template class for arrays of datatype ``T`` of length ``d``.

   .. cpp:function:: Array()

      Default constructor for ``Array`` class. 
      **Example usage:**

      .. code-block:: cpp

         bfio::Array<double,3> point;

   .. cpp:function:: Array( T alpha )

      Initializes all entries of the array to the value ``alpha``.
      **Example usage:**

      .. code-block:: cpp

         bfio::Array<int,4> indices(0);   

   .. cpp:function:: T& operator[]( unsigned j )

      Returns a modifiable reference to the ``j`` th index in the array.
      **Example usage:**

      .. code-block:: cpp

         bfio::Array<float,2> ratios;
         ratios[0] = 0.5f;
         ratios[1] = 1.0f;

   .. cpp:function:: const T& operator[]( unsigned j ) const

      Returns an immutable reference to the ``j`` th index in the array. This 
      is used when working with constant arrays.
      **Example usage:**

      .. code-block:: cpp

         template<typename T,unsigned d>
         void PrintArray( const bfio::Array<T,d>& array )
         {
             for( unsigned j=0; j<d; ++j )
                 std::cout << array[j] << " ";
             std::cout << std::endl;
         }
   .. cpp:function:: const Array<T,d>& operator=( const Array<T,d>& array )

      Copies the contents of one array into another. **Example usage:**

      .. code-block:: cpp

         bfio::Array<int,3> a;
         bfio::Array<int,3> b(5);
         a = b;

