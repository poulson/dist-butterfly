#
# ButterflyFIO: An abstract distributed-memory implementation of the
#               butterfly algorithm for Fourier Integral Operators
#
# Copyright 2010 Jack Poulson
#

incdir = include
testdir = test
bindir = bin

# Defining 'AVOID_COMPLEX_MPI' avoids buggy complex MPI_Reduce_scatter summation
# implementations by using the real version with doubled lengths
#
# Defining 'FUNDERSCORE' appends an underscore to BLAS routine names
CXX = mpicxx
CXXFLAGS = -I$(incdir) -DFUNDERSCORE -DAVOID_COMPLEX_MPI
CXXFLAGS_DEBUG = -g -Wall $(CXXFLAGS)
CXXFLAGS_RELEASE = -O3 -Wall -DRELEASE $(CXXFLAGS)
LDFLAGS = -L/usr/lib -lblas
AR = ar
ARFLAGS = rc

################################################################################
# Only developers should edit past this point.                                 #
################################################################################

includefiles = bfio.hpp \
               bfio/constants.hpp \
               bfio/freq_to_spatial.hpp \
               bfio/freq_to_spatial/freq_weight_recursion.hpp \
               bfio/freq_to_spatial/initialize_weights.hpp \
               bfio/freq_to_spatial/spatial_weight_recursion.hpp \
               bfio/freq_to_spatial/switch_to_spatial_interp.hpp  \
               bfio/structures/data.hpp \
               bfio/structures/htree_walker.hpp \
               bfio/structures/low_rank_potential.hpp \
               bfio/structures/low_rank_source.hpp \
               bfio/structures/phase_functor.hpp \
               bfio/tools/blas.hpp \
               bfio/tools/flatten_htree_index.hpp \
               bfio/tools/lagrange.hpp \
               bfio/tools/local_data.hpp \
               bfio/tools/mpi.hpp \
               bfio/tools/twiddle.hpp \
               bfio/tools/uniform.hpp 

includes = $(addprefix $(incdir)/,$(includefiles))

################################################################################
# make                                                                         #
################################################################################
bindir_debug = $(bindir)/debug
bindir_release = $(bindir)/release

tests = transform/GeneralizedRadon \
        transform/UpWave \
        transform/Random3DWaves \
        transform/VariableUpWave \
        tree/CHTreeWalker \
        tree/HTreeWalker
testobjs = $(addsuffix .o, $(tests))

tests_debug = $(addprefix $(bindir_debug)/, $(tests))
testobjs_debug = $(addprefix $(bindir_debug)/, $(testobjs))
tests_release = $(addprefix $(bindir_release)/, $(tests))

.PHONY : test
test: test-release test-debug

.PHONY : test-debug
test-debug: $(tests_debug) $(testobjs_debug)

.PHONY : test-release
test-release: $(tests_release)

$(bindir_debug)/%: $(bindir_debug)/%.o $(library_debug)
	@echo "[ debug ] Creating $@"
	@$(CXX) -o $@ $^ $(LDFLAGS)

$(bindir_release)/%: $(bindir_release)/%.o $(library_release)
	@echo "[release] Creating $@"
	@$(CXX) -o $@ $^ $(LDFLAGS)

$(bindir_debug)/%.o: $(testdir)/%.cpp $(includes)
	@mkdir -p $(dir $@)
	@echo "[ debug ] Compiling $<"
	@$(CXX) $(CXXFLAGS_DEBUG) -c -o $@ $<

$(bindir_release)/%.o: $(testdir)/%.cpp $(includes)
	@mkdir -p $(dir $@)
	@echo "[release] Compiling $<"
	@$(CXX) $(CXXFLAGS_RELEASE) -c -o $@ $<

################################################################################
# make clean                                                                   #
################################################################################
.PHONY : clean
clean: 
	@rm -Rf bin/

.PHONY : clean-debug
clean-debug:
	@rm -Rf bin/debug

.PHONY : clean-release
clean-release:
	@rm -Rf bin/release

