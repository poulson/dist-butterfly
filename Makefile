#
# ButterflyFIO: An abstract distributed-memory implementation of the
#               butterfly algorithm for discrete Fourier Integral Operators
#
# Copyright 2010 Jack Poulson
#

incdir = include
testdir = test
bindir = bin

CXX = mpicxx
CXXFLAGS = -I$(incdir) -DFUNDERSCORE
CXXFLAGS_DEBUG = -g -Wall $(CXXFLAGS)
CXXFLAGS_RELEASE = -O3 -Wall -DRELEASE $(CXXFLAGS)
LDFLAGS = -L/usr/lib -lblas
AR = ar
ARFLAGS = rc

################################################################################
# Only developers should edit past this point.                                 #
################################################################################

includefiles = BFIO.hpp \
               BFIO/Structures/Data.hpp \
               BFIO/Structures/HTree.hpp \
               BFIO/Structures/LRP.hpp \
               BFIO/Structures/PhaseFunctor.hpp \
               BFIO/Tools/BLAS.hpp \
               BFIO/Tools/Lagrange.hpp \
               BFIO/Tools/MPI.hpp \
               BFIO/Tools/Pow.hpp \
               BFIO/Tools/Twiddle.hpp \
               BFIO/Transform.hpp \
               BFIO/Transform/FreqWeightRecursion.hpp \
               BFIO/Transform/InitializeWeights.hpp \
               BFIO/Transform/SpatialWeightRecursion.hpp \
               BFIO/Transform/SwitchToSpatialInterp.hpp 

includes = $(addprefix $(incdir)/,$(includefiles))

################################################################################
# make                                                                         #
################################################################################
bindir_debug = $(bindir)/debug
bindir_release = $(bindir)/release

tests = Transform/3DWave \
        Transform/Accuracy \
        Tree/CHTreeWalker \
        Tree/HTreeWalker
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

