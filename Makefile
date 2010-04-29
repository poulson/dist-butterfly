#
# ButterflyFIO: An abstract distributed-memory implementation of the
#               butterfly algorithm for discrete Fourier Integral Operators
#
# Copyright 2010 Jack Poulson
#

incdir = include
testdir = test
bindir = bin

CXX = mpicxx.mpich2
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
               BFIO/BLAS.hpp \
               BFIO/Data.hpp \
               BFIO/FreqWeightRecursion.hpp \
               BFIO/HTree.hpp \
               BFIO/InitializeWeights.hpp \
               BFIO/Lagrange.hpp \
               BFIO/LRP.hpp \
               BFIO/MPI.hpp \
               BFIO/Pow.hpp \
               BFIO/SpatialWeightRecursion.hpp \
               BFIO/SwitchToSpatialInterp.hpp \
               BFIO/Transform.hpp \
               BFIO/Util.hpp
includes = $(addprefix $(incdir)/,$(includefiles))

################################################################################
# make                                                                         #
################################################################################
bindir_debug = $(bindir)/debug
bindir_release = $(bindir)/release

tests = CHTreeWalker \
        HTreeWalker \
        Transform
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

