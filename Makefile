#
#  ButterflyFIO: a distributed-memory fast algorithm for applying FIOs.
#  Copyright (C) 2010 Jack Poulson <jack.poulson@gmail.com>
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

# The GNU configuration is fairly portable, but the Intel configuration has 
# only been tested on TACC's Ranger, and the IBM one has only been tested on 
# a Blue Gene/P. The apple implementation has been tested on my laptop; its 
# only difference from gnu is the additional '-fast' compilation flag.
config = apple
ifneq ($(config),ibm)
  ifneq ($(config),intel)
    ifneq ($(config),gnu)
      ifneq ($(config),apple)
        $(error You must choose a valid configuration)
      endif
    endif
  endif
endif

incdir = include
testdir = test
bindir_base = bin
bindir = $(bindir_base)/$(config)

# Defining 'BLAS_UNDERSCORE' appends an underscore to BLAS routine names
# Defining 'LAPACK_UNDERSCORE' appends an underscore to LAPACK routine names
# Defining 'RELEASE' removes all unnecessary output and checks
ifeq ($(config),ibm)
  # This is for ANL's Blue Gene/P
  CXX = mpixlcxx_r
  ESSL_INC = /soft/apps/ESSL-4.3.1-1/include
  ESSL_LIB = /soft/apps/ESSL-4.3.1-1/lib
  XLF_LIB = /soft/apps/ibmcmp-aug2010/xlf/bg/11.1/bglib
  XLSMP_LIB = /soft/apps/ibmcmp-aug2010/xlsmp/bg/1.7/bglib
  XLMASS_INC = /soft/apps/ibmcmp-aug2010/xlmass/bg/4.4/include
  XLMASS_LIB = /soft/apps/ibmcmp-aug2010/xlmass/bg/4.4/bglib
  CXXFLAGS = -DIBM -I$(incdir) -I$(XLMASS_INC)
  CXXFLAGS_DEBUG = -g $(CXXFLAGS)
  CXXFLAGS_RELEASE = -O4 -DRELEASE $(CXXFLAGS)
  LDFLAGS = -L$(ESSL_LIB) -L$(XLF_LIB) -L$(XLSMP_LIB) -L$(XLMASS_LIB) \
            -lesslbg -lxlfmath -lxlf90_r -lxlomp_ser -lmassv -lmass
endif
ifeq ($(config),intel)
  # This is for TACC's Ranger
  CXX = mpicxx 
  MKL_INC = /opt/apps/intel/mkl/10.0.1.014/include
  MKL_LIB = /opt/apps/intel/mkl/10.0.1.014/lib/em64t
  CXXFLAGS = -DINTEL -I$(incdir) -I$(MKL_INC)
  CXXFLAGS_DEBUG = -g $(CXXFLAGS)
  CXXFLAGS_RELEASE = -O3 -DRELEASE $(CXXFLAGS)
  LDFLAGS = -Wl,-rpath,$(MKL_LIB) -L$(MKL_LIB) \
            -lmkl_em64t -lmkl -lguide -lpthread
endif
ifeq ($(config),gnu)
  # This is for a generic Linux machine with a BLAS/LAPACK libraries in /usr/lib
  CXX = mpicxx 
  CXXFLAGS = -DGNU -I$(incdir) -DBLAS_UNDERSCORE  -DLAPACK_UNDERSCORE
  CXXFLAGS_DEBUG = -g -Wall $(CXXFLAGS)
  CXXFLAGS_RELEASE = -O3 -ffast-math -Wall -DRELEASE $(CXXFLAGS)
  LDFLAGS = -L/usr/lib -llapack -lblas
endif
ifeq ($(config),apple)
  # This is for a Mac with a BLAS/LAPACK libraries in /usr/lib
  CXX = mpicxx 
  CXXFLAGS = -DGNU -I$(incdir) -DBLAS_UNDERSCORE -DLAPACK_UNDERSCORE
  CXXFLAGS_DEBUG = -g -Wall $(CXXFLAGS)
  CXXFLAGS_RELEASE = -fast -ffast-math -Wall -DRELEASE $(CXXFLAGS)
  LDFLAGS = -L/usr/lib -llapack -lblas
endif

AR = ar
ARFLAGS = rc

################################################################################
# Only developers should edit past this point.                                 #
################################################################################

includefiles = bfio.hpp \
               bfio/constants.hpp \
               bfio/functors/amplitude_functor.hpp \
               bfio/functors/phase_functor.hpp \
               bfio/general_fio.hpp \
               bfio/general_fio/context.hpp \
               bfio/general_fio/initialize_weights.hpp \
               bfio/general_fio/potential_field.hpp \
               bfio/general_fio/source_weight_recursion.hpp \
               bfio/general_fio/switch_to_target_interp.hpp \
               bfio/general_fio/target_weight_recursion.hpp \
               bfio/lagrangian_nuft.hpp \
               bfio/lagrangian_nuft/context.hpp \
               bfio/lagrangian_nuft/dot_product.hpp \
               bfio/lagrangian_nuft/potential_field.hpp \
               bfio/lagrangian_nuft/switch_to_target_interp.hpp \
               bfio/interpolative_nuft.hpp \
               bfio/interpolative_nuft/context.hpp \
               bfio/interpolative_nuft/form_check_potentials.hpp \
               bfio/interpolative_nuft/form_equivalent_sources.hpp \
               bfio/interpolative_nuft/initialize_check_potentials.hpp \
               bfio/interpolative_nuft/potential_field.hpp \
               bfio/structures/array.hpp \
               bfio/structures/box.hpp \
               bfio/structures/constrained_htree_walker.hpp \
               bfio/structures/htree_walker.hpp \
               bfio/structures/low_rank_potential.hpp \
               bfio/structures/plan.hpp \
               bfio/structures/point_grid.hpp \
               bfio/structures/source.hpp \
               bfio/structures/weight_grid.hpp \
               bfio/structures/weight_grid_list.hpp \
               bfio/tools/blas.hpp \
               bfio/tools/flatten_constrained_htree_index.hpp \
               bfio/tools/flatten_htree_index.hpp \
               bfio/tools/lapack.hpp \
               bfio/tools/mpi.hpp \
               bfio/tools/special_functions.hpp \
               bfio/tools/twiddle.hpp \
               bfio/tools/uniform.hpp 

includes = $(addprefix $(incdir)/,$(includefiles))

################################################################################
# make                                                                         #
################################################################################
bindir_debug = $(bindir)/debug
bindir_release = $(bindir)/release

tests = htree/ConstrainedHTreeWalker \
        htree/HTreeWalker \
        transform/GeneralizedRadon \
        transform/NonUniformFT \
        transform/UpWave \
        transform/Random3DWaves \
        transform/VariableUpWave 
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
.PHONY : real-clean
real-clean: 
	@rm -Rf $(bindir_base)

.PHONY : clean
clean: 
	@rm -Rf $(bindir)

.PHONY : clean-debug
clean-debug:
	@rm -Rf $(bindir_debug)

.PHONY : clean-release
clean-release:
	@rm -Rf $(bindir_release)

