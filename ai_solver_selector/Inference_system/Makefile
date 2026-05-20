# Builds collect_heat_data using the same MFEM config that the regular
# examples use. Run from this directory:
#   make
#
# By default MFEM is expected at the repo root (../../).

MFEM_DIR ?= ../..
MFEM_BUILD_DIR ?= ../../build
MFEM_INSTALL_DIR ?= ../../mfem
CONFIG_MK = $(or $(wildcard $(MFEM_BUILD_DIR)/config/config.mk),\
   $(wildcard $(MFEM_INSTALL_DIR)/share/mfem/config.mk))

MFEM_LIB_FILE = mfem_is_not_built
-include $(CONFIG_MK)

EXTRA_CFLAGS = -I.

PROGRAMS = heat_demo

.PHONY: all clean
.SUFFIXES:

all: $(PROGRAMS)

heat_demo: heat_demo.cpp  \
                      $(MFEM_LIB_FILE) $(CONFIG_MK)
	$(MFEM_CXX) $(MFEM_FLAGS) $(EXTRA_CFLAGS) heat_demo.cpp -o $@ $(MFEM_LIBS)

clean:
	rm -f $(PROGRAMS) *.o