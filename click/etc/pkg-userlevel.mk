# pkg-userlevel.mk -- build tools for Click
# Eddie Kohler
#
# Copyright (c) 2006-2007 Regents of the University of California
# Copyright (c) 2011 Eddie Kohler
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, subject to the conditions
# listed in the Click LICENSE file. These conditions include: you must
# preserve this copyright notice, and you cannot mention the copyright
# holders in advertising related to the Software without their permission.
# The Software is provided WITHOUT ANY WARRANTY, EXPRESS OR IMPLIED. This
# notice is a summary of the Click LICENSE file; the license in that file is
# legally binding.

CLICKBUILD = userlevel

CC ?= $(CLICKCC)
CPP ?= $(CLICKCPP)
CXX ?= $(CLICKCXX)
CXXCPP ?= $(CLICKCXXCPP)
AR_CREATE ?= $(CLICKAR_CREATE)
RANLIB ?= $(CLICKRANLIB)
STRIP ?= $(CLICKSTRIP)

CPPFLAGS ?= $(CLICKCPPFLAGS) -DCLICK_USERLEVEL
CFLAGS ?= $(CLICKCFLAGS) -fPIC
CXXFLAGS ?= $(CLICKCXXFLAGS) -fPIC
DEPCFLAGS ?= $(CLICKDEPCFLAGS)

DEFS ?= $(CLICKDEFS)
INCLUDES ?= $(CLICKINCLUDES)
LDFLAGS ?= $(CLICKLDMODULEFLAGS)

packagesrcdir ?= $(srcdir)
PACKAGE_OBJS ?= upackage.uo
PACKAGE_LIBS ?=
PACKAGE_DEPS ?=

STRIP_UPACKAGE ?= true

CXXCOMPILE = $(CXX) $(CPPFLAGS) $(CXXFLAGS) $(PACKAGE_CXXFLAGS) $(DEFS) $(INCLUDES)
CXXLD = $(CXX)
CXXLINK = $(CXXLD) $(CXXFLAGS) $(LDFLAGS) -o $@
COMPILE = $(CC) $(CPPFLAGS) $(CFLAGS) $(PACKAGE_CFLAGS) $(DEFS) $(INCLUDES)
CCLD = $(CC)
LINK = $(CCLD) $(CFLAGS) $(LDFLAGS) -o $@
FIXDEP = @-sed 's/\.o:/\.uo:/' < $*.d > $*.ud; /bin/rm -f $*.d

ifeq ($(V),1)
ccompile = $(COMPILE) $(DEPCFLAGS) $(1)
ccompile_nodep = $(COMPILE) $(1)
cxxcompile = $(CXXCOMPILE) $(DEPCFLAGS) $(1)
cxxcompile_nodep = $(CXXCOMPILE) $(1)
else
ccompile = @/bin/echo ' ' $(2) $< && $(COMPILE) $(DEPCFLAGS) $(1)
ccompile_nodep = @/bin/echo ' ' $(2) $< && $(COMPILE) $(1)
cxxcompile = @/bin/echo ' ' $(2) $< && $(CXXCOMPILE) $(DEPCFLAGS) $(1)
cxxcompile_nodep = @/bin/echo ' ' $(2) $< && $(CXXCOMPILE) $(1)
endif

.SUFFIXES:
.SUFFIXES: .c .cc .uo .uii

.c.uo:
	$(call ccompile,-c $< -o $@,CC)
	$(FIXDEP)
.cc.uo:
	$(call cxxcompile,-c $< -o $@,CXX)
	$(FIXDEP)
.cc.uii:
	$(call cxxcompile_nodep,-E $< > $@,CXXCPP)

ifneq ($(MAKECMDGOALS),clean)
-include uelements.mk
endif

OBJS = $(ELEMENT_OBJS) $(PACKAGE_OBJS)

$(package).uo: $(clickbuild_datadir)/pkg-userlevel.mk $(OBJS) $(PACKAGE_DEPS)
	$(CXXLINK) -o $(package).uo $(OBJS) $(ELEMENT_LIBS) $(PACKAGE_LIBS)
	$(STRIP_UPACKAGE) $(package).uo

elemlist uelements.conf: $(CLICK_BUILDTOOL)
	echo $(packagesrcdir) | $(CLICK_BUILDTOOL) $(CLICK_BUILDTOOL_FLAGS) findelem -r userlevel -r $(package) -P $(CLICKFINDELEMFLAGS) > uelements.conf
uelements.mk: uelements.conf $(CLICK_BUILDTOOL)
	$(CLICK_BUILDTOOL) $(CLICK_BUILDTOOL_FLAGS) elem2make -t userlevel < uelements.conf > uelements.mk
upackage.cc: uelements.conf $(CLICK_BUILDTOOL)
	$(CLICK_ELEM2PACKAGE) $(package) < uelements.conf > upackage.cc
	@rm -f upackage.ud

DEPFILES := $(wildcard *.ud)
ifneq ($(DEPFILES),)
include $(DEPFILES)
endif

clean:
	-rm -f *.ud *.uo uelements.conf uelements.mk upackage.cc $(package).uo

.PHONY: clean elemlist
