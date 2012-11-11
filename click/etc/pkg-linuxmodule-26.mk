# pkg-linuxmodule-26.mk -- build tools for Click
# Eddie Kohler
#
# Copyright (c) 2006-2007 Regents of the University of California
# Copyright (c) 2008 Meraki, Inc.
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

CLICKBUILD = linux26module

CLICKCPPFLAGS += -DCLICK_LINUXMODULE
CLICKINCLUDES := -I$(clickbuild_includedir) -I$(clickbuild_srcdir)

LINUXCFLAGS = $(shell echo "$(CPPFLAGS) $(CFLAGS) $(LINUXINCLUDE)" \
	"$(KBUILD_CPPFLAGS) $(KBUILD_CFLAGS) $(CFLAGS_MODULE)" | sed \
	-e s,-fno-unit-at-a-time,, -e s,-Wstrict-prototypes,, \
	-e s,-Wdeclaration-after-statement,, \
	-e s,-Wno-pointer-sign,, -e s,-fno-common,, \
	-e s,-Werror-implicit-function-declaration,, \
	-e "s,-Iinclude ,-I$(CLICKLINUX_BUILDDIR)/include ",g \
	-e "s,-Iinclude2 ,-I$(CLICKLINUX_BUILDDIR)/include2 ",g \
	-e s,-Iinclude/,-I$(CLICKLINUX_SRCDIR)include/,g $(CLICKLINUX_FIXINCLUDES_PROGRAM))

CXXFLAGS ?= $(CLICKKERNEL_CXXFLAGS)
DEPCFLAGS ?= -Wp,-MD,$(depfile)

DEFS ?= $(CLICKDEFS)
INCLUDES ?= $(CLICKINCLUDES)

CXXCOMPILE = $(CLICKKERNEL_CXX) $(LINUXCFLAGS) $(CLICKCPPFLAGS) \
	$(CLICKKERNEL_CFLAGS) $(CXXFLAGS) $(PACKAGE_CXXFLAGS) \
	$(DEFS) $(INCLUDES)
COMPILE = $(CLICKKERNEL_CC) $(LINUXCFLAGS) $(CLICKCPPFLAGS) \
	$(CLICKKERNEL_CFLAGS) $(PACKAGE_CFLAGS) \
	$(DEFS) $(INCLUDES)

packagesrcdir ?= $(srcdir)
PACKAGE_OBJS ?= kpackage.ko
PACKAGE_DEPS ?=

KBUILD_EXTRA_SYMBOLS ?= $(clickbuild_libdir)/click.symvers

ifeq ($(PREPROCESS),1)
compile_option = -E
else
compile_option = -c
endif

cmd_shortensyms = $(CLICK_BUILDTOOL) $(CLICK_BUILDTOOL_FLAGS) shortensyms $@

quiet_cmd_cxxcompile = CXX $(quiet_modtag) $(subst $(obj)/,,$@)
cmd_cxxcompile = $(CXXCOMPILE) $(DEPCFLAGS) $(compile_option) -o $@ $< && $(cmd_shortensyms)

quiet_cmd_cxxcompile_nodep = CXX $(quiet_modtag) $(subst $(obj)/,,$@)
cmd_cxxcompile_nodep = $(CXXCOMPILE) $(compile_option) -o $@ $< && $(cmd_shortensyms)

quiet_cmd_ccompile = CC $(quiet_modtag) $(subst $(obj)/,,$@)
cmd_ccompile = $(COMPILE) $(DEPCFLAGS) $(compile_option) -o $@ $<

quiet_cmd_ccompile_nodep = CC $(quiet_modtag) $(subst $(obj)/,,$@)
cmd_ccompile_nodep = $(COMPILE) $(compile_option) -o $@ $<

EXTRA_CFLAGS += $(CLICKCPPFLAGS) $(CLICKKERNEL_CFLAGS) $(CLICKDEFS) $(CLICKINCLUDES)

ifneq ($(KBUILD_EXTMOD),)
ifeq ($(srcdir),.)
top_srcdir := $(src)/..
srcdir := $(src)
else
ifneq ($(patsubst /%,/,$(srcdir)),/)
top_srcdir := $(obj)/$(top_srcdir)
srcdir := $(obj)/$(srcdir)
endif
top_builddir := $(obj)/$(top_builddir)
builddir := $(obj)
endif

-include $(obj)/kelements.mk

$(package)-objs := $(ELEMENT_OBJS) $(PACKAGE_OBJS) kversion.ko
endif

obj-m += $(package).o

$(obj)/kelements.conf: $(CLICK_BUILDTOOL)
	echo $(packagesrcdir) | $(CLICK_BUILDTOOL) $(CLICK_BUILDTOOL_FLAGS) findelem -r linuxmodule -r $(package) -P $(CLICKFINDELEMFLAGS) > $(obj)/kelements.conf
$(obj)/kelements.mk: $(obj)/kelements.conf $(CLICK_BUILDTOOL)
	$(CLICK_BUILDTOOL) $(CLICK_BUILDTOOL_FLAGS) elem2make -t linuxmodule < $(obj)/kelements.conf > $(obj)/kelements.mk
$(obj)/kpackage.ko: $(obj)/kpackage.cc
	$(call if_changed_dep,cxxcompile)
$(obj)/kpackage.cc: $(obj)/kelements.conf $(CLICK_BUILDTOOL)
	$(CLICK_ELEM2PACKAGE) $(package) < $(obj)/kelements.conf > $(obj)/kpackage.cc
	@rm -f $(obj)/kpackage.kd
$(obj)/kversion.ko: $(obj)/kversion.c
	$(call if_changed_dep,ccompile)
$(obj)/kversion.c: $(CLICK_BUILDTOOL)
	$(CLICK_BUILDTOOL) $(CLICK_BUILDTOOL_FLAGS) kversion $(KVERSIONFLAGS) > $(obj)/kversion.c

DEPFILES := $(wildcard *.kd)
ifneq ($(DEPFILES),)
include $(DEPFILES)
endif

.PHONY: clean elemlist
