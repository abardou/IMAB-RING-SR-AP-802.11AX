SHELL := /bin/bash
VERSION := 3.31

.PHONY: help download extract prepare build test

.DEFAULT: help

SUPPORTED_COMMANDS := pytest
SUPPORTS_MAKE_ARGS := $(findstring $(firstword $(MAKECMDGOALS)), $(SUPPORTED_COMMANDS))
ifneq "$(SUPPORTS_MAKE_ARGS)" ""
  COMMAND_ARGS := $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))
  $(eval $(COMMAND_ARGS):;@:)
endif

all: download extract prepare build test

pyall: download extract pyprepare pybuild pytest

help:
	@echo "make download"
	@echo "       Download the NS-3 tarball"
	@echo "make extract"
	@echo "       Extract the NS-3 tarball"
	@echo "make prepare"
	@echo "       Copy custom models to NS-3 directory"
	@echo "make build"
	@echo "       Build NS-3"
	@echo "make test"
	@echo "       Check that NS-3 is working"
	@echo "make simul"
	@echo "       Launch the NS-3 simulation"

download:
	@echo "Downloading NS-3 version $(VERSION)"
	wget --continue "https://www.nsnam.org/releases/ns-allinone-$(VERSION).tar.bz2" 

download-dev:
	@echo "Downloading NS-3 dev tree"
	wget --continue "https://gitlab.com/nsnam/ns-3-dev/-/archive/master/ns-3-dev-master.tar.gz"

extract: download
	@echo "Extracting NS-3 version $(VERSION)"
	tar -xf ns-allinone-$(VERSION).tar.bz2

extract-dev: download-dev
	@echo "Extracting NS-3 dev tree"
	tar -xf ns-3-dev-master.tar.gz

pyprepare: extract
	@echo "Patching NS-3 with Python scripts…"
	cp -r ./pyscratch ./ns-allinone-$(VERSION)/ns-$(VERSION)/

prepare:
	@echo "Patching NS-3…"
	cp -r ./scratch ./ns-allinone-$(VERSION)/ns-$(VERSION)/
	
prepare-dev: extract-dev
	@echo "Patching NS-3…"
	cp -r ./scratch ./ns-3-dev-master/

pybuild: pyprepare
	cd ns-allinone-$(VERSION)/ns-$(VERSION) && ./waf configure --enable-examples --enable-tests
	cd ns-allinone-$(VERSION)/ns-$(VERSION) && ./waf build

build: prepare
	cd ns-allinone-$(VERSION)/ns-$(VERSION) && ./waf configure --enable-examples --enable-tests
	cd ns-allinone-$(VERSION)/ns-$(VERSION) && ./waf build
	cp -r ./scratch/nsTest/topos ns-allinone-$(VERSION)/ns-$(VERSION)/build/scratch/nsTest

build-dev: prepare-dev
	cd ns-3-dev-master/ && ./waf configure --enable-examples --enable-tests
	cd ns-3-dev-master/ && ./waf build

pytest: pybuild
	cd ns-allinone-$(VERSION)/ns-$(VERSION) && ./waf --pyrun pyscratch/main.py

test: build
	cd ns-allinone-$(VERSION)/ns-$(VERSION) && echo "./build/scratch/nsTest/nsTest" | ./waf shell

test-dev: build-dev
	cd ns-3-dev-master && echo "./build/scratch/example" | ./waf shell