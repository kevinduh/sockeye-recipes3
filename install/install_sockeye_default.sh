#!/bin/bash
set -e

SOCKEYE_COMMIT=94cdad722cd9d0b2d24ddd401ec170db1280b83b # 3.1.14 (sockeye:main)

# Get this version of sockeye
rootdir="$(readlink -f "$(dirname "$0")/../")"
cd $rootdir
git submodule init
git submodule update --recursive --remote sockeye
cd sockeye
git checkout $SOCKEYE_COMMIT
cd ..

$rootdir/install/install_sockeye_custom.sh -s $rootdir/sockeye -e sockeye3
