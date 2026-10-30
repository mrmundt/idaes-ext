#!/bin/sh

# These scripts are not meant to be used for general builds.  They
# are tailored only to specific build systems.  They may provide some
# hints on how to build the solvers, but are limited.
#
# Usage: compile_solvers.sh PLATFORM [--without-hsl] [--restart]
#
# RESUMING: if a run dies part-way (network drop, Ctrl-C, compile error),
# just re-run the SAME command. Every expensive step leaves a marker in
# ./.build_stamps when it finishes, and finished steps are skipped on the
# next run. Pass --restart (or delete ./coinbrew) to start from scratch.

# Set to exit on error
set -e

usage() {
  echo "Usage: $0 PLATFORM [--without-hsl] [--restart]"
  echo "  PLATFORM       one of {windows, darwin, el8, el9,"
  echo "                 ubuntu2004, ubuntu2204, ubuntu2404}"
  echo "  --without-hsl  build solvers without HSL"
  echo "  --restart      discard any previous partial build and start over"
}

# First argument is OS name provided by user when running this
osname=$1;
if [ -z $osname ]
then
  echo "Must specify platform in {windows, darwin, el8, el9, "
  echo "  ubuntu2004, ubuntu2204, ubuntu2404}."
  exit 1
fi
shift

# Remaining arguments are optional flags, in any order
without_hsl_arg="NO"
restart="NO"
for arg in "$@"; do
  case "$arg" in
    --without-hsl) without_hsl_arg="YES" ;;
    --restart)     restart="YES" ;;
    *)
      echo "Unknown argument: $arg" >&2
      usage >&2
      exit 1
      ;;
  esac
done

# Get the machine type
export MNAME=`uname -m`

# Get path of directory we're working in
# Note that, when accessing patch files, we assume this directory
# to be the root of the idaes-ext repository.
export IDAES_EXT=`pwd`

if [ "$without_hsl_arg" = "YES" ]; then
  echo "--without-hsl flag received. Building solvers without HSL." >&2
  echo "HSL: NO"
  hslflag="--without-hsl"
  with_hsl="NO"
  build_hsl="NO"
elif [ -f $IDAES_EXT/../coinhsl.tar.gz ]; then
  echo "coinhsl.tar.gz found. Building solvers with HSL." >&2
  echo "HSL: YES"
  hslflag="--with-hsl"
  with_hsl="YES"
  build_hsl="YES"
else
  echo "coinhsl.tar.gz not found. Attempting to build with installed HSL." >&2
  echo "To skip HSL-reliant build steps, send the --without-hsl argument to this script" >&2
  echo "HSL: YES"
  hslflag="--with-hsl"
  with_hsl="YES"
  build_hsl="NO"
fi

# Set a few basic things
export IPOPT_BRANCH="stable/3.14"
export IPOPT_REPO="https://github.com/IDAES/Ipopt"
export PYNU_BRANCH="main"
export PYNU_REPO="https://github.com/pyomo/pyomo"
export K_AUG_BRANCH="default"
export K_AUG_REPO="https://github.com/idaes/k_aug"
export CC="gcc"
export CXX="g++"

# set PETSc location
# These petsc environment variables are used by the makefile
# in the $IDAES_EXT/petsc subdirectory.
if [ -z $PETSC_DIR ]; then
    # We only set PETSC_DIR if it is not already set. This is useful when
    # running this script locally (i.e. not via Docker)
    if [ ${osname} = "windows" ]
    then
      export PETSC_DIR=/c/repo/petsc-dist
    elif [ ${osname} = "darwin" ]; then
      export PETSC_DIR="$HOME/src/petsc-dist"
    else
      export PETSC_DIR=/repo/petsc-dist
    fi
    echo "PETSC_DIR has been set to $PETSC_DIR. If this is incorrect, set"
    echo "the PETSC_DIR environment variable before running this script."
fi
export PETSC_ARCH=""

# The newer MinGW-w64 toolchain links binaries against symbols (e.g.
# nanosleep64) that only exist in its matching libwinpthread-1.dll. If an
# older libwinpthread-1.dll is resolved first at runtime, the binary fails to
# launch with an "entry point not found" error (and, for the solver exes,
# shows up downstream as an empty version string / IndexError in Pyomo's ASL
# driver). Incorporating the GCC/MinGW runtime into each artifact removes that
# external dependency entirely.
#
# All are empty on non-Windows platforms, so the configure/cmake lines below
# are unchanged there (static glibc + dynamic BLAS/LAPACK would break Linux,
# and -static is unsupported on macOS).
if [ "$osname" = "windows" ]; then
  # COIN solvers: gcc-valid runtime flags in LDFLAGS ...
  STATIC_EXE_LDFLAGS="-static-libgcc -static-libstdc++"
  # ... and the libtool-only fully-static flag in LT_LDFLAGS.
  STATIC_EXE_LTFLAGS="-all-static"
  # cmake/raw-gcc executables (k_aug): gcc's own -static.
  STATIC_CMAKE_EXE_LDFLAGS="-static -static-libgcc -static-libstdc++"
  # Libraries that must be appended at the end of a static link line
  STATIC_CMAKE_END_LIBS="-lquadmath"
  # Shared objects (libpynumero_ASL, etc.): fold in GCC runtime only.
  STATIC_LIB_LDFLAGS="-static-libgcc -static-libstdc++ -Wl,-Bstatic -lwinpthread -lgfortran -lquadmath -Wl,-Bdynamic"
else
  STATIC_EXE_LDFLAGS=""
  STATIC_EXE_LTFLAGS=""
  STATIC_CMAKE_EXE_LDFLAGS=""
  STATIC_CMAKE_END_LIBS=""
  STATIC_LIB_LDFLAGS=""
fi
# ----------------------------------------------------------------------------

if [ "$osname" = "darwin" ]; then
  if [ -n "$HOMEBREW_PREFIX" ]; then
    # Find a GCC formula: prefer unversioned gcc, then fall back to gcc@*
    for formula in gcc gcc@15 gcc@14 gcc@13; do
      GCC_PREFIX=$($HOMEBREW_PREFIX/bin/brew --prefix "$formula" 2>/dev/null || true)
      if [ -n "$GCC_PREFIX" ] && [ -d "$GCC_PREFIX/bin" ]; then
        GCC_BIN="$GCC_PREFIX/bin"
        break
      fi
    done

    if [ -n "$GCC_BIN" ]; then
      GCC_EXE=$(ls "$GCC_BIN"/gcc-[0-9]* 2>/dev/null \
                | grep -E 'gcc-[0-9]+$' \
                | sort -V \
                | tail -1)
      if [ -n "$GCC_EXE" ]; then
        GCC_VER="${GCC_EXE##*-}"

        export CC="$GCC_BIN/gcc-$GCC_VER"
        export CXX="$GCC_BIN/g++-$GCC_VER"
        export F77="$GCC_BIN/gfortran-$GCC_VER"
        export FC="$GCC_BIN/gfortran-$GCC_VER"

        echo "Using GCC toolchain:"
        echo "  CC = $CC"
        echo "  CXX = $CXX"
        echo "  F77 = $F77"
        echo "  FC = $FC"
      fi

      # BREWLIB: find matching libgfortran dir
      gcc_libroot="$HOMEBREW_PREFIX/opt/$(basename "$GCC_PREFIX")/lib/gcc"
      if [ -d "$gcc_libroot" ]; then
        gcc_libver=$(ls "$gcc_libroot" | sort -V | tail -1)
        export BREWLIB="$gcc_libroot/$gcc_libver/"
      fi
    fi
  fi
fi

echo "#########################################################################"
echo "# Resume support                                                        #"
echo "#########################################################################"
# Each expensive step drops a marker file in $STAMP_DIR when it succeeds.
# On a re-run, steps with a marker are skipped, so an interrupted build picks
# up where it left off. Packaging into ./dist is cheap and is always redone
# from scratch, so the final tarball never contains stale files.
STAMP_DIR="$IDAES_EXT/.build_stamps"
build_config="osname=$osname machine=$MNAME hslflag=$hslflag build_hsl=$build_hsl"

if [ "$restart" = "YES" ]; then
  echo "--restart given: removing coinbrew/, dist/ and saved build progress."
  rm -rf "$STAMP_DIR" "$IDAES_EXT/coinbrew" "$IDAES_EXT/dist"
fi

if [ ! -d "$IDAES_EXT/coinbrew" ]; then
  # Nothing to resume from; make sure no stale markers survive.
  rm -rf "$STAMP_DIR"
  echo "Starting a fresh build."
elif [ -f "$STAMP_DIR/config" ]; then
  prev_config=$(cat "$STAMP_DIR/config")
  if [ "$prev_config" != "$build_config" ]; then
    echo "The existing build in $IDAES_EXT/coinbrew used different settings:" >&2
    echo "  previous run: $prev_config" >&2
    echo "  this run:     $build_config" >&2
    echo "Re-run with the same arguments as before to resume it, or add" >&2
    echo "--restart to discard it and start over with the new settings." >&2
    exit 1
  fi
  echo "Resuming previous build; completed steps will be skipped."
else
  echo "Found $IDAES_EXT/coinbrew with no saved progress (probably from an"
  echo "older version of this script). Reusing it and re-running every step;"
  echo "if anything misbehaves, re-run with --restart."
fi
mkdir -p "$STAMP_DIR"
echo "$build_config" > "$STAMP_DIR/config"

# run_stage NAME COMMAND [ARGS...]
# Runs COMMAND unless step NAME already finished on an earlier run, then
# records it as finished. COMMAND runs in a subshell, so a step that fails
# half-way can't leave the rest of the script in the wrong directory.
run_stage() {
  stage_name=$1
  shift
  if [ -f "$STAMP_DIR/$stage_name.done" ]; then
    echo ">>> Step '$stage_name' already completed on a previous run; skipping."
    return 0
  fi
  ( set -e; "$@" )
  touch "$STAMP_DIR/$stage_name.done"
}

# get_repo URL DIR
# Clone URL into DIR, or clean DIR if it is already a working clone. A clone
# that was cut off part-way (e.g. by a network drop) is deleted and redone.
get_repo() {
  _url=$1
  _dir=$2
  # GIT_CEILING_DIRECTORIES stops git from walking up into the idaes-ext
  # repo itself when $_dir is not a usable repository.
  if [ -d "$_dir" ] && ! { [ -e "$_dir/.git" ] && \
       GIT_CEILING_DIRECTORIES="$IDAES_EXT" \
       git -C "$_dir" rev-parse --verify -q HEAD >/dev/null 2>&1; }; then
    echo "$_dir looks like an incomplete clone; removing it and cloning again."
    rm -rf "$_dir"
  fi
  if [ ! -d "$_dir" ]; then
    git clone "$_url" "$_dir"
    echo "Repository cloned from $_url."
  else
    git -C "$_dir" clean -fd
    echo "Untracked files removed from $_dir."
  fi
}

mkdir -p coinbrew
cd coinbrew

# Download to a temporary name first so an interrupted download never leaves
# a truncated coinbrew script behind.
if [ ! -s coinbrew ]; then
  rm -f coinbrew.part
  if [ ${osname} = "darwin" ]; then
    curl --fail --output coinbrew.part https://raw.githubusercontent.com/coin-or/coinbrew/master/coinbrew
  else
    wget --secure-protocol tlsv1 -O coinbrew.part https://raw.githubusercontent.com/coin-or/coinbrew/master/coinbrew
  fi
  mv coinbrew.part coinbrew
fi

# Work-around for mumps gcc v10 gfortran bug
GFORT_VERSION=`gfortran -dumpversion`
GFMV=(${GFORT_VERSION//./ })
if [ ${GFMV[0]} -ge 10 ]; then
  export FCFLAGS="-w -fallow-argument-mismatch -O2"
  export FFLAGS="-w -fallow-argument-mismatch -O2"
fi

# Fetch coin-or stuff and dependencies
SKIP_PKGS='ThirdParty/Lapack ThirdParty/Blas ThirdParty/Glpk ThirdParty/Mumps'
run_stage fetch_clp bash coinbrew fetch Clp --no-prompt --skip "$SKIP_PKGS"
run_stage fetch_cbc bash coinbrew fetch Cbc --no-prompt --skip "$SKIP_PKGS"
SKIP_PKGS="$SKIP_PKGS Cbc Clp Cgl Osi"
run_stage fetch_bonmin bash coinbrew fetch Bonmin --no-prompt --skip "$SKIP_PKGS"
run_stage fetch_couenne bash coinbrew fetch Couenne --no-prompt --skip "$SKIP_PKGS"
# Couenne <= 0.5.8 needs const comparators; fixed upstream in 0.5.9
# (Safe to repeat: each patch is applied only if it isn't already there.)
cd Couenne
if ! grep -q 'const Node &b) const' Couenne/src/problem/CouenneProblem.hpp; then
  patch Couenne/src/problem/CouenneProblem.hpp < $IDAES_EXT/scripts/CouenneProblem.hpp.patch
fi
if ! grep -q 'const &b) const' Couenne/src/cut/sdpcuts/CouenneMatrix.hpp; then
  patch Couenne/src/cut/sdpcuts/CouenneMatrix.hpp < $IDAES_EXT/scripts/CouenneMatrix.hpp.patch
fi
cd ..

stage_fetch_ipopt() {
  # We don't want to coin-or Ipopt (or a half-made copy from an earlier run)
  rm -rf Ipopt Ipopt_share
  # We DO want the custom IDAES Ipopt
  bash coinbrew fetch $IPOPT_REPO@$IPOPT_BRANCH --no-prompt --skip 'ThirdParty/Lapack ThirdParty/Blas ThirdParty/Glpk'
  cp -r Ipopt Ipopt_share
}
run_stage fetch_ipopt stage_fetch_ipopt

# Make sure I don't include any dependencies I don't want
rm -rf ThirdParty/Blas
rm -rf ThirdParty/Lapack
rm -rf ThirdParty/FilterSQP
rm -rf ThirdParty/SCIP
rm -rf ThirdParty/SoPlex
rm -rf ThirdParty/Glpk

echo "#########################################################################"
echo "# Use novariadic metis 4.0.3 if available                               #"
echo "#########################################################################"
if [ -f $IDAES_EXT/../metis-4.0-novariadic.tar.gz ]; then
  stage_metis_novariadic() {
    echo "Found metis-4.0-novariadic.tar.gz"
    rm -rf ./ThirdParty/Metis/metis-4.0/*
    cp $IDAES_EXT/../metis-4.0-novariadic.tar.gz ./ThirdParty/Metis/metis-4.0/
    cd ThirdParty/Metis/metis-4.0
    tar -zxvf metis-4.0-novariadic.tar.gz
    rm metis-4.0-novariadic.tar.gz
    make $PARALLEL
  }
  run_stage metis_novariadic stage_metis_novariadic
fi

######
# Set PKG_CONFIG_PATH so configure scripts can find the stuff we build
#######
export PKG_CONFIG_PATH=${PKG_CONFIG_PATH}:${IDAES_EXT}/coinbrew/dist/lib/pkgconfig

echo "#########################################################################"
echo "# Get coinhsl.tar.gz if available                                       #"
echo "#########################################################################"
# If we have the HSL stuff copy and extract it in the right place
if [ -f $IDAES_EXT/../coinhsl.tar.gz ]; then
  stage_extract_hsl() {
    echo "Found coinhsl.tar.gz"
    # if the HSL source tarball is in place...
    mkdir -p ThirdParty/HSL/
    cp $IDAES_EXT/../coinhsl.tar.gz ThirdParty/HSL/
    cd ThirdParty/HSL/
    tar -xvf coinhsl.tar.gz
  }
  run_stage extract_hsl stage_extract_hsl
fi

echo "#########################################################################"
echo "# Thirdparty/ASL                                                        #"
echo "#########################################################################"
stage_asl() {
  cd ThirdParty/ASL
  ./configure --disable-shared --enable-static --prefix=$IDAES_EXT/coinbrew/dist
  make $PARALLEL
  make install
}
run_stage build_asl stage_asl

echo "#########################################################################"
echo "# Thirdparty/Metis                                                      #"
echo "#########################################################################"
stage_metis() {
  cd ThirdParty/Metis
  if [ "$MNAME" = "aarch64" ]; then
    # MNAME of darwin is arm64, so this is linux only
    ./configure --disable-shared --enable-static --prefix=$IDAES_EXT/coinbrew/dist \
      --build=aarch64-unknown-linux-gnu FFLAGS="-fPIC" CFLAGS="-fPIC" CXXFLAGS="-fPIC"
  else
    ./configure --disable-shared --enable-static --prefix=$IDAES_EXT/coinbrew/dist \
      FFLAGS="-fPIC" CFLAGS="-fPIC -Wno-error=implicit-function-declaration" CXXFLAGS="-fPIC"
  fi
  make $PARALLEL
  make install
}
run_stage build_metis stage_metis

echo "#########################################################################"
echo "# Thirdparty/HSL                                                        #"
echo "#########################################################################"
if [ "$build_hsl" = "YES" ]; then
  stage_hsl() {
    cd ThirdParty/HSL

    if [ "$MNAME" != "arm64" ]; then
      METIS_CFLAGS="-I$IDAES_EXT/coinbrew/dist/include/coin/ThirdParty"
      METIS_LFLAGS="-L$IDAES_EXT/coinbrew/dist/lib -lcoinmetis -lm"

      ./configure \
        --disable-shared --enable-static --with-metis \
        --with-metis-cflags="$METIS_CFLAGS" \
        --with-metis-lflags="$METIS_LFLAGS" \
        --prefix=$IDAES_EXT/coinbrew/dist \
        FFLAGS="-fPIC" CFLAGS="-fPIC" CXXFLAGS="-fPIC"
    else
      # Metis + arm64 do not play well together. We'd rather
      # have something HSL functional than nothing at all,
      # so at least for now, we skip metis.
      ./configure \
        --disable-shared --enable-static --without-metis \
        --prefix=$IDAES_EXT/coinbrew/dist \
        FFLAGS="-fPIC" CFLAGS="-fPIC" CXXFLAGS="-fPIC"
    fi

    make $PARALLEL
    make install
  }
  run_stage build_hsl stage_hsl
fi

echo "#########################################################################"
echo "# Thirdparty/Mumps                                                      #"
echo "#########################################################################"
stage_mumps() {
  cd ThirdParty/Mumps
  # We compile without metis to get around a suspected Metis/Mumps version
  # incompatibility. See https://github.com/IDAES/idaes-ext/issues/268
  # ThirdParty/Metis uses v4.0.3, while ThirdParty/Mumps uses the latest release.
  METISFLAG="--without-metis"
  ./configure --disable-shared --enable-static $METISFLAG \
   --prefix=$IDAES_EXT/coinbrew/dist FFLAGS="-fPIC" CFLAGS="-fPIC" CXXFLAGS="-fPIC"
  make $PARALLEL
  make install
}
run_stage build_mumps stage_mumps

echo "#########################################################################"
echo "# Ipopt ampl executables                                                #"
echo "#########################################################################"
stage_ipopt() {
  cd Ipopt
  ./configure --disable-shared --enable-static --with-mumps $hslflag \
    --prefix=$IDAES_EXT/coinbrew/dist \
    LDFLAGS="$STATIC_EXE_LDFLAGS" LT_LDFLAGS="$STATIC_EXE_LTFLAGS"
  make $PARALLEL
  make install
}
run_stage build_ipopt stage_ipopt

echo "#########################################################################"
echo "# CoinUtils                                                             #"
echo "#########################################################################"
stage_coinutils() {
  cd CoinUtils
  if [ "$MNAME" = "aarch64" ]; then
    # MNAME of darwin is arm64, so this is linux only
    ./configure --build=aarch64-unknown-linux-gnu --disable-shared --enable-static \
      --prefix=$IDAES_EXT/coinbrew/dist
  else
    ./configure --disable-shared --enable-static --prefix=$IDAES_EXT/coinbrew/dist
  fi
  make $PARALLEL
  make install
}
run_stage build_coinutils stage_coinutils

echo "#########################################################################"
echo "# Osi                                                                   #"
echo "#########################################################################"
stage_osi() {
  cd Osi
  if [ "$MNAME" = "aarch64" ]; then
    # MNAME of darwin is arm64, so this is linux only
    ./configure --build=aarch64-unknown-linux-gnu --disable-shared --enable-static \
      --prefix=$IDAES_EXT/coinbrew/dist
  else
    ./configure --disable-shared --enable-static --prefix=$IDAES_EXT/coinbrew/dist
  fi
  make $PARALLEL
  make install
}
run_stage build_osi stage_osi

echo "#########################################################################"
echo "# Clp                                                                   #"
echo "#########################################################################"
stage_clp() {
  cd Clp
  if [ "$MNAME" = "aarch64" ]; then
    # MNAME of darwin is arm64, so this is linux only
    ./configure --build=aarch64-unknown-linux-gnu --disable-shared --enable-static \
     --prefix=$IDAES_EXT/coinbrew/dist LDFLAGS="$STATIC_EXE_LDFLAGS" LT_LDFLAGS="$STATIC_EXE_LTFLAGS"
  else
    ./configure --disable-shared --enable-static --prefix=$IDAES_EXT/coinbrew/dist \
     LDFLAGS="$STATIC_EXE_LDFLAGS" LT_LDFLAGS="$STATIC_EXE_LTFLAGS"
  fi
  make $PARALLEL
  make install
}
run_stage build_clp stage_clp

echo "#########################################################################"
echo "# Cgl                                                                   #"
echo "#########################################################################"
stage_cgl() {
  cd Cgl
  if [ "$MNAME" = "aarch64" ]; then
    # MNAME of darwin is arm64, so this is linux only
    ./configure --build=aarch64-unknown-linux-gnu --disable-shared --enable-static \
      --prefix=$IDAES_EXT/coinbrew/dist
  else
    ./configure --disable-shared --enable-static --prefix=$IDAES_EXT/coinbrew/dist
  fi
  make $PARALLEL
  make install
}
run_stage build_cgl stage_cgl

echo "#########################################################################"
echo "# Cbc                                                                   #"
echo "#########################################################################"
stage_cbc() {
  cd Cbc
  if [ "$MNAME" = "aarch64" ]; then
    # MNAME of darwin is arm64, so this is linux only
    ./configure --build=aarch64-unknown-linux-gnu --disable-shared --enable-static \
      --prefix=$IDAES_EXT/coinbrew/dist LDFLAGS="$STATIC_EXE_LDFLAGS" LT_LDFLAGS="$STATIC_EXE_LTFLAGS"
  else
    ./configure --disable-shared --enable-static --prefix=$IDAES_EXT/coinbrew/dist \
      LDFLAGS="$STATIC_EXE_LDFLAGS" LT_LDFLAGS="$STATIC_EXE_LTFLAGS"
  fi
  make $PARALLEL
  make install
}
run_stage build_cbc stage_cbc

echo "#########################################################################"
echo "# Bonmin                                                                #"
echo "#########################################################################"
stage_bonmin() {
  cd Bonmin
  # Two replacements below are a temporary fix until I can nail the problem down better
  # There is a problem linking the exception class TMINLP_INVALID so replaced
  # it with a similar one from IPOPT that works
  # (Safe to repeat: a second pass finds nothing left to replace.)
  sed s/"TMINLP_INVALID"/"INVALID_TNLP"/g Bonmin/src/Interfaces/BonTMINLP2TNLP.cpp > atmpfile
  mv atmpfile Bonmin/src/Interfaces/BonTMINLP2TNLP.cpp
  sed s/"TMINLP_INVALID"/"INVALID_TNLP"/g Bonmin/src/Interfaces/BonBranchingTQP.cpp > atmpfile
  mv atmpfile Bonmin/src/Interfaces/BonBranchingTQP.cpp
  if [ "$MNAME" = "aarch64" ]; then
    # MNAME of darwin is arm64, so this is linux only
    ./configure --build=aarch64-unknown-linux-gnu --disable-shared --enable-static \
      --prefix=$IDAES_EXT/coinbrew/dist LDFLAGS="-fopenmp $STATIC_EXE_LDFLAGS" LT_LDFLAGS="$STATIC_EXE_LTFLAGS"
  else
    ./configure --disable-shared --enable-static --prefix=$IDAES_EXT/coinbrew/dist \
      LDFLAGS="-fopenmp $STATIC_EXE_LDFLAGS" LT_LDFLAGS="$STATIC_EXE_LTFLAGS"
  fi
  make $PARALLEL
  make install
}
run_stage build_bonmin stage_bonmin

echo "#########################################################################"
echo "# Couenne                                                               #"
echo "#########################################################################"
stage_couenne() {
  cd Couenne
  if [ "$MNAME" = "aarch64" ]; then
    # MNAME of darwin is arm64, so this is linux only
    ./configure --build=aarch64-unknown-linux-gnu --disable-shared --enable-static \
      --prefix=$IDAES_EXT/coinbrew/dist LDFLAGS="-fopenmp $STATIC_EXE_LDFLAGS" LT_LDFLAGS="$STATIC_EXE_LTFLAGS"
  else
    ./configure --disable-shared --enable-static --prefix=$IDAES_EXT/coinbrew/dist \
      LDFLAGS="-fopenmp $STATIC_EXE_LDFLAGS" LT_LDFLAGS="$STATIC_EXE_LTFLAGS"
  fi
  make $PARALLEL
  make install
}
run_stage build_couenne stage_couenne

echo "#########################################################################"
echo "# Ipopt Shared Libraries                                                #"
echo "#########################################################################"
# NOTE: This build deliberately produces shared libipopt*.dll, so it must NOT
# be linked with -all-static. Left unchanged.
stage_ipopt_share() {
  cd Ipopt_share
  ./configure --enable-shared --disable-static --without-asl --disable-java \
    --with-mumps $hslflag --enable-relocatable --prefix=$IDAES_EXT/coinbrew/dist-share
  make $PARALLEL
  make install
}
run_stage build_ipopt_share stage_ipopt_share

# The above compiles and installs solvers into ./coinbrew/dist*
# Now we will copy the files we wish to distribute into a ./dist directory,
# with standard bin, include, and lib sub-directories.

echo "#########################################################################"
echo "# Copy Coin Solver Files to dist                                        #"
echo "#########################################################################"
cd $IDAES_EXT
# dist/ is always rebuilt from scratch (it's quick), so a re-run never
# packages leftovers from an earlier attempt.
rm -rf dist
mkdir dist dist/bin dist/include dist/lib
# I'll try to avoid cd-ing around, for now, to make this script more
# explicit for the reader.
# cd dist
# Executables
cp ./coinbrew/dist/bin/ipopt ./dist/bin/
cp ./coinbrew/dist/bin/ipopt_sens ./dist/bin/
cp ./coinbrew/dist/bin/clp ./dist/bin/
cp ./coinbrew/dist/bin/cbc ./dist/bin/
cp ./coinbrew/dist/bin/bonmin ./dist/bin/
cp ./coinbrew/dist/bin/couenne ./dist/bin/
# Run strip to remove unneeded symbols (it's okay that some files
#  aren't exe or libraries)
if [ $osname = "darwin" ]; then
  # I'm not sure if this exactly reproduces --strip-unneeded on darwin
  strip ./dist/bin/*
else
  strip --strip-unneeded ./dist/bin/*
fi

# Copy contents of dist-share (ipopt lib, include, and share) into dist
if [ "${osname}" = "windows" ]; then
  # includes, docs, pkgconfig
  cp -R ./coinbrew/dist-share/include ./dist/
  cp -R ./coinbrew/dist-share/share   ./dist/
  cp -R ./coinbrew/dist-share/lib     ./dist/

  # Move/copy Ipopt DLLs that Autotools put in dist-share/bin into dist/lib
  cp ./coinbrew/dist-share/bin/libipopt*.dll  ./dist/lib/
  cp ./coinbrew/dist-share/bin/libsipopt*.dll ./dist/lib/
else
  cp -R ./coinbrew/dist-share/* ./dist/
fi

# Handle platform-dependent sed behavior...
if [ $osname = "darwin" ]; then
  sed -i "" "s/^Requires\.private/#Requires.private/" dist/lib/pkgconfig/ipopt.pc
else
  sed -i "s/^Requires\.private/#Requires.private/" dist/lib/pkgconfig/ipopt.pc
fi

echo "#########################################################################"
echo "# Copy License and Version Files to dist-solvers                        #"
echo "#########################################################################"
# Text information files include build time
cp ./LICENSE.md ./dist/
cp ./VERSION.md ./dist/VERSION_SOLVERS.md
build_date="$(date +%Y%m%d-%H%M)"
build_plat="${osname}-${MNAME}"
sed "s/(DATE)/${build_date}/g; s/(PLAT)/${build_plat}/g" \
  "./dist/VERSION_SOLVERS.md" > "./dist/.VERSION_SOLVERS.md.tmp"
mv "./dist/.VERSION_SOLVERS.md.tmp" "./dist/VERSION_SOLVERS.md"

#
# Copy some linked libraries from homebrew or mingw, covered by gcc runtime
# exception https://www.gnu.org/licenses/gcc-exception-3.1.en.html license
# info is included in the license text file.
#
# NOTE: Keep copying these even after the static-linking changes above. The
# hardened executables no longer need them, but the in-process shared objects
# (libpynumero_ASL*, libipopt*.dll) may still rely on them until each has been
# verified clean. Remove only what an import-table check proves is unused.
#
echo "#########################################################################"
echo "# Copy GCC/MinGW Runtime Libraries to dist-solvers                      #"
echo "#########################################################################"
if [ ${osname} = "windows" ]; then
    # Windows MinGW linked redistributable libraries
    cp /mingw64/bin/libstdc++-6.dll ./dist/lib/
    cp /mingw64/bin/libgcc_s_seh-1.dll ./dist/lib/
    cp /mingw64/bin/libwinpthread-1.dll ./dist/lib/
    cp /mingw64/bin/libgfortran-*.dll ./dist/lib/
    cp /mingw64/bin/libquadmath-0.dll ./dist/lib/
    cp /mingw64/bin/libgomp-1.dll ./dist/lib/
    cp /mingw64/bin/liblapack.dll ./dist/lib/
    cp /mingw64/bin/libblas.dll ./dist/lib/
    cp /mingw64/bin/libbz2-*.dll ./dist/lib/
    cp /mingw64/bin/zlib*.dll ./dist/lib/
    cp /mingw64/bin/libssp*.dll ./dist/lib/ 2>/dev/null || true
fi

if [ ${osname} = "darwin" ]; then
  cp ${BREWLIB}libgfortran.5.dylib ./dist/lib/
  cp ${BREWLIB}libgcc_s.1.1.dylib ./dist/lib/
  cp ${BREWLIB}libstdc++.6.dylib ./dist/lib/
  cp ${BREWLIB}libgomp.1.dylib ./dist/lib/
  cp ${BREWLIB}libquadmath.0.dylib ./dist/lib/
fi

echo "#########################################################################"
echo "# Pynumero                                                              #"
echo "#########################################################################"

if [ ${osname} = "darwin" ]; then
  export CC="cc"
  export CXX="c++"
fi

stage_pynumero() {
  cd $IDAES_EXT
  get_repo "$PYNU_REPO" pyomo
  cd pyomo
  git checkout $PYNU_BRANCH
  cd pyomo/contrib/pynumero/src
  mkdir -p build
  cd build
  if [ ${osname} = "windows" ]
  then
    # libpynumero_ASL is a shared object loaded into the Python process, so it
    # gets the shared-object runtime flags (can't be fully static).
    cmake -DENABLE_HSL=no -DIPOPT_DIR=$IDAES_EXT/coinbrew/dist \
      -DCMAKE_SHARED_LINKER_FLAGS="$STATIC_LIB_LDFLAGS" -G"MSYS Makefiles" ..
  else
    cmake .. -DENABLE_HSL=no -DIPOPT_DIR=$IDAES_EXT/coinbrew/dist
  fi
  make $PARALLEL
}
run_stage build_pynumero stage_pynumero
cp $IDAES_EXT/pyomo/pyomo/contrib/pynumero/src/build/libpynumero_ASL* $IDAES_EXT/dist/lib/
# Return to root directory
cd $IDAES_EXT

echo "#########################################################################"
echo "# k_aug, dot_sens                                                       #"
echo "#########################################################################"
if [ $with_hsl = "YES" ]; then
  stage_k_aug() {
    cd $IDAES_EXT
    get_repo "$K_AUG_REPO" k_aug
    cp ./scripts/k_aug_CMakeLists.txt ./k_aug/CMakeLists.txt
    cd k_aug
    git checkout $K_AUG_BRANCH
    if [ ${osname} = "windows" ]
    then
      # k_aug and dot_sens are executables run as subprocesses.
      # cmake links via raw gcc (no libtool), so use gcc's own -static.
      # CMAKE_C_STANDARD_LIBRARIES appends -lquadmath at the end of the link so
      # the static libgfortran.a can resolve quadmath_snprintf
      cmake -DWITH_MINGW=ON -DCMAKE_C_COMPILER=$CC \
        -DCMAKE_EXE_LINKER_FLAGS="$STATIC_CMAKE_EXE_LDFLAGS" \
        -DCMAKE_C_STANDARD_LIBRARIES="$STATIC_CMAKE_END_LIBS" \
        -G"MSYS Makefiles" .
    else
      cmake -DCMAKE_C_COMPILER=$CC .
    fi
    make $PARALLEL
  }
  run_stage build_k_aug stage_k_aug
  cp $IDAES_EXT/k_aug/bin/k_aug* $IDAES_EXT/dist/bin/
  cp $IDAES_EXT/k_aug/dot_sens* $IDAES_EXT/dist/bin/
  # Return to root directory
  cd $IDAES_EXT
else
  echo "Skipping k_aug and dot_sens as we are not using HSL"
fi

echo "#########################################################################"
echo "# PETSc                                                                 #"
echo "#########################################################################"
export ASL_INC=$IDAES_EXT/coinbrew/dist/include/coin-or/asl
export ASL_LIB=$IDAES_EXT/coinbrew/dist/lib/libcoinasl.a
stage_petsc() {
  cd $IDAES_EXT/petsc
  if [ ${osname} = "windows" ]
  then
    # (Safe to repeat: already-patched lines no longer match these patterns.)
    FILES=("fg_dae.c" "fg_nl.c" "petsc.c" "printing.c")

    for f in "${FILES[@]}"; do
      if [ ! -f "$f" ]; then
        echo "Warning: $f not found, skipping"
        continue
      fi

      echo "Patching $f..."
      # Replace simple declarations: int err;  int err = 0;
      sed -i 's/\bint[[:space:]]\+err\b/fint err/g' "$f"
      sed -i 's/\bint[[:space:]]\+err\s*=/fint err=/g' "$f"

      # Fix comma declarations like:
      #      int i=0, err=0;
      #
      # The strategy:
      #   - detect lines with "int ... , err"
      #   - rewrite them into:
      #       int i=0;
      #       fint err=0;
      if grep -qE '\bint\b.*,' "$f" && grep -qE ',[[:space:]]*err' "$f"; then
        sed -i \
          -e 's/\bint\(.*\),[[:space:]]*err[[:space:]]*=\([^;]*\);/int\1;\nfint err=\2;/g' \
          -e 's/\bint\(.*\),[[:space:]]*err;/int\1;\nfint err;/g' \
          "$f"
        echo "  - Split comma declarations containing err"
      fi
    done

    echo "ASL err patch complete."
  fi
  # PETSc links through its own makefile (raw gcc CLINKER, not libtool), so it
  # uses the plain -static spelling via STATIC_RUNTIME. This was verified to
  # produce a self-contained petsc.exe (no libwinpthread/libgfortran imports).
  if [ ${osname} = "windows" ]; then
    make $PARALLEL STATIC_RUNTIME="-static -static-libgcc -static-libstdc++"
  else
    make $PARALLEL
  fi
  make py
}
run_stage build_petsc stage_petsc
cd $IDAES_EXT/petsc
if [ ${osname} = "windows" ]
then
  cp petsc.exe $IDAES_EXT/dist/bin/
else
  cp petsc $IDAES_EXT/dist/bin/
fi
cp -r petscpy $IDAES_EXT/dist/lib/

# Return to root directory
cd $IDAES_EXT

if [ ${osname} = "darwin" ]; then
  echo "#########################################################################"
  echo "# macOS update rpaths                                                   #"
  echo "#########################################################################"
  update_rpath_darwin() {
    install_name_tool -change ${BREWLIB}libgfortran.5.dylib @rpath/libgfortran.5.dylib $1
    install_name_tool -change ${BREWLIB}libgcc_s.1.1.dylib @rpath/libgcc_s.1.1.dylib $1
    install_name_tool -change ${BREWLIB}libstdc++.6.dylib @rpath/libstdc++.6.dylib $1
    install_name_tool -change ${BREWLIB}libgomp.1.dylib @rpath/libgomp.1.dylib $1
    install_name_tool -change ${BREWLIB}libquadmath.0.dylib @rpath/libquadmath.0.dylib $1
  }
  # Dynamic libraries also need to update their own install names
  update_library_rpath_darwin() {
    install_name_tool -id @rpath/$1 $1
  }
  update_rpath_darwin dist/bin/ipopt
  update_rpath_darwin dist/bin/ipopt_sens
  update_rpath_darwin dist/bin/clp
  update_rpath_darwin dist/bin/cbc
  update_rpath_darwin dist/bin/bonmin
  update_rpath_darwin dist/bin/couenne
  update_rpath_darwin dist/lib/libipopt.dylib
  update_rpath_darwin dist/lib/libsipopt.dylib
  update_rpath_darwin dist/lib/libipopt.3.dylib
  update_rpath_darwin dist/lib/libsipopt.3.dylib
  update_rpath_darwin dist/lib/libpynumero_ASL.dylib
  update_library_rpath_darwin dist/lib/libipopt.dylib
  update_library_rpath_darwin dist/lib/libsipopt.dylib
  update_library_rpath_darwin dist/lib/libipopt.3.dylib
  update_library_rpath_darwin dist/lib/libsipopt.3.dylib
  update_library_rpath_darwin dist/lib/libpynumero_ASL.dylib
  # if no hsl k_aug and dot_sens won't exist
  update_rpath_darwin dist/bin/k_aug || true
  update_rpath_darwin dist/bin/dot_sens || true
  update_rpath_darwin dist/bin/petsc
fi

# here you pack files
echo "#########################################################################"
echo "# Finish                                                                #"
echo "#########################################################################"
cd dist
if [ "$osname" = "el9" ]; then
  # Ensure our exes look in dist/lib first
  pushd bin
  for exe in *; do
    patchelf --set-rpath '$ORIGIN/../lib' "$exe" 2>/dev/null || true
  done
  popd

  # Add EL9-specific absolute symlinks
  # There is a bug on el9 machines regarding the blas/lapack links.
  # See: https://bugs.rockylinux.org/view.php?id=7855
  pushd lib
  ln -sfn /usr/lib64/libblas.so.3.9.0   libblas.so.3
  ln -sfn /usr/lib64/liblapack.so.3.9.0 liblapack.so.3
  popd
fi
tar -czvf idaes-solvers-${osname}-${MNAME}.tar.gz *
cd $IDAES_EXT
echo "Done"
echo "HSL Present: ${with_hsl}"
echo "(Build progress is saved in $STAMP_DIR; a re-run will only repackage."
echo " Use --restart to rebuild everything from scratch.)"

