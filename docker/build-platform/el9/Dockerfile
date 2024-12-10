FROM rockylinux:9

# Install dependencies
RUN dnf -y update; dnf clean all; \
    dnf config-manager --set-enabled crb; \
    dnf -y install git make gcc gcc-c++ gcc-gfortran lapack-devel; \
    dnf -y install wget unzip patch which file cmake; \
    dnf -y install python3 python-devel findutils diffutils boost boost-devel;

# Fix the symlinks: https://bugs.rockylinux.org/view.php?id=7855
RUN cd /usr/lib64; \
    ln -sf libblas.so.3.9.0 libblas.so.3; \
    ln -sf liblapack.so.3.9.0 liblapack.so.3;

# Create directory structure
RUN mkdir repo

# Install PETSc
# 3.20.6 is the newest version that works with the PETSc extras
# included in this repo
RUN cd repo; \
    wget https://gitlab.com/petsc/petsc/-/archive/v3.20.6/petsc-v3.20.6.tar.gz; \
    tar -zxvf petsc-v3.20.6.tar.gz; \
    mv petsc-v3.20.6 petsc; \
    mkdir petsc-dist; \
    cd petsc; \
    ./configure configure --with-debug=0 --with-shared=0 --with-mpi=0 --with-fortran-bindings=0 --download-metis --download-mumps --with-mumps-serial=1 --prefix=/repo/petsc-dist; \
    make;\
    make install;

# Install boost
RUN cd repo; \
    wget https://boostorg.jfrog.io/artifactory/main/release/1.86.0/source/boost_1_86_0.tar.gz; \
    tar -zxvf boost_1_86_0.tar.gz; \
    cd ./boost_1_86_0; \
    sh bootstrap.sh --prefix=/usr/local/; \
    chmod a+x ./b2; \
    ./b2 install || true;


