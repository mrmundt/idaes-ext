ALL: petsc

CFLAGS = -I${ASL_INC}

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

bindir = $(prefix)/bin

# Statically link the MinGW runtime so petsc.exe has no external
# dependency on libwinpthread-1.dll / libgfortran-5.dll / etc.
STATIC_RUNTIME = -static -static-libgcc -static-libstdc++

petsc: petsc.o fg_nl.o fg_dae.o printing.o
	${CLINKER} ${STATIC_RUNTIME} -o petsc petsc.o fg_nl.o fg_dae.o printing.o $(ASL_LIB) ${PETSC_LIB}

py: petscpy
	${PYTHON} get_petsc_py.py

install: petsc
	${CP} petsc ${bindir}
