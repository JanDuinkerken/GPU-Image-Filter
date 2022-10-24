# Include directory
IDIR=utils

# Cross compiler and compiler flags 
CC=nvcc
CFLAGS=-I $(IDIR)

# Source code directory
SRCDIR=src

# Directory where the object files will be stored
ODIR=build/obj

# Directory where the executable will be stored
BDIR=build

# External libraries imported
LIBS=-lpng

# Dependencies that where programmed by us
_DEPS = utils.h
DEPS = $(patsubst %,$(IDIR)/%,$(_DEPS))

# Object files needed for compiling the filter
_OBJ = main.o utils.o
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))

#Rules for compiling the dependencies
$(ODIR)/%.o: $(SRCDIR)/%.cu $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

$(ODIR)/%.o: $(IDIR)/%.cu $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

# Rule for compiling the final filter
filter: $(OBJ)
	$(CC) -o $(BDIR)/$@ $^ $(CFLAGS) $(LIBS)

# Clean the object files
.PHONY: clean

clean: 
	rm -f $(ODIR)/*.o *- core $(IDIR)/*-