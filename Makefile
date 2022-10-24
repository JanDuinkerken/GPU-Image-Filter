# Include directory
IDIR=utils

# Cross compiler and compiler flags 
CC=nvcc
CFLAGS=-I $(IDIR)

# Source code directory
SRCDIR=src

# Output images directory
OUTDIR=output

# Directory where the object files will be stored
ODIR=build/obj

# Directory where the executable will be stored
BDIR=build

# External libraries imported
LIBS=-lm -lz

# Dependencies that where programmed by us
_DEPS = utils.h spng/spng.h
DEPS = $(patsubst %,$(IDIR)/%,$(_DEPS))

# Object files needed for compiling the filter
_OBJ = main.o utils.o spng.o
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))

#Rules for compiling the dependencies
$(ODIR)/%.o: $(SRCDIR)/%.cu $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

$(ODIR)/%.o: $(IDIR)/%.cu $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

$(ODIR)/%.o: $(IDIR)/spng/%.c $(DEPS)
	gcc -c -o $@ $< $(CFLAGS) $(LIBS)

# Rule for compiling the final filter
filter: $(OBJ)
	$(CC) -o $(BDIR)/$@ $^ $(CFLAGS) $(LIBS)

# Clean the object files
.PHONY: clean

clean: 
	rm -f $(ODIR)/*.o *- core $(IDIR)/*-
	rm -rf $(IDIR)/spng/
	rm -rf $(OUTDIR)
	rm -rf $(BDIR)

setup:
	mkdir -p $(BDIR) 
	mkdir -p $(ODIR)
	mkdir -p $(OUTDIR)
	mkdir -p $(IDIR)/spng/
	wget -O $(IDIR)/spng/spng.c https://raw.githubusercontent.com/randy408/libspng/master/spng/spng.c 
	wget -O $(IDIR)/spng/spng.h https://raw.githubusercontent.com/randy408/libspng/master/spng/spng.h