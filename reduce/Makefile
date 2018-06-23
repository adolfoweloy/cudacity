CC=nvcc
OUTPUT=main
OUTTEST=test
#CFLAGS=-Xptxas --opt-level=3 -arch sm_30
CFLAGS=-g -G

# run the binary
run: compile
	./$(OUTPUT) input.dat

# compiling main file
compile: clean reduce.cu main.cu
	$(CC) $(CFLAGS) -o main reduce.cu main.cu

test: compile_test
	./$(OUTTEST) input.dat

compile_test: clean reduce.cu reduce_seq.cu test.cu
	$(CC) $(CFLAGS) -o test reduce.cu reduce_seq.cu test.cu

# removes all object and binary files
clean:
	find ./ -iname "*.o" | xargs rm -f
	rm -f $(OUTPUT)
