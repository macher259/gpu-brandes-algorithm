CC:=nvcc
EXE:=brandes
OBJS:=main.o brandes.o graph.o stride.o
CFLAGS:=-O2 -gencode arch=compute_70,code=sm_70

%.o: %.cu
	$(CC) $(CFLAGS) $< -c -o $@

$(EXE): $(OBJS)
	$(CC) $^ -o $@

clean:
	rm -f $(OBJS) $(EXE)