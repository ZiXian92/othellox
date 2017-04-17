CC=/usr/lib64/openmpi/bin/mpicc
OPTS=-lm

othellox: othellox.c
	$(CC) $(OPTS) -o $@ $<

othellox-serial: othellox-serial.c
	gcc -lm -o $@ $<

bcasttest: bcasttest.c
	$(CC) $(OPTS) -o $@ $<
