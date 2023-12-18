all: compute_container.sif


%.sif: %.def
ifdef $(OAR_NODEFILE)
	sudo-g5k /grid5000/code/bin/singularity -d build $@ $<
else
	singularity -d build --fakeroot $@ $<  
endif

clean:
	rm *.sif