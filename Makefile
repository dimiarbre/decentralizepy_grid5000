all: compute_container.sif


%.sif: %.def
	# singularity -d build --fakeroot $@ $<  
	sudo-g5k /grid5000/code/bin/singularity -d build $@ $<  


clean:
	rm *.sif