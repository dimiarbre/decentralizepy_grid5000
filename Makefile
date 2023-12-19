all: compute_container.sif


%.sif: %.def
ifdef OAR_NODEFILE
	sudo-g5k /grid5000/spack/v1/opt/spack/linux-debian11-x86_64_v2/gcc-10.4.0/singularity-3.8.7-rv6m5rw2bda5vu5cb7kcw6jfjg24xp6h/bin/singularity -d build $@ $<
else
	singularity -d build --fakeroot $@ $<  
endif

clean:
	rm *.sif