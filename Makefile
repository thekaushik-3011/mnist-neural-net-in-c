
run:multiLayerNN_notquant.c
	gcc -O3 -march=native -funroll-loops -o train.o multiLayerNN_notquant.c -fopenmp -lopenblas -lm

test:nn_inference_3h.c
	gcc -o test.o nn_inference.c -pthread -fopenmp -lopenblas -lm
#gcc -o test.o inference_multiLayerNN_notquant.c -lm

data_download:
	bash data_download.sh
