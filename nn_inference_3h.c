#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <pthread.h>
#include <omp.h>

// Check for AVX support at compile time
#if defined(__AVX__) && defined(__AVX2__)
#include <immintrin.h>
#define USE_AVX 1
#else
#define USE_AVX 0
#endif

const int hidden_size = 256;
const int input_size = 784;
const int output_size = 10;
const int num_train_images = 60000;
const int num_test_images = 10000;
const int num_epochs = 10;
const int batch_size = 128;

// Cache-aligned structure for better memory performance
typedef struct __attribute__((aligned(64))) {
    float *training_images;
    int *training_labels;
    float *test_images;
    int *test_labels;
} data;

typedef struct __attribute__((aligned(64))) {
    int size;
    float *activations;
    float *weights;
    float *biases;
} layer;

typedef struct {
    int n;
    layer *layers;
    int num_correct_predictions;
} NN;

// Improved thread pool structure
typedef struct {
    int start_idx;
    int end_idx;
    NN *model;
    data *dataset;
    int thread_id;
    int num_threads;
    int *global_correct;
    pthread_mutex_t *mutex;
} ThreadPoolArg;

// Thread-local storage for better cache performance
typedef struct {
    float *temp_hidden1;
    float *temp_hidden2;
    float *temp_output;
    float *current_input;
} ThreadLocalBuffers;

int reverse_int(int i) {
    unsigned char c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

void read_mnist_images(const char *filename, float *images, int num_images) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        printf("Could not open file %s\n", filename);
        exit(1);
    }
    int magic_number, number_of_images, rows, cols;
    fread(&magic_number, sizeof(int), 1, fp);
    magic_number = reverse_int(magic_number);
    fread(&number_of_images, sizeof(int), 1, fp);
    number_of_images = reverse_int(number_of_images);
    fread(&rows, sizeof(int), 1, fp);
    rows = reverse_int(rows);
    fread(&cols, sizeof(int), 1, fp);
    cols = reverse_int(cols);
    
    for (int i = 0; i < num_images; ++i) {
        for (int r = 0; r < rows * cols; ++r) {
            unsigned char pixel = 0;
            fread(&pixel, sizeof(unsigned char), 1, fp);
            images[i * input_size + r] = (float)(pixel / 255.0);
        }
    }
    fclose(fp);
}

void read_mnist_labels(const char *filename, int *labels, int num_labels) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        printf("Could not open file %s\n", filename);
        exit(1);
    }
    int magic_number, number_of_labels;
    fread(&magic_number, sizeof(int), 1, fp);
    magic_number = reverse_int(magic_number);
    fread(&number_of_labels, sizeof(int), 1, fp);
    number_of_labels = reverse_int(number_of_labels);
    for (int i = 0; i < num_labels; ++i) {
        unsigned char label = 0;
        fread(&label, sizeof(unsigned char), 1, fp);
        labels[i] = (int)label;
    }
    fclose(fp);
}

void set_sizes(NN *rs) {
    rs->layers[0].size = input_size;
    for (int i = 0; i < rs->n; i++) {
        rs->layers[i + 1].size = hidden_size;
    }
    rs->layers[rs->n + 1].size = output_size;
}

void alloc_runstate(NN *rs, int n) {
    rs->n = n;
    rs->layers = (layer *)malloc((rs->n + 2) * sizeof(layer));
    set_sizes(rs);
    for (int i = 0; i < rs->n + 1; i++) {
        rs->layers[i].activations = (float *)calloc(rs->layers[i].size, sizeof(float));
        rs->layers[i].biases = (float *)calloc(rs->layers[i + 1].size, sizeof(float));
        rs->layers[i].weights = (float *)calloc(rs->layers[i + 1].size * rs->layers[i].size, sizeof(float));
    }
    rs->layers[rs->n + 1].activations = (float *)calloc(output_size, sizeof(float));
    rs->layers[rs->n + 1].biases = NULL;
    rs->layers[rs->n + 1].weights = NULL;
}

void free_runstate(NN *rs) {
    for (int i = 0; i < rs->n + 1; i++) {
        free(rs->layers[i].activations);
        free(rs->layers[i].biases);
        free(rs->layers[i].weights);
    }
    free(rs->layers[rs->n + 1].activations);
    free(rs->layers);
}

void malloc_data(data *dataset) {
    dataset->test_images = (float *)calloc(num_test_images * input_size, sizeof(float));
    dataset->test_labels = (int *)calloc(num_test_images, sizeof(int));
    dataset->training_images = (float *)calloc(num_train_images * input_size, sizeof(float));
    dataset->training_labels = (int *)calloc(num_train_images, sizeof(int));
}

void free_data(data *dataset) {
    free(dataset->test_images);
    free(dataset->test_labels);
    free(dataset->training_images);
    free(dataset->training_labels);
}

void prep_rs(NN *rs, data *dataset, int image_number, int is_training) {
    int previous_images = image_number * input_size;
    float *source = is_training ? dataset->training_images : dataset->test_images;
    memcpy(rs->layers[0].activations, source + previous_images, input_size * sizeof(float));
}

#if USE_AVX
// AVX-optimized ReLU
void relu_avx(float *x, int n) {
    const __m256 zero = _mm256_setzero_ps();
    int i;
    for (i = 0; i <= n - 8; i += 8) {
        __m256 vals = _mm256_loadu_ps(&x[i]);  // Use unaligned load for safety
        vals = _mm256_max_ps(vals, zero);
        _mm256_storeu_ps(&x[i], vals);  // Use unaligned store for safety
    }
    // Handle remaining elements
    for (; i < n; i++) {
        x[i] = x[i] < 0 ? 0 : x[i];
    }
}
#endif

void relu(float *x, int n) {
    for (int i = 0; i < n; i++) {
        x[i] = x[i] < 0 ? 0 : x[i];
    }
}

// Optimized softmax with numerical stability
void softmax_stable(float *x, int size) {
    // Find maximum for numerical stability
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    
    // Compute exp(x - max) and sum
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    
    // Normalize
    const float inv_sum = 1.0f / sum;
    for (int i = 0; i < size; i++) {
        x[i] *= inv_sum;
    }
}

int max_index(float *out, int size) {
    int max_i = 0;
    float max_val = out[0];
    for (int i = 1; i < size; i++) {
        if (out[i] > max_val) {
            max_val = out[i];
            max_i = i;
        }
    }
    return max_i;
}

#if USE_AVX
// AVX-optimized matrix multiplication
void matrix_multiply_avx(const float *weights, const float *input, float *output,
                        const float *biases, int input_size, int output_size) {
    for (int i = 0; i < output_size; i++) {
        __m256 sum = _mm256_setzero_ps();
        const float *w_row = &weights[i * input_size];
        
        int j;
        for (j = 0; j <= input_size - 8; j += 8) {
            __m256 w_vals = _mm256_loadu_ps(&w_row[j]);
            __m256 x_vals = _mm256_loadu_ps(&input[j]);
            sum = _mm256_fmadd_ps(w_vals, x_vals, sum);
        }
        
        // Horizontal sum of AVX register
        __m128 sum_high = _mm256_extractf128_ps(sum, 1);
        __m128 sum_low = _mm256_castps256_ps128(sum);
        __m128 sum128 = _mm_add_ps(sum_high, sum_low);
        __m128 sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
        __m128 sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));
        float result = _mm_cvtss_f32(sum32);
        
        // Handle remaining elements
        for (; j < input_size; j++) {
            result += w_row[j] * input[j];
        }
        
        output[i] = result + biases[i];
    }
}
#endif

// Regular matrix multiplication fallback
void matrix_multiply_regular(const float *weights, const float *input, float *output,
                           const float *biases, int input_size, int output_size) {
    for (int i = 0; i < output_size; i++) {
        float sum = biases[i];
        const float *w_row = &weights[i * input_size];
        for (int j = 0; j < input_size; j++) {
            sum += w_row[j] * input[j];
        }
        output[i] = sum;
    }
}

// Optimized forward pass with proper buffer management
void forward_pass_optimized(NN *model, const float *input, ThreadLocalBuffers *buffers) {
    const float *current_input = input;
    float *current_output;
    
    for (int layer = 0; layer < model->n + 1; layer++) {
        int input_dim = model->layers[layer].size;
        int output_dim = model->layers[layer + 1].size;
        
        // Select appropriate output buffer
        if (layer == 0) {
            current_output = buffers->temp_hidden1;
        } else if (layer == model->n) {
            current_output = buffers->temp_output;
        } else {
            current_output = (current_input == buffers->temp_hidden1) ? 
                           buffers->temp_hidden2 : buffers->temp_hidden1;
        }
        
        // Matrix multiplication
        #if USE_AVX
        if (input_dim >= 8 && input_dim % 8 == 0) {
            matrix_multiply_avx(model->layers[layer].weights, current_input, current_output,
                              model->layers[layer].biases, input_dim, output_dim);
        } else {
            matrix_multiply_regular(model->layers[layer].weights, current_input, current_output,
                                  model->layers[layer].biases, input_dim, output_dim);
        }
        #else
        matrix_multiply_regular(model->layers[layer].weights, current_input, current_output,
                              model->layers[layer].biases, input_dim, output_dim);
        #endif
        
        // Apply activation function (ReLU for hidden layers)
        if (layer < model->n) {
            #if USE_AVX
            if (output_dim >= 8 && output_dim % 8 == 0) {
                relu_avx(current_output, output_dim);
            } else {
                relu(current_output, output_dim);
            }
            #else
            relu(current_output, output_dim);
            #endif
        }
        
        current_input = current_output;
    }
    
    // Apply softmax to final output
    softmax_stable(buffers->temp_output, output_size);
}

ThreadLocalBuffers* allocate_thread_buffers() {
    ThreadLocalBuffers *buffers = (ThreadLocalBuffers*)malloc(sizeof(ThreadLocalBuffers));
    buffers->temp_hidden1 = (float*)calloc(hidden_size, sizeof(float));
    buffers->temp_hidden2 = (float*)calloc(hidden_size, sizeof(float));
    buffers->temp_output = (float*)calloc(output_size, sizeof(float));
    return buffers;
}

void free_thread_buffers(ThreadLocalBuffers *buffers) {
    if (buffers) {
        free(buffers->temp_hidden1);
        free(buffers->temp_hidden2);
        free(buffers->temp_output);
        free(buffers);
    }
}

void load_weights(NN *model, char *file_name) {
    FILE *file = fopen(file_name, "rb");
    if (file == NULL) {
        printf("Error opening file %s\n", file_name);
        exit(1);
    }
    int n;
    fread(&n, sizeof(int), 1, file);
    printf("This model has %d hidden layers\n", n);
    model->n = n;
    alloc_runstate(model, n);
    for (int i = 0; i < model->n + 1; i++) {
        fread(model->layers[i].weights, sizeof(float), model->layers[i].size * model->layers[i + 1].size, file);
        fread(model->layers[i].biases, sizeof(float), model->layers[i + 1].size, file);
    }
    fclose(file);
}

double test_sequential(NN *model, data *dataset) {
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    model->num_correct_predictions = 0;
    
    ThreadLocalBuffers *buffers = allocate_thread_buffers();
    
    for (int i = 0; i < num_test_images; i++) {
        const float *input = &dataset->test_images[i * input_size];
        forward_pass_optimized(model, input, buffers);
        
        int prediction = max_index(buffers->temp_output, output_size);
        if (prediction == dataset->test_labels[i]) {
            model->num_correct_predictions++;
        }
    }
    
    free_thread_buffers(buffers);
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    return (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
}

// Fixed pthread implementation
void *pthread_worker_fixed(void *arg) {
    ThreadPoolArg *targ = (ThreadPoolArg *)arg;
    ThreadLocalBuffers *buffers = allocate_thread_buffers();
    
    int local_correct = 0;
    
    for (int i = targ->start_idx; i < targ->end_idx; i++) {
        const float *input = &targ->dataset->test_images[i * input_size];
        forward_pass_optimized(targ->model, input, buffers);
        
        int prediction = max_index(buffers->temp_output, output_size);
        if (prediction == targ->dataset->test_labels[i]) {
            local_correct++;
        }
    }
    
    // Thread-safe update of global counter
    pthread_mutex_lock(targ->mutex);
    *(targ->global_correct) += local_correct;
    pthread_mutex_unlock(targ->mutex);
    
    free_thread_buffers(buffers);
    return NULL;
}

double test_pthread_fixed(NN *model, data *dataset) {
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    int num_threads = omp_get_max_threads();
    pthread_t *threads = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
    ThreadPoolArg *args = (ThreadPoolArg *)malloc(num_threads * sizeof(ThreadPoolArg));
    int global_correct = 0;
    pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
    
    int images_per_thread = num_test_images / num_threads;
    
    for (int t = 0; t < num_threads; t++) {
        args[t].start_idx = t * images_per_thread;
        args[t].end_idx = (t == num_threads - 1) ? num_test_images : (t + 1) * images_per_thread;
        args[t].model = model;
        args[t].dataset = dataset;
        args[t].thread_id = t;
        args[t].num_threads = num_threads;
        args[t].global_correct = &global_correct;
        args[t].mutex = &mutex;
        
        pthread_create(&threads[t], NULL, pthread_worker_fixed, &args[t]);
    }
    
    for (int t = 0; t < num_threads; t++) {
        pthread_join(threads[t], NULL);
    }
    
    model->num_correct_predictions = global_correct;
    
    free(threads);
    free(args);
    pthread_mutex_destroy(&mutex);
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    return (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
}

double test_openmp_optimized(NN *model, data *dataset) {
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    int total_correct = 0;
    
    #pragma omp parallel reduction(+:total_correct)
    {
        ThreadLocalBuffers *buffers = allocate_thread_buffers();
        
        #pragma omp for schedule(static)
        for (int i = 0; i < num_test_images; i++) {
            const float *input = &dataset->test_images[i * input_size];
            forward_pass_optimized(model, input, buffers);
            
            int prediction = max_index(buffers->temp_output, output_size);
            if (prediction == dataset->test_labels[i]) {
                total_correct++;
            }
        }
        
        free_thread_buffers(buffers);
    }
    
    model->num_correct_predictions = total_correct;
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    return (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
}

// Batch processing version for maximum throughput
double test_openmp_batch(NN *model, data *dataset) {
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    int total_correct = 0;
    const int local_batch_size = 32; // Smaller batches for better cache usage
    
    #pragma omp parallel reduction(+:total_correct)
    {
        ThreadLocalBuffers *buffers = allocate_thread_buffers();
        
        #pragma omp for schedule(dynamic)
        for (int batch_start = 0; batch_start < num_test_images; batch_start += local_batch_size) {
            int batch_end = (batch_start + local_batch_size <= num_test_images) ? 
                           batch_start + local_batch_size : num_test_images;
            
            for (int i = batch_start; i < batch_end; i++) {
                const float *input = &dataset->test_images[i * input_size];
                forward_pass_optimized(model, input, buffers);
                
                int prediction = max_index(buffers->temp_output, output_size);
                if (prediction == dataset->test_labels[i]) {
                    total_correct++;
                }
            }
        }
        
        free_thread_buffers(buffers);
    }
    
    model->num_correct_predictions = total_correct;
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    return (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
}

int main() {
    char *file_name = "saved_model.NN";
    NN model;
    data dataset;
    
    malloc_data(&dataset);
    
    printf("Loading test data...\n");
    read_mnist_images("./data/t10k-images.idx3-ubyte", dataset.test_images, num_test_images);
    read_mnist_labels("./data/t10k-labels.idx1-ubyte", dataset.test_labels, num_test_images);
    printf("Loaded test data\n");
    
    load_weights(&model, file_name);
    
    printf("Number of OpenMP threads: %d\n", omp_get_max_threads());
    printf("AVX support: %s\n", 
    #if USE_AVX
           "Yes"
    #else
           "No"
    #endif
    );
    
    printf("\nRunning sequential inference...\n");
    double seq_time = test_sequential(&model, &dataset);
    float seq_accuracy = (float)model.num_correct_predictions / num_test_images;
    printf("Sequential - Time: %.4f seconds, Accuracy: %.6f\n", seq_time, seq_accuracy);
    
    printf("\nRunning fixed pthread inference...\n");
    double pthread_time = test_pthread_fixed(&model, &dataset);
    float pthread_accuracy = (float)model.num_correct_predictions / num_test_images;
    printf("Pthread Fixed - Time: %.4f seconds, Accuracy: %.6f, Speedup: %.2fx\n", 
           pthread_time, pthread_accuracy, seq_time / pthread_time);
    
    printf("\nRunning optimized OpenMP inference...\n");
    double omp_time = test_openmp_optimized(&model, &dataset);
    float omp_accuracy = (float)model.num_correct_predictions / num_test_images;
    printf("OpenMP - Time: %.4f seconds, Accuracy: %.6f, Speedup: %.2fx\n", 
           omp_time, omp_accuracy, seq_time / omp_time);
    
    printf("\nRunning batch OpenMP inference...\n");
    double batch_time = test_openmp_batch(&model, &dataset);
    float batch_accuracy = (float)model.num_correct_predictions / num_test_images;
    printf("OpenMP Batch - Time: %.4f seconds, Accuracy: %.6f, Speedup: %.2fx\n", 
           batch_time, batch_accuracy, seq_time / batch_time);
    
    free_runstate(&model);
    free_data(&dataset);
    return 0;
}