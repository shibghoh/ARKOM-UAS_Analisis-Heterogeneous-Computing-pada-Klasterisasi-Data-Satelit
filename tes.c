#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <omp.h>
#include <CL/cl.h>

// ==========================================
// KODE KERNEL OPENCL (Dijalankan di GPU)
// ==========================================
const char *kernel_source = 
"__kernel void hitung_jarak_openCL(__global const uchar* pixel, __global const int* centroid, __global float* jarak, const int jumlah_pixel) {\n"
"    int i = get_global_id(0);\n"
"    if (i >= jumlah_pixel) return;\n"
"    \n"
"    for(int j = 0; j < 2; j++) {\n"
"        float r = (float)pixel[i * 3 + 0] - centroid[j * 3 + 0];\n"
"        float g = (float)pixel[i * 3 + 1] - centroid[j * 3 + 1];\n"
"        float b = (float)pixel[i * 3 + 2] - centroid[j * 3 + 2];\n"
"        \n"
"        jarak[i * 2 + j] = sqrt(r*r + g*g + b*b);\n"
"    }\n"
"}\n";

// ==========================================
// FUNGSI OPENMP (Dijalankan di CPU)
// ==========================================
void hitung_jarak_openMP(unsigned char* pixel, int* centroid, float* jarak, int jumlah_pixel) {
    #pragma omp parallel for
    for(int i = 0; i < jumlah_pixel; i++) {
        for(int j = 0; j < 2; j++) {
            float r = (float)pixel[i * 3 + 0] - centroid[j * 3 + 0];
            float g = (float)pixel[i * 3 + 1] - centroid[j * 3 + 1];
            float b = (float)pixel[i * 3 + 2] - centroid[j * 3 + 2];
            
            jarak[i * 2 + j] = sqrtf(r*r + g*g + b*b);
        }
    }
}

int main() {
    int ukuran_gambar[] = {128, 256, 512, 1024, 2048};
    int jumlah_pixel[] = {128*128, 256*256, 512*512, 1024*1024, 2048*2048};
    int jumlah_gambar = 5;
    
    int titik_centroid[5][2][3];
    srand(time(NULL));

    // 1. Cetak info gambar
    for (int i = 0; i < jumlah_gambar; i++) {
        printf("Gambar ke-%d | Ukuran: %dx%d\t | Total Pixel: %d\n", i+1, ukuran_gambar[i], ukuran_gambar[i], jumlah_pixel[i]);
    }
    printf("----------------------------------------------------------\n");

    // Inisialisasi data centroid acak
    for (int i = 0; i < jumlah_gambar; i++) {
        for(int c = 0; c < 2; c++) {
            titik_centroid[i][c][0] = rand() % 256;
            titik_centroid[i][c][1] = rand() % 256;
            titik_centroid[i][c][2] = rand() % 256;
        }
    }

    // 2. Setup Awal OpenCL (Boilerplate)
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;

    clGetPlatformIDs(1, &platform, NULL);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    queue = clCreateCommandQueue(context, device, 0, NULL);

    program = clCreateProgramWithSource(context, 1, &kernel_source, NULL, NULL);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    kernel = clCreateKernel(program, "hitung_jarak_openCL", NULL);

    // 3. Proses Warm-up (Pemanasan) OpenMP agar alokasi thread pertama kali tidak merusak catatan waktu
    unsigned char dummy_pixel[3] = {0, 0, 0};
    int dummy_centroid[2][3] = {{0,0,0}, {0,0,0}};
    float dummy_jarak[2] = {0, 0};
    hitung_jarak_openMP(dummy_pixel, (int*)dummy_centroid, dummy_jarak, 1);

    printf("\n=== MEMULAI PENGUJIAN KOMPARASI ===\n\n");

    // Loop Utama Pengujian Gambar
    for(int i = 0; i < jumlah_gambar; i++) {
        int n_pixel = jumlah_pixel[i];
        
        // Alokasi memori di RAM (Host) menggunakan pointer biasa yang ringkas
        unsigned char* data_pixel = (unsigned char*)malloc(n_pixel * 3 * sizeof(unsigned char));
        float* jarak_omp = (float*)malloc(n_pixel * 2 * sizeof(float));
        float* jarak_ocl = (float*)malloc(n_pixel * 2 * sizeof(float));

        if (data_pixel == NULL || jarak_omp == NULL || jarak_ocl == NULL) {
            printf("Memori penuh untuk gambar ke-%d!\n", i+1);
            return -1;
        }

        // Isi piksel dengan angka acak
        for(int j = 0; j < n_pixel * 3; j++) {
            data_pixel[j] = rand() % 256;
        }
        
        // --- PENGUJIAN 1: OPENMP (CPU) ---
        double start_omp = omp_get_wtime();
        hitung_jarak_openMP(data_pixel, (int*)titik_centroid[i], jarak_omp, n_pixel);
        double end_omp = omp_get_wtime();
        double waktu_omp = end_omp - start_omp;

        // --- PENGUJIAN 2: OPENCL (GPU) ---
        struct timespec start_ocl, end_ocl;
        clock_gettime(CLOCK_MONOTONIC, &start_ocl);

        // Alokasi dan transfer memori ke VRAM GPU
        cl_mem pixel_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, n_pixel * 3 * sizeof(unsigned char), data_pixel, NULL);
        cl_mem centroid_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 2 * 3 * sizeof(int), titik_centroid[i], NULL);
        cl_mem jarak_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, n_pixel * 2 * sizeof(float), NULL, NULL);

        clSetKernelArg(kernel, 0, sizeof(cl_mem), &pixel_buf);
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &centroid_buf);
        clSetKernelArg(kernel, 2, sizeof(cl_mem), &jarak_buf);
        clSetKernelArg(kernel, 3, sizeof(int), &n_pixel);

        size_t global_size = n_pixel;
        clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, NULL, 0, NULL, NULL);
        clEnqueueReadBuffer(queue, jarak_buf, CL_TRUE, 0, n_pixel * 2 * sizeof(float), jarak_ocl, 0, NULL, NULL);

        clock_gettime(CLOCK_MONOTONIC, &end_ocl);
        double waktu_ocl = (end_ocl.tv_sec - start_ocl.tv_sec) + (end_ocl.tv_nsec - start_ocl.tv_nsec) / 1e9;

        // Cetak Perbandingan Hasil
        printf("Gambar ke-%d (%dx%d):\n", i+1, ukuran_gambar[i], ukuran_gambar[i]);
        printf("  -> Waktu OpenMP (CPU) : %f detik\n", waktu_omp);
        printf("  -> Waktu OpenCL (GPU) : %.9f detik\n", waktu_ocl);
        printf("----------------------------------------------------------\n");

        // Bersihkan memori buffer GPU dan RAM CPU per iterasi gambar
        clReleaseMemObject(pixel_buf);
        clReleaseMemObject(centroid_buf);
        clReleaseMemObject(jarak_buf);
        free(data_pixel);
        free(jarak_omp);
        free(jarak_ocl);
    }
    
    // Bersihkan objek OpenCL utama di akhir program
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}