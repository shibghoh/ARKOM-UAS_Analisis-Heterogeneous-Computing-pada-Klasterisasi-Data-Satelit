#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <omp.h>

void hitung_jarak_openMP(unsigned char (*pixel)[3], int (*centroid)[3], float (*jarak)[2], int jumlah_pixel) {
    
    #pragma omp parallel for
    for(int i = 0; i < jumlah_pixel; i++) {
        // printf("Pixel %d\n", i);
        for(int c = 0; c < 2; c++) {
            float r = (float)pixel[i][0] - centroid[c][0];
            float g = (float)pixel[i][1] - centroid[c][1];
            float b = (float)pixel[i][2] - centroid[c][2];
            
            jarak[i][c] = sqrtf(r*r + g*g + b*b);
        }
    }
}

int main() {
    int ukuran_gambar[] = {128, 256, 512, 1024, 2048};
    int jumlah_pixel[] = {128*128, 256*256, 512*512, 1024*1024, 2048*2048};
    int jumlah_gambar = 5;
    
    int titik_centroid[5][2][3];
    
    srand(time(NULL));

    

    for (int i = 0; i < jumlah_gambar; i++) {
        printf("Gambar ke-%d | Ukuran: %dx%d | Total Pixel: %d\n", i+1, ukuran_gambar[i], ukuran_gambar[i], jumlah_pixel[i]);
    }
    printf("--------------------------------------------------\n");

    for (int i = 0; i < jumlah_gambar; i++) {
        for(int c = 0; c < 2; c++) {
            titik_centroid[i][c][0] = rand() % 256;
            titik_centroid[i][c][1] = rand() % 256;
            titik_centroid[i][c][2] = rand() % 256;
        }
    }
    
    unsigned char dummy_pixel[3] = {0, 0, 0};
    int dummy_centroid[2][3] = {{0,0,0}, {0,0,0}};
    double dummy_jarak[2] = {0, 0};
    hitung_jarak_openMP(dummy_pixel, (int*)dummy_centroid, dummy_jarak, 1);
        

    for(int i = 0; i < jumlah_gambar; i++) {
        
        int n_pixel = jumlah_pixel[i];
        
        unsigned char (*data_pixel)[3] = malloc(sizeof(unsigned char[n_pixel][3]));
        float (*jarak)[2] = malloc(sizeof(float[n_pixel][2]));

        if (data_pixel == NULL || jarak == NULL) {
            printf("Memori penuh untuk gambar ke-%d!\n", i+1);
            return -1;
        }

        for(int j = 0; j < n_pixel; j++) {
            data_pixel[j][0] = rand() % 256;
            data_pixel[j][1] = rand() % 256;
            data_pixel[j][2] = rand() % 256;
        }
        
        double start_time = omp_get_wtime();

        hitung_jarak_openMP(data_pixel, titik_centroid[i], jarak, n_pixel);

        double end_time = omp_get_wtime();
        
        printf("Waktu eksekusi OpenMP untuk gambar ke-%d (%dx%d): %f detik\n", 
               i+1, ukuran_gambar[i], ukuran_gambar[i], end_time - start_time);

        free(data_pixel);
        free(jarak);
    }
    
    return 0;
}