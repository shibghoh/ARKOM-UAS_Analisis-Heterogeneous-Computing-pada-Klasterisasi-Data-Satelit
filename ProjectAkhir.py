import time
from tracemalloc import start
from tracemalloc import start
import numpy as np
from numba import config, njit, prange
from PIL import Image
import pyopencl as cl

@njit
def euclidean(a, b):
  return np.sqrt(np.sum((a-b)**2))

@njit(parallel=True)
def KMeans_euclidean(pixel, centroid):
    converged = False
    q = 1
    
    # Siapkan array label kosong (lebih cepat daripada list .append)
    label = np.zeros(pixel.shape[0], dtype=np.int32)
    
    # Salin centroid ke tipe float agar kalkulasi rata-rata akurat
    centroid = centroid.astype(np.float32)
    
    while not converged:
        # Catatan: Numba mendukung print standar, hindari f-string kompleks jika error
        print("Iterasi ke-", q) 
        q += 1

        # PARALELISASI (Konsep OpenMP): Membagi proses hitung piksel ke semua core CPU
        for p_idx in prange(pixel.shape[0]):
            p = pixel[p_idx]
            jarak = np.zeros(centroid.shape[0])
            for c_idx in range(centroid.shape[0]):
                jarak[c_idx] = euclidean(p, centroid[c_idx])
            label[p_idx] = np.argmin(jarak)

        old_centroid = centroid.copy()

        # Update centroid
        for i in range(centroid.shape[0]):
            # Filter piksel yang masuk ke dalam cluster i
            mask = (label == i)
            total_piksel_cluster = np.sum(mask)

            if total_piksel_cluster > 0:
                # Menghitung rata-rata manual karena axis pada array 2D yang difilter 
                # kadang memicu error typing di Numba
                sum_rgb = np.zeros(3, dtype=np.float32)
                for p_idx in range(pixel.shape[0]):
                    if label[p_idx] == i:
                        sum_rgb += pixel[p_idx]
                centroid[i] = sum_rgb / total_piksel_cluster
            else:
                centroid[i] = old_centroid[i]

        # Cek konvergensi
        if np.allclose(old_centroid, centroid, atol=1e-4):
            converged = True

    return label, centroid  # PERBAIKAN 3: Pindahkan return ke luar loop while

IMG_SIZE = [(512, 512), (1024, 1024), (2048, 2048)]

img = Image.open('10452_sat.jpg')
img_resize  = [None] * len(IMG_SIZE)
pixel_img = [None] * len(IMG_SIZE)
label_img = [None] * len(IMG_SIZE)
start_time = [None] * len(IMG_SIZE)
end_time = [None] * len(IMG_SIZE)
start_time_cl = [None] * len(IMG_SIZE)
end_time_cl = [None] * len(IMG_SIZE)
centroid = [None] * len(IMG_SIZE)
centroid_cl = [None] * len(IMG_SIZE)


for i, size in enumerate(IMG_SIZE):
    img_resize[i] = img.resize(size)
    print(f"Gambar {i} : {size}")
    img_resize[i] = np.array(img_resize[i])
    pixel_img[i] = img_resize[i].reshape(-1, 3)

K = 3
for i in range(len(IMG_SIZE)):
    centroid[i] = pixel_img[i][np.random.choice(len(pixel_img[i]), K, replace=False)]

for i in range(len(IMG_SIZE)):
    start_time[i] = time.time()
    label_img[i], centroid[i] = KMeans_euclidean(pixel_img[i], centroid[i])
    end_time[i] = time.time()
    print(f"Waktu eksekusi untuk gambar {i} : {end_time[i] - start_time[i]:.4f} detik")

for i in range(len(IMG_SIZE)):
    print(f"Waktu eksekusi untuk gambar berukuran {IMG_SIZE[i]} : {end_time[i] - start_time[i]:.4f} detik")


ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags

# 2. Tulis Kode Kernel OpenCL (Bahasa C khusus GPU)
# Kernel ini menghitung jarak dari tiap piksel ke semua centroid secara paralel
kernel_src = """
__kernel void hitung_jarak_kmeans(
    __global const uchar* pixel,     // Data piksel (R,G,B)
    __global const float* centroid, // Data posisi centroid
    __global int* label,            // Output cluster hasil id
    const int num_clusters,
    const int total_pixels) 
{
    int p_idx = get_global_id(0);
    if (p_idx >= total_pixels) return;

    // Ambil data RGB piksel ini
    float r = (float)pixel[p_idx * 3];
    float g = (float)pixel[p_idx * 3 + 1];
    float b = (float)pixel[p_idx * 3 + 2];

    int min_idx = 0;
    float min_dist = 1e10; // Angka awal besar

    // Cari jarak Euclidean terkecil
    for (int c_idx = 0; c_idx < num_clusters; c_idx++) {
        float c_r = centroid[c_idx * 3];
        float c_g = centroid[c_idx * 3 + 1];
        float c_b = centroid[c_idx * 3 + 2];

        // Rumus kuadrat jarak Euclidean (tidak perlu akar/sqrt untuk performa lebih cepat)
        float dist = (r - c_r)*(r - c_r) + (g - c_g)*(g - c_g) + (b - c_b)*(b - c_b);

        if (dist < min_dist) {
            min_dist = dist;
            min_idx = c_idx;
        }
    }

    // Simpan id cluster terdekat ke memori output
    label[p_idx] = min_idx;
}
"""
# Compile program di GPU
prg = cl.Program(ctx, kernel_src).build()

def KMeans_OpenCL(pixel, centroid):
    converged = False
    q = 1
    total_pixels = pixel.shape[0]
    num_clusters = centroid.shape[0]
    
    # Siapkan array penampung label di CPU
    label_host = np.zeros(total_pixels, dtype=np.int32)
    
    # Alokasi buffer memori di GPU (VRAM)
    pixel_gpu = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=pixel)
    label_gpu = cl.Buffer(ctx, mf.WRITE_ONLY, label_host.nbytes)

    while not converged:
        print("Iterasi ke-", q)
        q += 1

        # Pastikan tipe data centroid bertipe float32 untuk GPU
        centroid = centroid.astype(np.float32)
        
        # Salin posisi centroid terbaru dari CPU ke GPU
        centroid_gpu = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=centroid)

        # 1. TAHAP ASSIGNMENT (Eksekusi di GPU secara paralel masal)
        # Global size disesuaikan dengan total piksel gambar
        prg.hitung_jarak_kmeans(
            queue, (total_pixels,), None,
            pixel_gpu, centroid_gpu, label_gpu,
            np.int32(num_clusters), np.int32(total_pixels)
        )

        # Salin balik data label hasil hitungan GPU ke RAM CPU
        cl.enqueue_copy(queue, label_host, label_gpu)

        old_centroid = centroid.copy()

        # 2. TAHAP UPDATE (Dilakukan di CPU karena lebih efisien mengagregasi data)
        sum_rgb = np.zeros((num_clusters, 3), dtype=np.float32)
        count_cluster = np.zeros(num_clusters, dtype=np.int32)
        
        for p_idx in range(total_pixels):
            l = label_host[p_idx]
            sum_rgb[l] += pixel[p_idx]
            count_cluster[l] += 1
            
        for i in range(num_clusters):
            if count_cluster[i] > 0:
                centroid[i] = sum_rgb[i] / count_cluster[i]
            else:
                centroid[i] = old_centroid[i]

        # Cek konvergensi
        if np.allclose(old_centroid, centroid, atol=1e-4):
            converged = True

    return label_host, centroid

for i in range(len(IMG_SIZE)):
    # Inisialisasi ulang centroid awal secara acak
    centroid_awal = pixel_img[i][np.random.choice(len(pixel_img[i]), K, replace=False)].astype(np.float32)
    
    start_time_cl[i] = time.time()
    label_hasil, centroid_hasil = KMeans_OpenCL(pixel_img[i], centroid_awal)
    end_time_cl[i] = time.time()
    centroid_cl[i] = centroid_hasil

    print(f"Waktu eksekusi OpenCL untuk gambar {i} : {end_time_cl[i] - start_time_cl[i]:.4f} detik\n")


print("\Hasil Waktu Eksekusi:")
for i in range(len(IMG_SIZE)):
    print(f"Waktu eksekusi OpenMP untuk gambar berukuran {IMG_SIZE[i]} : {end_time[i] - start_time[i]:.4f} detik")

for i in range(len(IMG_SIZE)):
    print(f"Waktu eksekusi OpenCL untuk gambar berukuran {IMG_SIZE[i]} : {end_time_cl[i] - start_time_cl[i]:.4f} detik")