from numba import config, njit, prange
import numpy as np

# 1. Cek jumlah thread CPU yang terdeteksi oleh Numba
print(f"Jumlah Thread CPU yang siap digunakan: {config.NUMBA_NUM_THREADS}")

# 2. Tes fungsi paralel sederhana
@njit(parallel=True)
def tes_paralel():
    sistem_ok = True
    for i in prange(10):
        pass
    return sistem_ok

if tes_paralel():
    print("Status OpenMP (Numba): SIAP & BERFUNGSI!")