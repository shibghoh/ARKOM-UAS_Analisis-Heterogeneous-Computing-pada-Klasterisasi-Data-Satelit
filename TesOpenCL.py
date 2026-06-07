import pyopencl as cl

try:
    # Mengambil semua platform OpenCL yang terinstal di PC (Nvidia, AMD, atau Intel)
    platforms = cl.get_platforms()
    
    if not platforms:
        print("Status OpenCL: ERROR (Tidak ada platform OpenCL yang ditemukan).")
    else:
        print(f"Ditemukan {len(platforms)} Platform OpenCL:\n" + "="*40)
        for p in platforms:
            print(f"Nama Platform: {p.name}")
            print(f"Vendor       : {p.vendor}")
            print(f"Versi Driver : {p.version}")
            
            # Cek perangkat (GPU/CPU) di dalam platform tersebut
            devices = p.get_devices()
            for d in devices:
                print(f"  -> Perangkat: {d.name} ({cl.device_type.to_string(d.type)})")
            print("="*40)
            print("Status OpenCL: SIAP & BERFUNGSI!")

except Exception as e:
    print("Status OpenCL: ERROR!")
    print(f"Detail Error: {e}")
    print("\nSolusi: Pastikan driver GPU (Nvidia/AMD/Intel) Anda sudah terinstal dengan benar.")