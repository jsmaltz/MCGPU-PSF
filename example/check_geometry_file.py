import gzip
import numpy as np


#filename = "phantoms/pc_1344613780_crop_res.raw.gz"; nx, ny, nz = 280, 520, 107
filename = "phantoms/Graff_hetero_15087760.raw.gz";  nx, ny, nz = 1280,   1950,   940  

with gzip.open(filename, "rb") as f:
    data = np.frombuffer(f.read(), dtype=np.uint8)

print("Filename:", filename)    
print("Voxel count:", data.size)
print("Expected:", nx*ny*nz)
print("Unique materials:", np.unique(data))
