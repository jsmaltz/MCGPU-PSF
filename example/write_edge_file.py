# write_voxel_edges.py
# Creates a plain-text edges file:
#   MCGPU_EDGES v1
#   nx ny nz
#   x: <nx+1 floats>
#   y: <ny+1 floats>
#   z: <nz+1 floats>

from pathlib import Path

def write_edges_txt(out_path, nx, ny, nz, origin_cm, pitch_cm, float_fmt="{:.6f}"):
    """
    out_path : str/Path - output edges.txt
    nx,ny,nz : int      - voxel counts along x,y,z
    origin_cm: (ox,oy,oz) in cm (world coordinates of lower/back corner)
    pitch_cm : (px,py,pz) voxel sizes in cm (uniform per axis)
    float_fmt: format string for floats
    """
    ox, oy, oz = origin_cm
    px, py, pz = pitch_cm

    # Build edges (strictly increasing)
    x_edges = [ox + i*px for i in range(nx+1)]
    y_edges = [oy + j*py for j in range(ny+1)]
    z_edges = [oz + k*pz for k in range(nz+1)]

    # Validate monotonicity
    def mono(v): return all(v[i] > v[i-1] for i in range(1, len(v)))
    assert mono(x_edges) and mono(y_edges) and mono(z_edges), "Edges must be strictly increasing"

    # Write file
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("# MCGPU_EDGES v1\n")
        f.write(f"{nx} {ny} {nz}\n")
        f.write("x: " + " ".join(float_fmt.format(v) for v in x_edges) + "\n")
        f.write("y: " + " ".join(float_fmt.format(v) for v in y_edges) + "\n")
        f.write("z: " + " ".join(float_fmt.format(v) for v in z_edges) + "\n")

if __name__ == "__main__":
    # ---- Fill from your VOXELIZED GEOMETRY section ----
    voxel_geom_file = "phantoms/pc_1344613780_crop_res.raw.gz"   # not used here, just for reference
    origin_cm = (0.0, 0.0, 0.0)                                  # OFFSET [cm]
    nx, ny, nz = 280, 520, 107                                   # NUMBER OF VOXELS
    pitch_cm = (0.050, 0.050, 0.050)                             # VOXEL SIZES [cm]
    density_vox_path = "phantoms/pc_1344613780_density.raw.gz"   # not used here
    vox_edge_path = "phantoms/voxel_edges.txt"                   # output

    write_edges_txt(vox_edge_path, nx, ny, nz, origin_cm, pitch_cm)
    print(f"Wrote edges to: {vox_edge_path}")
