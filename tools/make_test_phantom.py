import gzip
import numpy as np
from pathlib import Path
from write_edge_file import write_edges_txt

def write_material_list(path_txt: str, material_paths_with_density):
    """
    Writes a MC-GPU-style MATERIAL FILE LIST block (text).
    material_paths_with_density: list of tuples (path, density_or_None, comment)
    """
    p = Path(path_txt)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        f.write("#[SECTION MATERIAL FILE LIST v.2009-11-30]\n")
        for i, (mpath, dens, comment) in enumerate(material_paths_with_density, start=1):
            if dens is None:
                f.write(f"{mpath:<40} # {i}th MATERIAL FILE {comment}\n")
            else:
                f.write(f"{mpath:<40} density={dens:<8g} # {i}th MATERIAL FILE {comment}\n")


def write_vox_ascii_gz(path_vox_gz: str,
                       nx: int, ny: int, nz: int,
                       voxel_size_cm=(0.1, 0.1, 0.1),
                       mat_id_1based=None,
                       rho_gcc=None):
    """
    Writes an ASCII .vox.gz file readable by your load_voxels() (VOXELS HEADER + per-voxel lines).
    mat_id_1based: uint16/int array shaped (nz, ny, nx) with values >=1
    rho_gcc: float array shaped (nz, ny, nx) with densities >0
    """
    assert mat_id_1based is not None
    assert rho_gcc is not None
    assert mat_id_1based.shape == (nz, ny, nx)
    assert rho_gcc.shape == (nz, ny, nx)

    vx, vy, vz = voxel_size_cm
    p = Path(path_vox_gz)
    p.parent.mkdir(parents=True, exist_ok=True)

    header = []
    header.append("# Test voxel file for MC-GPU (ASCII VOX format)\n")
    header.append("# Voxel order: X runs first, then Y, then Z.\n")
    header.append("[SECTION VOXELS HEADER v.2008-04-13]\n")
    header.append(f"{nx} {ny} {nz}      No. OF VOXELS IN X,Y,Z\n")
    header.append(f"{vx:.6e} {vy:.6e} {vz:.6e}    VOXEL SIZE (cm) ALONG X,Y,Z\n")
    header.append("1                  COLUMN NUMBER WHERE MATERIAL ID IS LOCATED\n")
    header.append("2                  COLUMN NUMBER WHERE THE MASS DENSITY IS LOCATED\n")
    header.append("0                  BLANK LINES AT END OF X,Y-CYCLES (1=YES,0=NO)\n")
    header.append("[END OF VXH SECTION]\n")

    with gzip.open(p, "wt", encoding="utf-8", newline="\n") as f:
        f.writelines(header)

        # Write voxel lines: i fastest, then j, then k (matches your loader loops)
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    mid = int(mat_id_1based[k, j, i])
                    dens = float(rho_gcc[k, j, i])
                    # Your loader requires mid>=1 and dens>0
                    f.write(f"{mid:d} {dens:.8g}\n")


def write_density_cube_gz(path_rho_gz: str, rho_gcc_zyx: np.ndarray):
    """
    Writes float32 LE density cube as raw.gz (binary) in x-fastest order.
    Your load_density_cube expects nvox float32.
    rho_gcc_zyx: (nz, ny, nx)
    """
    rho = np.asarray(rho_gcc_zyx, dtype=np.float32)
    # Convert to x-fastest linear (i + nx*(j + ny*k))
    rho_linear = rho.reshape(-1)  # already C-order z,y,x -> linear with x fastest
    p = Path(path_rho_gz)
    p.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(p, "wb") as f:
        f.write(rho_linear.tobytes(order="C"))


def main():
    # Geometry
    nx, ny, nz = 10, 20, 50
    voxel_size_cm = (1, 0.5, 0.1)
    origin_cm = (0.0, 0.0, 0.0)                                  # OFFSET [cm]
    pitch_cm =  voxel_size_cm
    
    # Materials (1-based IDs in voxel file!)
    MAT_AIR = 1
    MAT_TISSUE = 2

    rho_air = 0.001205
    rho_tissue = 1.00

    # Build phantom: air everywhere, tissue brick in the middle
    mat = np.full((nz, ny, nx), MAT_AIR, dtype=np.int16)      # (z,y,x)
    rho = np.full((nz, ny, nx), rho_air, dtype=np.float32)

    # Tissue region (edit as you like)
    # Center-ish brick: x=3..6, y=3..7, z=1..3
    mat[1:4, 3:8, 3:7] = MAT_TISSUE
    rho[1:4, 3:8, 3:7] = rho_tissue

    # Output paths
    out_dir = Path("../example/phantoms/")
    vox_path = out_dir / "toy_10x11x5.vox.gz"
    rho_path = out_dir / "toy_10x11x5_density.raw.gz"
    matlist_path = out_dir / "toy_material_list.txt"
    vox_edge_path = out_dir/ "toy_voxel_edges.txt" # output
  
    # Material list block (you can paste into your input deck)
    write_material_list(
        matlist_path,
        [
            ("Materials/air_nist.mcgpu.gz",    rho_air,    "(air)"),
            ("Materials/pmma_nist.mcgpu.gz", rho_tissue, "(tissue)"),
        ],
    )

    # Voxel file for load_voxels() (ASCII)
    write_vox_ascii_gz(
        vox_path,
        nx, ny, nz,
        voxel_size_cm=voxel_size_cm,
        mat_id_1based=mat,
        rho_gcc=rho,
    )

    # Density cube for your load_density_cube() (binary float32 LE)
    write_density_cube_gz(rho_path, rho)

    write_edges_txt(vox_edge_path, nx, ny, nz, origin_cm, pitch_cm)

    
    print("Wrote:")
    print(f"  {matlist_path}")
    print(f"  {vox_path}")
    print(f"  {rho_path}")
    print(f"  {vox_edge_path}")    
    print("Notes:")
    print("  - voxel file uses 1-based material IDs (required by your load_voxels).")
    print("  - density cube matches the same rho field (float32).")


if __name__ == "__main__":
    main()
