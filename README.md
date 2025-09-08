# MC-GPU PSF

---

# MCGPU-PSF (Windows & CUDA 13 compatible)

This fork adapts **MC-GPU-PSF** for modern CUDA toolkits (12/13) and Windows/MSVC builds,
fixes binary output on Windows.

## What’s changed

- **CUDA 13 compatibility**
  - Replace `cudaThreadSynchronize`/`cudaThreadExit` with device-runtime aliases.
  - Query device attributes via `cudaDeviceGetAttribute` (e.g., clock rate).
  - `#if defined(CUDART_VERSION) && (CUDART_VERSION >= 13000)` guards around legacy fields.

- **Windows build**
  - Example `nvcc` command for VS2022 toolchain.
  - Increase stack reserve (`/STACK:33554432`) to avoid early stack overflow.
  - Link against zlib (vcpkg or static) cleanly.

- **Binary output correctness**
  - Open `.raw` outputs with `"wb"` to prevent CRLF corruption on Windows.

- **MPI (optional)**
  - `-DUSE_MPI` support with MS-MPI include/lib paths; guards to allow non-MPI builds.

## Typical quick build (Windows / CUDA 13 / RTX 4080)

```bat
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
nvcc -O3 -use_fast_math -DUSING_CUDA -arch=sm_89 ^
  -DcudaThreadSynchronize=cudaDeviceSynchronize ^
  -DcudaThreadExit=cudaDeviceReset ^
  -Xlinker /STACK:33554432 ^
  -I . -I "cuda-samples\Common" -I "%CUDA_PATH%\include" ^
  -I "C:\vcpkg\installed\x64-windows\include" ^
  -L "C:\vcpkg\installed\x64-windows\lib" "C:\vcpkg\installed\x64-windows\lib\zlib.lib" ^
  -o mcgpu.exe .\MC-GPU_v1.5b.cu

## MPI build (Code is leaky in having MPI calls in non-MPI branches, easier to just use it)

set MSMPI_INC=C:\Program Files (x86)\Microsoft SDKs\MPI\Include
set MSMPI_LIB64=C:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x64

nvcc ... -DUSE_MPI ^
  -I "%MSMPI_INC%" -L "%MSMPI_LIB64%" "%MSMPI_LIB64%\msmpi.lib" ^
  -o mcgpu.exe .\MC-GPU_v1.5b.cu

License & attribution

This repository is a fork of gfrmd-ifgw/MCGPU-PSF
.
Original authors and license apply; see LICENSE.

---


This repository contains the tools described in the technical note:
**"Technical Note: MC-GPU breast dosimetry validations with other Monte Carlo codes and Phase Space File implementation"**
RT Massera, RM Thomson and A Tomal.

How to cite:

Massera, RT, Thomson, RM, Tomal, A. Technical note: MC-GPU breast dosimetry validations with other Monte Carlo codes and phase space file implementation. Med. Phys. 2022; 49: 244– 253. https://doi.org/10.1002/mp.15342

Each folder contains specific codes and implementations (with further informations of the original sources).


Contens:
1. [MCGPU_files](MCGPU_files/README.md): the modified MC-GPU files to include phase space files
2. [example](example/README.md): a simulation for testing the phase space files
3. [PSFBin2IAEA](PSFBin2IAEA/README.md):  Fortran routine to convert raw binary files to IAEA format

Instructions:
- Compile MC-GPU binary [1](MCGPU_files/) and place on example folder [2](example/). 
- Run the example following the instructions within the folder.
- After the simulation is finished, three files will be generated in examples/images: mcgpu_image_x_psf.raw (x=1,2,3).
- Compile the psfconvert binary [3](PSFBin2IAEA/).
- Run psfbin2fort.py script following the instructions to convert the PSF in binary to Fortran compatible.
- Copy the psfconvert executable to the directory containing the PSF binary (the one generated with the Python script) and run
- If successful, two files will be generated: filename.IAEAheader and  filename.IAEAphsp.
- Those PSF now can be loaded in other MC codes.

Each code has a specific license, indicated in each folder. Otherwise, if not indicated, the contributions of this work are released under the LICENSE file in the main directory.
