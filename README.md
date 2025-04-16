# Convolution 2D Library

This project provides a 2D convolution implementation in C, using both **naive** and **FFT-based** approaches. It supports the use of the **FFTW3** library for fast Fourier transforms and utilizes **OpenMP** for parallel computation.

## Features

- Naive convolution for small kernels
- FFT-based convolution for large inputs
- Configurable build system with **CMake**
- Written in pure C
- Linux-compatible

## Dependencies

Make sure you have the following installed on your Linux system:

- `gcc` or `clang`
- `cmake` (version >= 3.10)
- `pkg-config`
- `libfftw3f-dev` (single precision FFTW3 library)
- `libomp-dev` (for OpenMP support)

You can install dependencies using:

```bash
sudo apt update
sudo apt install build-essential cmake pkg-config libfftw3f-dev libomp-dev
```

## Build Instructions

1. Clone this repository:
   ```bash
   git clone https://github.com/isela31/convolution_2D.git
   cd convolution_2D
   ```

2. Create a build directory:
   ```bash
   mkdir build
   cd build
   ```

3. Run `cmake` and build:
   ```bash
   cmake ..
   make
   ```

4. Run the test program:
   ```bash
   ./conv2d_test
   ```

## Project Structure

```
.
├── CMakeLists.txt
├── include/
│   └── conv2d.h
├── src/
│   ├── conv2d.c
│   └── main.c
└── build/
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

Feel free to contribute or suggest improvements. 
