# Description

Eye detection algorithm using OpenCV image processing library.

## Build

CMake required to build project.

```bash
mkdir build
cd build
cmake ..
make -j4
```

## Usage
#### Using a recorded video:
```bash
./track ../your_video_path
```
#### Using default system camera:
```bash
./track 0
```