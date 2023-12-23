# NeRF-VO

Real-Time Sparse Visual Odometry with Neural Radiance Fields

## Installation

### PyTorch and PyTorch Scatter
```bash
pip install torch==1.9.0+cu118 torchvision==0.10.0+cu118 -f https://download.pytorch.org/whl/cu118
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.9.0+cu118.html
```

### Requirements
```bash
pip install -r requirements.txt
```

### Build dependencies
#### Build Tiny CUDA Neural Networks
```bash
cd nerf_vo/thirdparty/tiny_cuda_nn/bindings/torch
python setup.py install
```

#### Build Lietorch
```bash
cd nerf_vo/thirdparty/lietorch
python setup.py install
```

#### Build DPVO
```bash
cd nerf_vo/thirdparty/dpvo
wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip
unzip eigen-3.4.0.zip -d thirdparty
pip install .
```

### Download pre-trained models
#### DPVO
```bash
wget https://www.dropbox.com/s/nap0u8zslspdwm4/models.zip -P nerf_vo/build/dpvo/
```

#### Omnidata Depth Estimator
```bash
gdown '1Jrh-bRnJEjyMCS7f-WsaFlccfPjJPPHI&confirm=t' -O nerf_vo/build/omnidata_models/
```

#### Omnidata Normal Estimator
```bash
gdown '1wNxVO4vVbDEMEpnAi_jwQObf2MFodcBR&confirm=t' -O nerf_vo/build/omnidata_models/
```

#### DROID-SLAM
```bash
gdown '1PpqVt1H4maBa_GbPJp4NwxRsd9jk-elh&confirm=t' -O nerf_vo/build/droid_slam/
```

### Build NeRF-SLAM

#### Build Instant-NGP
```bash
cmake nerf_vo/thirdparty/nerf_slam/thirdparty/instant-ngp -B nerf_vo/build/instant_ngp
cmake --build nerf_vo/build/instant_ngp --config RelWithDebInfo -j8
```

#### Build GTSAM
```bash
cmake nerf_vo/thirdparty/nerf_slam/thirdparty/gtsam -DGTSAM_BUILD_PYTHON=1 -B nerf_vo/build/gtsam
cmake --build nerf_vo/build/gtsam --config RelWithDebInfo -j8
```

- Fix GTSAM build error by modifying `nerf_vo/build/gtsam/python/linear.cpp` (line 400):
```cpp
.def(py::init<const gtsam::KeyVector&, const std::vector<gtsam::Matrix>&, const std::vector<gtsam::Vector>&, double>(), py::arg("js"), py::arg("Gs"), py::arg("gs"), py::arg("f"))
```

```bash
cd nerf_vo/build/gtsam
make python-install
```

#### Build NeRF-SLAM
```bash
cd nerf_vo/thirdparty/nerf_slam
python setup.py install
```

## Datasets

### Replica
```bash
bash scripts/download_replica.sh 
```

### ScanNet
```bash
python scripts/download_scannet.py
```

### 7-Scenes
```bash
bash scripts/download_7_scenes.sh
```

### TUM-RGBD
```bash
bash scripts/download_tum_rgbd.sh 
```

## Execution
```bash
python run.py
```