<p align="center">

  <h1 align="center">NeRF-VO: Real-Time Sparse Visual Odometry With Neural Radiance Fields</h1>
  <p align="center">
    <a href="https://www.linkedin.com/in/jens-naumann/" target="_blank"><strong>Jens Naumann</strong></a>
    ·
    <a href="https://binbin-xu.github.io/" target="_blank"><strong>Binbin Xu</strong></a>
    ·
    <a href="https://scholar.google.ch/citations?user=SmGQ48gAAAAJ&hl=de" target="_blank"><strong>Stefan Leutenegger</strong></a>
    ·
    <a href="https://xingxingzuo.github.io/" target="_blank"><strong>Xingxing Zuo</strong></a>
</p>

  <h2 align="center">IEEE Robotics and Automation Letters 2024</h2>
  <h3 align="center"><a href="https://ieeexplore.ieee.org/document/10578010" target="_blank">Paper</a> | <a href="https://youtu.be/El3-hSnuOz0?si=bGjMPECjWvTAlCLg"  target="_blank">Video</a> | <a href="" target="_blank">Project Page</a></h3>
  <div align="center"></div>
</p>

<!-- TABLE OF CONTENTS -->
<details open="open" style='padding: 10px; border-radius:5px 30px 30px 5px; border-style: solid; border-width: 1px;'>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#installation">Installation</a>
    </li>
    <li>
      <a href="#datasets">Datasets</a>
    </li>
    <li>
      <a href="#execution">Execution</a>
    </li>
    <li>
      <a href="#citation">Citation</a>
    </li>
    <li>
      <a href="#contact">Contact</a>
    </li>
  </ol>
</details>

## Installation

### Clone repository
```bash
git clone https://github.com/jens-nau/NeRF-VO.git
git submodule update --init --recursive
```

### Create conda environment
```bash
conda create -n nerf_vo python=3.8
```

### PyTorch and PyTorch Scatter
```bash
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
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

## Citation
If you find our code or paper useful, please cite:
```bibtex
@article{naumann2024nerfvo,
  author={Naumann, Jens and Xu, Binbin and Leutenegger, Stefan and Zuo, Xingxing},
  journal={IEEE Robotics and Automation Letters}, 
  title={NeRF-VO: Real-Time Sparse Visual Odometry With Neural Radiance Fields}, 
  year={2024},
}
```

## Contact
Contact [Jens Naumann](mailto:jens.naumann@tum.de) and [Xingxing Zuo](mailto:xingxing.zuo@tum.de) for questions, comments and bug reports.
