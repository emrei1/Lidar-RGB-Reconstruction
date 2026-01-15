#!/usr/bin/env bash

set -e  # herhangi bir hata olursa script dursun

echo "🔧 Python packages are installed..."

# Python 3 için pip kullan (en güvenlisi)
pip install --upgrade pip
pip install opencv-python
pip install scikit-image
pip install POT
pip install ninja
pip install --ignore-install open3d
pip install pymeshlab
pip install plyfile
pip install git+https://github.com/facebookresearch/pytorch3d.git --no-build-isolation
git clone https://github.com/emrei1/Lidar-RGB-Reconstruction.git
cd Lidar-RGB-Reconstruction
cd submodules
cd diff-gaussian-rasterization
python setup.py install && pip install . --no-build-isolation
cd ..
cd simple-knn
pip install . --no-build-isolation
cd ..
cd ..


echo "✅ Completed installations"
