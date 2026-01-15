#!/usr/bin/env bash

set -e  # herhangi bir hata olursa script dursun

echo "🔧 MyUnityScene folder is being re-formatted..."

wget https://github.com/emrei1/Lidar-RGB-Reconstruction/releases/download/v1.0-data/MyUnityScene.zip

python reformat_myunityscene.py


echo "✅ Reformatted MyUnityScene"
