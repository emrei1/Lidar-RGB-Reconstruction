# LiDAR-RGB Reconstruction

---

## 🚀 Getting Started

### Install dependencies
First, install the required Python dependencies by running:

```bash
./setup_env.sh
```

### Prepare sample reconstruction data
To reconstruct and re-format the sample initial reconstruction used in this project, run:

```bash
./reformat_MyUnityScene.sh
```

## 🖥️ Command line prompt

```bash
python train.py -s MyUnityScene --hist_far 20 --num_hist_bins 128 --transi_only_until 100 --max_gaussians 2000000 --transi_weight 0.01 --cull_over 15 --cull_over_transi_only 5 --cull_every 5 --loss_curv_w 0.05
```

## Mesh comparison

<p align="center"> <img src="ground_truth_scene_recording.gif" width="912" height="2048" /> <img src="camera_view_reconstruction.gif" width="912" height="2048" /> /> &nbsp;&nbsp;&nbsp; <img src="trimmed_reconstructed_mesh.gif" width="912" height="2048" /> </p> 


