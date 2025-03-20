# MSF-MHDGCN-Pose
This repository contains the implementation of pseudo code for 6D object pose estimation. The model leverages multi-scale feature extraction, cross-modal attention, and multi-hedron dynamic graph convolution to improve robustness and accuracy.

Features
RGB + Point Cloud Fusion: Uses a modified ResNet for RGB feature extraction and MHDGCN (Multi-Head Dynamic Graph Convolution Network) for point cloud processing.
Multi-Scale Feature Extraction: Integrates MFP (Multi-Feature Pooling) to extract hierarchical information.
Cross-Modal Attention: Enhances feature fusion between RGB and point cloud.
Pose Refinement: Uses PoseRefineNet to further optimize the predicted pose.
The above implementation can be queried in the network_pseudo.py file

In addition, the dataset processing and training process can be accessed in dataset_pseudo.py and train_pseudo.py
