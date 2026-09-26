# Autonomous Drone Adaptive Industrial Inspection Pipeline

![Python 3](https://img.shields.io/badge/Language-Python%203.10-blue)
![OpenCV](https://img.shields.io/badge/Library-OpenCV%20CLAHE-green)
![Developer](https://img.shields.io/badge/Developer-Ayoub%20Lahmar-brightgreen)

An adaptive image preprocessing and feature extraction pipeline built for high-speed industrial UAV inspections (e.g., power line transmission towers, solar arrays). Rejects motion-blurred frames dynamically via **2D Laplacian Variance Analysis** and normalizes lighting non-uniformities using **Contrast Limited Adaptive Histogram Equalization (CLAHE)**.

Developed by **Ayoub Lahmar** ([@Redayoub-lang](https://github.com/Redayoub-lang)) for application to **Jiangsu University (JSU)**.

## 📐 Blur Detection Metric

Frame degradation is quantified by taking the spatial variance of the image Laplacian operator \(\nabla^2 I\):
$$S_{blur} = \text{Var}\left( \nabla^2 I \right) = \text{Var}\left( \frac{\partial^2 I}{\partial x^2} + \frac{\partial^2 I}{\partial y^2} \right)$$
Frames evaluated with $S_{blur} < \tau$ are dropped immediately to optimize downstream edge processing workloads.
💻 Build & Run
pip install opencv-python numpy
python inspection_pipeline.py
