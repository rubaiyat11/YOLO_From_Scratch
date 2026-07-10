# YOLO From Scratch

Learning YOLO from first principles, with the goal of understanding not only how to use YOLO, but how it works internally.

This repository documents my journey from YOLO theory to building, training, and integrating object detection into robotics systems.

---

## Objectives

- Understand every tensor flowing through the network
- Read and understand the Ultralytics implementation
- Build a miniature YOLO detector from scratch
- Train YOLO on custom datasets
- Integrate YOLO into ROS2 perception pipelines
- Develop engineering-level intuition for object detection

---

## Learning Roadmap

### Phase 1 — Foundations
- [x] CNN Fundamentals
- [x] Training & Optimization
- [x] YOLO Core Theory

### Phase 2 — YOLO Under the Hood
- [ ] Model Overview
- [ ] Backbone
- [ ] Neck
- [ ] Detection Head
- [ ] Prediction Tensor
- [ ] Post Processing (NMS)
- [ ] Ultralytics Source Code

### Phase 3 — Engineering Practice
- [ ] Build Tiny YOLO
- [ ] Inference Pipeline
- [ ] Custom Training
- [ ] Performance Analysis

### Phase 4 — Robotics Integration
- [ ] Camera Pipeline
- [ ] ROS2 Integration
- [ ] TF2 Coordinate Transformations
- [ ] Real-Time Robot Perception

---

## Repository Structure

```
YOLO_From_Scratch/

├── src/          # Lesson code
├── notes/        # Concepts and explanations
├── outputs/      # Visualizations and experiment outputs
├── README.md
└── requirements.txt
```

---

## Goal

By the end of this repository I should be able to:

- Read and understand a modern YOLO implementation.
- Debug and modify the architecture.
- Train detectors for custom robotics tasks.
- Deploy YOLO as part of an end-to-end robot perception pipeline.