from ultralytics import YOLO
import torch

print("Pytorch:", torch.__version__)

model = YOLO("yolo11n.pt")

print(model)

print(type(model))

print(model.model)

