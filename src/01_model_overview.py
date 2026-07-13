from ultralytics import YOLO
import torch
import matplotlib.pyplot as plt

model = YOLO("yolo11n.pt")
net = model.model

# Fake image
x = torch.randn(1, 3, 640, 640)

# First convolution
x = net.model[0](x)

print(x.shape)

# Show first feature map
plt.imshow(x[0, 0].detach().numpy(), cmap="gray")
plt.title("Feature Map 0")
plt.axis("off")
plt.show()