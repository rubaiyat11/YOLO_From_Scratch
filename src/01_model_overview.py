from ultralytics import YOLO
from torchvision import transforms
from PIL import Image
import torch
import matplotlib.pyplot as plt


model = YOLO("yolo11n.pt")      #Instantiating yolo v11 model
net = model.model               #Model Stats

# Load real image
image = Image.open("../images/cat.jpg").convert("RGB")

transform = transforms.Compose([   #Reshaping into usable ratio
    transforms.Resize((640, 640)),
    transforms.ToTensor()
])

x = transform(image).unsqueeze(0)

# First convolution
x = net.model[0](x)

print("\nFeature Map Statistics")

for i in range(16):
    fmap = x[0, i]

    print(
        f"Map {i:02d} | "
        f"Min: {fmap.min():.3f} | "
        f"Max: {fmap.max():.3f} | "
        f"Mean: {fmap.mean():.3f}"
    )

print(x.shape)

fig, axes = plt.subplots(4, 4, figsize=(10, 10))

for i, ax in enumerate(axes.flat):
    ax.imshow(x[0, i].detach().numpy(), cmap="gray")
    ax.set_title(f"Map {i}")
    ax.axis("off")

plt.tight_layout()
plt.show()