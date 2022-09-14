from models import ModelFADSR
from datasets import denormalize, MeanStd
import torch
from torch.autograd import Variable
import argparse
import os
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import torch.nn as nn

parser = argparse.ArgumentParser()
parser.add_argument("--image_path", type=str, default="F:\\data\\4Bands_crop_copy_tifs_jpgs_test\\2021_11_5.jpg", help="Path to image")
parser.add_argument("--checkpoint_model", type=str, default="G:\\FADSR\\saved_models\\model_200.pth", help="Path to checkpoint model")
parser.add_argument("--channels", type=int, default=3, help="Number of image channels")
parser.add_argument("--scale", type=int, default=5, help="Image magnification")

opt = parser.parse_args()
print(opt)

mean, std = MeanStd("F:/data/4Bands_crop_copy_tifs_jpgs_test/")

os.makedirs("outputs", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define model and load model checkpoint
model = ModelFADSR(scale=opt.scale).to(device)
model.load_state_dict(torch.load(opt.checkpoint_model))
model.eval()

transform = transforms.Compose([transforms.Resize((20, 20), Image.BICUBIC), transforms.ToTensor(), transforms.Normalize(mean, std)])

# Prepare input
image_tensor = Variable(transform(Image.open(opt.image_path))).to(device).unsqueeze(0)

# Upsample image
with torch.no_grad():
    sr_image = denormalize(model(image_tensor), mean, std).cpu()

# Save SR image
fn = opt.image_path.split("\\")[-1]
save_image(sr_image, f"outputs\\sr-{fn}")

# Save nearest image
nearest = nn.functional.interpolate(image_tensor, scale_factor=opt.scale)
nearest_01 = denormalize(nearest, mean, std)
save_image(nearest_01, f"outputs\\nearest-{fn}")

# Save LR image
LR_01 = denormalize(image_tensor, mean, std)
save_image(LR_01, f"outputs\\lr-{fn}")
