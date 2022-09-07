from models import Generator
from datasets import denormalize, MeanStd
import torch
from torch.autograd import Variable
import argparse
import os
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--image_path", type=str, default="F:\\data\\4Bands_crop_copy_tifs_jpgs_test\\2020_1_1.jpg", help="Path to image")
parser.add_argument("--checkpoint_model", type=str, default="C:\\SSRGAN\\saved_models\\generator_90.pth", help="Path to checkpoint model")
parser.add_argument("--channels", type=int, default=3, help="Number of image channels")

opt = parser.parse_args()
print(opt)

mean, std = MeanStd("F:/data/4Bands_crop_copy_tifs_jpgs_test/")

os.makedirs("outputs", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define model and load model checkpoint
generator = Generator(channels=opt.channels, scale=5, filters=64,
                              num_res_blocks=8).to(device)
generator.load_state_dict(torch.load(opt.checkpoint_model))
generator.eval()

transform = transforms.Compose([transforms.Resize((20, 20), Image.BICUBIC), transforms.ToTensor(), transforms.Normalize(mean, std)])

# Prepare input
image_tensor = Variable(transform(Image.open(opt.image_path))).to(device).unsqueeze(0)

# Upsample image
with torch.no_grad():
    sr_image = denormalize(generator(image_tensor), mean, std).cpu()

# Save image
fn = opt.image_path.split("\\")[-1]
save_image(sr_image, f"outputs\\sr-{fn}")