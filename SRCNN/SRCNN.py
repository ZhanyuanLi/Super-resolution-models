import argparse
import sys
from models import *
from utils import *
import torch
from tensorboardX import SummaryWriter


def main():
    # Create a file in the current path to store the model and results
    os.makedirs("images_train", exist_ok=True)
    os.makedirs("images_valid", exist_ok=True)
    os.makedirs("saved_models", exist_ok=True)

    # Add a command line parser to manage the parameters used by the model
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--train_dataset_name", type=str, default="4Bands_crop_copy_tifs_jpgs_train",
                        help="name of the train dataset")
    parser.add_argument("--valid_dataset_name", type=str, default="4Bands_crop_copy_tifs_jpgs_valid",
                        help="name of the validation dataset")
    parser.add_argument("--test_dataset_name", type=str, default="4Bands_crop_copy_tifs_jpgs_test",
                        help="name of the test dataset")
    parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--hr_height", type=int, default=100, help="high res. image height")
    parser.add_argument("--hr_width", type=int, default=100, help="high res. image width")
    parser.add_argument("--scale", type=int, default=5, help="Image magnification")  # 2 5 10
    parser.add_argument("--summary_writer_name", type=str, default="SRCNN_Loss_PSNR_SSIM",
                        help="name of the summary writer file")
    parser.add_argument("--channels", type=int, default=3, help="number of image channels")
    parser.add_argument("--sample_interval", type=int, default=100, help="interval between saving image samples")
    parser.add_argument("--checkpoint_interval", type=int, default=10, help="interval between model checkpoints")
    # Obtain parameters
    opt = parser.parse_args()
    print(opt)

    # Load the model and loss function onto the GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Size of HD images
    hr_shape = (opt.hr_height, opt.hr_width)

    # Initialize the model
    model = ModelSRCNN(num_channels=opt.channels).to(device)

    # Set the loss function
    criterion_mse = torch.nn.MSELoss().to(device)

    # From the second cycle, load the trained model.
    if opt.epoch != 0:
        model.load_state_dict(torch.load("saved_models/model_%d.pth" % opt.epoch))

    # Set optimizer
    optimizer_model = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    # Tensor type
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

    # Create data loaders that can be iterated
    train_mean, train_std = MeanStd("F:/data/%s/" % opt.train_dataset_name)
    dataloader = DataLoader(
        ImageDataset("F:/data/%s" % opt.train_dataset_name, hr_shape=hr_shape, scale=opt.scale,
                     mean=train_mean, std=train_std),
        batch_size=opt.batch_size,
        shuffle=True,  # Reorder the data at the beginning of each epoch
        num_workers=opt.n_cpu,
    )

    # ----------
    #  Training
    # ----------
    # Create a writer instance
    summary_writer = SummaryWriter("%s" % opt.summary_writer_name)

    for epoch in range(opt.epoch, opt.n_epochs):
        # Calculate the average loss per epoch
        losses = AverageMeter()
        for i, imgs in enumerate(dataloader):

            # Define low and high resolution image pairs
            imgs_lr = Variable(imgs["lr"].type(Tensor))
            imgs_hr = Variable(imgs["hr"].type(Tensor))

            # ------------------
            #  Train Model
            # ------------------

            optimizer_model.zero_grad()

            # Generate high-resolution images from low-resolution images using the model.
            imgs_sr = model(imgs_lr)

            # Loss
            loss_pixel = criterion_mse(imgs_sr, imgs_hr)
            loss_model = loss_pixel

            # Update the current loss of the entire epoch
            losses.update(loss_model.item(), opt.hr_height)

            loss_model.backward()

            optimizer_model.step()
            # --------------
            #  Log Progress
            # --------------

            sys.stdout.write(
                "[Epoch %d/%d] [Batch %d/%d] [Model loss: %f]\n"
                % (epoch, opt.n_epochs, i, len(dataloader), loss_model.item())
            )
            batches_done = epoch * len(dataloader) + i + 1
            if batches_done % opt.sample_interval == 0:
                # Splice 3 tensors together
                img_grid = denormalize(torch.cat((imgs_lr, imgs_sr, imgs_hr), -1), train_mean, train_std)
                save_image(img_grid, "images_train/%d_%s.tif" % (batches_done, imgs['path'][0].rsplit("\\")[-1]),
                           nrow=1, normalize=False)

        # The mean value of the loss of this epoch
        train_loss = losses.avg
        print("**********[Epoch %d/%d]**********\n[Train Loss: %f]\n"
              % (epoch, opt.n_epochs, train_loss))

        # Update PSNR
        valid_loss, valid_psnr, valid_ssim = validationOrTest(opt.valid_dataset_name, hr_shape, opt.scale, model,
                                                              criterion_mse, epoch)
        print("[Validation Loss: %f] [Validation PSNR: %f] [Validation SSIM: %f]\n"
              % (valid_loss, valid_psnr, valid_ssim))
        # Save loss and PSNR
        summary_writer.add_scalars('Loss', {'Train': train_loss, 'Validation': valid_loss}, epoch + 1)
        summary_writer.add_scalar('PSNR', valid_psnr, epoch + 1)
        summary_writer.add_scalar('SSIM', valid_ssim, epoch + 1)
        # F:\SRCNN>tensorboard --logdir ./SRCNN_Loss_PSNR_SSIM --port=6006

        if opt.checkpoint_interval != -1 and (epoch + 1) % opt.checkpoint_interval == 0:
            # Save model checkpoints
            torch.save(model.state_dict(), "saved_models/model_%d.pth" % (epoch + 1))


if __name__ == '__main__':
    main()
