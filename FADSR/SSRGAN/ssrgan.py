import argparse
from models import *
from utils import *
import torch.nn as nn
import torch
from tensorboardX import SummaryWriter


def main():
    os.makedirs("images_train", exist_ok=True)
    os.makedirs("images_valid", exist_ok=True)
    os.makedirs("saved_models", exist_ok=True)

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
    parser.add_argument("--n_cpu", type=int, default=0,
                        help="number of cpu threads to use during batch generation")
    parser.add_argument("--hr_height", type=int, default=100, help="high res. image height")
    parser.add_argument("--hr_width", type=int, default=100, help="high res. image width")
    parser.add_argument("--scale", type=int, default=5, help="Image magnification")
    parser.add_argument("--summary_writer_name", type=str, default="SSRGAN_Loss_PSNR_SSIM",
                        help="name of the summary writer file")
    parser.add_argument("--channels", type=int, default=3, help="number of image channels")
    parser.add_argument("--sample_interval", type=int, default=100, help="interval between saving image samples")
    parser.add_argument("--checkpoint_interval", type=int, default=10,
                        help="batch interval between model checkpoints")
    parser.add_argument("--residual_blocks", type=int, default=8, help="number of residual blocks in the generator")
    parser.add_argument("--warmup_batches", type=int, default=0,
                        help="number of batches with pixel-wise loss only")
    parser.add_argument("--lambda_adv", type=float, default=1e-5, help="adversarial loss weight")
    parser.add_argument("--lambda_pixel", type=float, default=1, help="pixel-wise loss weight")
    parser.add_argument("--lambda_content", type=float, default=1e-3, help="Weight of the content loss")
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hr_shape = (opt.hr_height, opt.hr_width)

    # Initialize generator and discriminator
    generator = Generator(channels=opt.channels, scale=opt.scale,
                          n_resblocks=opt.residual_blocks, filters=64).to(device)
    discriminator = Discriminator(imgs_shape=(opt.channels, *hr_shape)).to(device)
    feature_extractor = FeatureExtractor().to(device)

    # Losses
    criterion_GAN = torch.nn.BCEWithLogitsLoss().to(device)  # Adversarial loss
    criterion_content = torch.nn.L1Loss().to(device)  # Content loss
    criterion_pixel = torch.nn.L1Loss().to(device)  # Pixel loss

    # From the second cycle, load the trained model.
    if opt.epoch != 0:
        generator.load_state_dict(torch.load("saved_models/generator_%d.pth" % opt.epoch))
        discriminator.load_state_dict(torch.load("saved_models/discriminator_%d.pth" % opt.epoch))

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

    # Create data loaders that can be iterated
    train_mean, train_std = MeanStd("F:/data/%s/" % opt.train_dataset_name)
    dataloader = DataLoader(
        ImageDataset("F:/data/%s" % opt.train_dataset_name, hr_shape=hr_shape, scale=opt.scale,
                     mean=train_mean, std=train_std),
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
    )

    # ----------
    #  Training
    # ----------
    # Create a writer instance
    summary_writer = SummaryWriter("%s" % opt.summary_writer_name)

    for epoch in range(opt.epoch, opt.n_epochs):
        # Calculate the average loss per epoch
        losses_G = AverageMeter()
        losses_D = AverageMeter()
        for i, imgs in enumerate(dataloader):

            batches_done = epoch * len(dataloader) + i + 1

            # Define low and high resolution image pairs
            imgs_lr = Variable(imgs["lr"].type(Tensor))
            imgs_hr = Variable(imgs["hr"].type(Tensor))

            # Adversarial ground truths
            valid = Variable(Tensor(np.ones((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)
            fake = Variable(Tensor(np.zeros((imgs_lr.size(0), *discriminator.output_shape))), requires_grad=False)

            # ------------------
            #  Train Generators
            # ------------------
            optimizer_G.zero_grad()

            # Generate a high resolution image from low resolution input
            gen_hr = generator(imgs_lr)

            # Measure pixel-wise loss against ground truth
            loss_pixel = criterion_pixel(gen_hr, imgs_hr)

            if batches_done <= opt.warmup_batches:
                # Warm-up (pixel-wise loss only)
                loss_pixel.backward()
                optimizer_G.step()
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [G pixel: %f]"
                    % (epoch, opt.n_epochs, i, len(dataloader), loss_pixel.item())
                )
                continue

            # Extract validity predictions from discriminator
            pred_real = discriminator(imgs_hr).detach()
            pred_fake = discriminator(gen_hr)

            # Calculate adversarial losses
            loss_GAN_fr = criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), valid)
            loss_GAN_rf = criterion_GAN(pred_real - pred_fake.mean(0, keepdim=True), fake)
            loss_GAN = (loss_GAN_rf + loss_GAN_fr) / 2

            # Content loss
            gen_features = feature_extractor(gen_hr)
            real_features = feature_extractor(imgs_hr).detach()
            loss_content = criterion_content(gen_features, real_features)

            # Total generator loss
            loss_G = opt.lambda_content * loss_content + opt.lambda_adv * loss_GAN + opt.lambda_pixel * loss_pixel

            loss_G.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()

            pred_real = discriminator(imgs_hr)
            pred_fake = discriminator(gen_hr.detach())

            # Adversarial loss for real and fake images (relativistic average GAN)
            loss_real = criterion_GAN(pred_real - pred_fake.mean(0, keepdim=True), valid)
            loss_fake = criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), fake)

            # Total loss
            loss_D = (loss_real + loss_fake) / 2

            loss_D.backward()
            optimizer_D.step()

            # --------------
            #  Log Progress
            # --------------

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, content: %f, adv: %f, pixel: %f]"
                % (
                    epoch,
                    opt.n_epochs,
                    i,
                    len(dataloader),
                    loss_D.item(),
                    loss_G.item(),
                    loss_content.item(),
                    loss_GAN.item(),
                    loss_pixel.item(),
                )
            )

            if batches_done % opt.sample_interval == 0:
                # Save the upsampled image, the model output SR image and the original HR image
                imgs_lr = nn.functional.interpolate(imgs_lr, scale_factor=opt.scale)
                img_grid = denormalize(torch.cat((imgs_lr, gen_hr, imgs_hr), -1), train_mean, train_std)
                save_image(img_grid, "images_train/%d_%s.tif" % (batches_done, imgs['path'][0].rsplit("\\")[-1]),
                           nrow=1, normalize=False)

            # The mean value of the loss of this epoch
            losses_G.update(loss_G.item(), opt.hr_height)
            losses_D.update(loss_D.item(), opt.hr_height)

        # This epoch's loss
        train_loss_G = losses_G.avg
        train_loss_D = losses_D.avg
        print("**********[Epoch %d/%d]**********\n[Train Loss G: %f] [Train Loss D: %f]\n"
              % (epoch, opt.n_epochs, train_loss_G, train_loss_D))

        # Update PSNR
        valid_loss_G, valid_loss_D, valid_psnr, valid_ssim = validationOrTest(opt.valid_dataset_name, hr_shape,
                                                                              opt.scale, generator, discriminator,
                                                                              opt.lambda_content, criterion_content,
                                                                              opt.lambda_pixel, criterion_pixel,
                                                                              opt.lambda_adv, criterion_GAN,
                                                                              feature_extractor, epoch)

        print("[Validation Loss G: %f] [Validation Loss D: %f] [Validation PSNR: %f] [Validation SSIM: %f]\n"
              % (valid_loss_G, valid_loss_D, valid_psnr, valid_ssim))

        # Save loss function and PSNR
        summary_writer.add_scalars('Loss_G', {'Train': train_loss_G, 'Validation': valid_loss_G}, epoch + 1)
        summary_writer.add_scalars('Loss_D', {'Train': train_loss_D, 'Validation': valid_loss_D}, epoch + 1)
        summary_writer.add_scalar('PSNR', valid_psnr, epoch + 1)
        summary_writer.add_scalar('SSIM', valid_ssim, epoch + 1)
        # F:\SSRGAN>tensorboard --logdir ./SSRGAN_Loss_PSNR_SSIM --port=6006

        # if batches_done % opt.checkpoint_interval == 0:
        if opt.checkpoint_interval != -1 and (epoch + 1) % opt.checkpoint_interval == 0:
            # Save model checkpoints
            torch.save(generator.state_dict(), "saved_models/generator_%d.pth" % (epoch + 1))
            torch.save(discriminator.state_dict(), "saved_models/discriminator_%d.pth" % (epoch + 1))


if __name__ == '__main__':
    main()

