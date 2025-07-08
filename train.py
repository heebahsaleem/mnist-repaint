from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
from torchvision import datasets, transforms

# U-Net Model
model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4),
    channels = 1  # MNIST is grayscale
)

# Gaussian Diffusion
diffusion = GaussianDiffusion(
    model,
    image_size = 28,
    timesteps = 1000,   # number of steps
    sampling_timesteps = 250,
    loss_type = 'l2'    # default loss
)

# MNIST Dataset
transform = transforms.Compose([
    transforms.ToTensor()
])
dataset = datasets.MNIST(root='.', train=True, transform=transform, download=True)

# Trainer
trainer = Trainer(
    diffusion,
    dataset,
    train_batch_size = 64,
    train_lr = 8e-5,
    train_num_steps = 70000,
    gradient_accumulate_every = 2,
    ema_decay = 0.995,
    save_and_sample_every = 1000,
    results_folder = './results_mnist'
)

trainer.train()
