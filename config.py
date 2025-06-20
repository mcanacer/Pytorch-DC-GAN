import argparse
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import wandb
from model import Generator, Discriminator


def parse_args(args):
    parser = argparse.ArgumentParser()

    # Dataset
    parser.add_argument('--image-size', type=int, default=64)
    parser.add_argument('--num-channels', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-workers', type=int, default=2)

    # Model
    parser.add_argument('--z-dim', type=int, default=100)
    parser.add_argument('--nfd', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--num-epochs', type=int, default=100)

    # Wandb
    parser.add_argument('--projet', type=str, required=True)
    parser.add_argument('--name', type=str, required=True)

    # Save
    parser.add_argument('--gen-checkpoint-path', type=str, required=True)
    parser.add_argument('--disc-checkpoint-path', type=str, required=True)

    return parser.parse_args(args)


def everything(args):
    args = parse_args(args)

    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),  # Normalize [0, 1]
        transforms.Lambda(lambda t: (t * 2) - 1)  # Scale [-1, 1]
    ])

    train_dataset = datasets.STL10(
        root='./data',
        split='unlabeled',
        download=True,
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gen = Generator(z_dim=args.z_dim, nfd=args.nfd, nc=args.num_channels).to(device).to(device)
    disc = Discriminator(nfd=args.nfd, nc=args.num_channels).to(device)

    loss = torch.nn.BCELoss()

    gen_optim = torch.optim.Adam(gen.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
    disc_optim = torch.optim.Adam(disc.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
    epochs = args.num_epochs

    run = wandb.init(
        project=args.project,
        name=args.name,
        reinit=True,
        config=vars(args)
    )

    return {
        'train_loader': train_loader,
        'gen': gen,
        'disc': disc,
        'loss': loss,
        'gen_optim': gen_optim,
        'disc_optim': disc_optim,
        'z_dim': args.z_dim,
        'epochs': epochs,
        'device': device,
        'run': run,
        'gen_checkpoint_path': args.gen_checkpoint_path,
        'disc_checkpoint_path': args.disc_checkpoint_path,
    }
