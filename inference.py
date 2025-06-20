import sys
import importlib

import torch
from torchvision.utils import save_image


def get_everything(config_path, args):
    module_path = config_path.replace('/', '.').replace('.py', '')
    module = importlib.import_module(module_path, package=None)
    return module.everything(args)


def generate_samples(model, device, latent_dim=100, num_samples=8):
    model.eval()
    noise = torch.randn(num_samples, latent_dim, 1, 1).to(device)
    with torch.no_grad():
        x_gen = model(noise).cpu()

    # Rescale from [-1, 1] to [0, 1]
    x_gen = (x_gen + 1) / 2

    return x_gen


def main(config_path, args):
    evy = get_everything(config_path, args)

    gen = evy['gen']
    z_dim = evy['z_dim']
    device = evy['device']

    gen_checkpoint_path = evy['gen_checkpoint_path']

    gen.load_state_dict(torch.load(gen_checkpoint_path, map_location=device))

    x_gen = generate_samples(gen, device, z_dim)

    for i in range(x_gen.shape[0]):
        save_image(x_gen[i], f'Generated_image{i}.png')


if __name__ == '__main__':
    if len(sys.argv) == 1:
        raise ValueError('you must provide config file')
    main(sys.argv[1], sys.argv[2:])
