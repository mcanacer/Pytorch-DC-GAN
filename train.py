import sys
import importlib

import torch


def get_everything(config_path, args):
    module_path = config_path.replace('/', '.').replace('.py', '')
    module = importlib.import_module(module_path, package=None)
    return module.everything(args)


def main(config_path, args):
    evy = get_everything(config_path, args)

    train_loader = evy['train_loader']

    gen = evy['gen']
    disc = evy['disc']
    gen_optim = evy['gen_optim']
    disc_optim = evy['disc_optim']
    epochs = evy['epochs']
    device = evy['device']

    loss = evy['loss']

    run = evy['run']

    gen_checkpoint_path = evy['gen_checkpoint_path']
    disc_checkpoint_path = evy['disc_checkpoint_path']

    gen.train()
    disc.train()
    for epoch in range(epochs):
        for x, _ in train_loader:
            x = x.to(device)
            noise = torch.randn(x.shape[0], 100, 1, 1).to(device)

            fake = gen(noise)

            disc_real = disc(x).reshape(-1)
            disc_real_loss = loss(disc_real, torch.ones_like(disc_real))
            disc_fake = disc(fake.detach()).reshape(-1)
            disc_fake_loss = loss(disc_fake, torch.zeros_like(disc_fake))
            disc_loss = 0.5 * (disc_real_loss + disc_fake_loss)

            disc_optim.zero_grad()
            disc_loss.backward()
            disc_optim.step()

            output = disc(fake).reshape(-1)
            gen_loss = loss(output, torch.ones_like(output))

            gen_optim.zero_grad()
            gen_loss.backward()
            gen_optim.step()

            run.log({
                "discriminator_loss": disc_loss,
                "generative_loss": gen_loss,
                "total_loss": disc_loss + gen_loss,
                "epoch": epoch})

        torch.save(gen, gen_checkpoint_path)
        torch.save(disc, disc_checkpoint_path)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        raise ValueError('you must provide config file')
    main(sys.argv[1], sys.argv[2:])
