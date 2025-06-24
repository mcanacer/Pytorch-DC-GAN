DC-GAN from Scratch in PyTorch

Training

python train.py config.py --gen-checkpoint-path path/to/GEN.pth --disc-checkpoint-path path/to/DISC.pth

Inference

python inference.py config.py --gen-checkpoint-path path/to/GEN.pth --disc-checkpoint-path path/to/DISC.pth
