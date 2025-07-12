# DC-GAN from Scratch in PyTorch

This repository contains a from-scratch implementation of the paper:

> (https://arxiv.org/abs/1511.06434)

## 🏁 Training

```bash
python train.py config.py --gen-checkpoint-path path/to/GEN.pth --disc-checkpoint-path path/to/DISC.pth
```

## 🎨 Inference

```bash
python inference.py config.py --gen-checkpoint-path path/to/GEN.pth --disc-checkpoint-path path/to/DISC.pth
```

## 🖼 Sample Generated Images From CelebA

![Generated Image](gen_images/generated_image0.png)
![Generated Image](gen_images/generated_image15.png)
![Generated Image](gen_images/generated_image17.png)
![Generated Image](gen_images/generated_image18.png)
![Generated Image](gen_images/generated_image19.png)
![Generated Image](gen_images/generated_image26.png)
![Generated Image](gen_images/generated_image27.png)
![Generated Image](gen_images/generated_image31.png)
![Generated Image](gen_images/generated_image32.png)
![Generated Image](gen_images/generated_image33.png)
![Generated Image](gen_images/generated_image24.png)
