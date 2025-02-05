# Weak-to-Strong Diffusion With Reflection (W2SD)

## Overview

This guide offers detailed instructions on utilizing W2Sd, an innovative sampling framework that employs a weak-to-strong model pairing strategy to refine variable latent representations, bringing them closer to the ground truth.

Here we provide the inference code which supports ***Weight Difference***. More architectures will be released later.

## Results

<div align="center">
  <img src="./res/framework.jpg" width=70%/>
</div>
**W2SD leverages the gap between weak and strong models to approximate the gap between strong and ideal models.**

<div align="center">
  <img src="./res/res.jpg" width=90%/>
</div>
**Qualitative comparisons with weak model (left), strong model (middle) and W2SD (right). Our method utilizes the differences between chosen strong and weak models (e.g., high-detail LoRA vs. standard model) to deliver improvements in various dimensions, including style, character, clothing, and beyond.**

## Requirements

- `python version == 3.10.14`
- `pytorch with cuda version`
- `diffusers == 0.22.3`
- `PIL`
- `numpy`
- `timm`
- `tqdm`
- `argparse`
- `einops`

## Installation🚀️

Make sure you have successfully built `python` environment and installed `pytorch` with cuda version. Before running the script, ensure you have all the required packages installed. Specifically, you need to install the `diffusers` library with a specific version.

```bash
pip install diffusers == 0.22.3
```

## Usage👀️ 

To use the W2SD, you need to run the `w2sd_lora.py` with appropriate command-line arguments. Below are the available options:

### Command-Line Arguments

- `--weak_lora_scale`: The LoRA scale used by weak model. Default is -1.5.
- `--strong_lora_scale`: The LoRA scale used by strong model. Default is 1.5.
- `--weak_guidance_scale`: The classifier free guidance scale used by weak model. Default is 1.0.
- `--strong_guidance_scale`: The classifier free guidance scale used by weak model. Default is 5.5.
- `--size`: The size (height and width) of the generated image.  Default is 1024.
- `--T`: Number of inference timesteps. Default is 50.
- `--seed`: Random seed to determine the initial latent.
- `--lora_path`: The path of dpo lora ckpt.
- `--prompt`: Condition text prompt for generation.
- `--save_dir`: Path to save the generated images. Default is `./res`.

### Running the Script

You can execute the inference step by running the following command:
```bash
python ./w2sd_lora.py
```
This command will execute the generation process, generating images corresponding to the `predefined prompts` under both `weak model` and `strong model` and `w2sd`.

You can also modify these prompts in `infer.py` to the content you want.


### Output🎉️ 

The composed images will be saved to the directory designated by `--save_dir`.

## Pre-trained Weights Download❤️

You need to manually or automatically download the SDXL model via Hugging Face. Please ensure a stable internet connection and the correct version.

**Note**: The LoRA DPO checkpoint can be downloaded from CivitAI at the following link: https://civitai.com/models/124347/xlmoreart-full-xlreal-enhancer. Please place the downloaded file in the `--lora_path`


If you encounter any issues with deployment and operation, you are kindly invited to contact us, thank you very much!
