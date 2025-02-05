import os.path
from diffusers import UNet2DConditionModel,DDIMInverseScheduler,DDIMScheduler
from utils.pipeline_stable_diffusion_xl import StableDiffusionXLPipeline
import torch
from PIL import Image
import argparse


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--save_dir", type=str, default="./res")
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--guidance_scale", type=float, default=5.5)
    parser.add_argument("--T", type=int, default=50)
    parser.add_argument("--prompt", type=str, default="A Lego-style man is in the car.")
    parser.add_argument("--seed", type=int, default=5555)

    parser.add_argument("--lora_sclae", type=float, default=0.8)
    parser.add_argument("--lora_path", type=str, default='./ckpt/xlMoreArtFullV1.pREw.safetensors')

    parser.add_argument("--weak_lora_scale", type=float, default=-1.5)
    parser.add_argument("--strong_lora_scale", type=float, default=1.5)

    parser.add_argument("--weak_guidance_scale", type=float, default=1.0)
    parser.add_argument("--strong_guidance_scale", type=float, default=5.5)

    args = parser.parse_args()
    return args


def get_generator(random_seed):
    torch.manual_seed(int(random_seed))
    torch.cuda.manual_seed(int(random_seed))
    generator = torch.manual_seed(random_seed)
    return generator

def compose_img(img_list):
    new_width = img_list[0].width * len(img_list)
    new_image = Image.new("RGB", (new_width, img_list[0].height))
    for idx, img in enumerate(img_list):
        new_image.paste(img, (img.width * idx, 0))
    return new_image

def create_dirs(base_dir):
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)



if __name__ == '__main__':
    args = get_args()
    create_dirs(args.save_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load pipeline
    model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    dtype = torch.float16
    pipe = StableDiffusionXLPipeline.from_pretrained(model_id, torch_dtype=dtype,
                                                     variant='fp16',
                                                     safety_checker=None, requires_safety_checker=False).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.inv_scheduler = DDIMInverseScheduler.from_pretrained('stabilityai/stable-diffusion-xl-base-1.0',
                                                             subfolder='scheduler')
    #load dpo lora as strong model
    lora_name = 'user_lora'
    pipe.load_lora_weights(args.lora_path, adapter_name=lora_name)


    #generate image with sdxl (weak model)
    generator = get_generator(args.seed)
    pipe.disable_lora()
    image_sdxl = pipe(prompt=args.prompt, height=args.size, width=args.size, guidance_scale=args.guidance_scale,
                             num_inference_steps=args.T, generator=generator).images[0]

    # generate image with dpo lora (strong model)
    generator = get_generator(args.seed)
    pipe.enable_lora()
    pipe.set_adapters(lora_name, adapter_weights=args.lora_sclae)
    image_dpo_lora = pipe(prompt=args.prompt, height=args.size, width=args.size, guidance_scale=args.guidance_scale,
                            num_inference_steps=args.T, generator=generator).images[0]


    # generate image with w2sd (ours)
    generator = get_generator(args.seed)
    pipe.disable_lora()
    image_w2sd = pipe.w2sd_lora(prompt=args.prompt, height=args.size, width=args.size, guidance_scale=args.guidance_scale,
                                        denoise_lora_scale=args.lora_sclae,
                                        num_inference_steps=args.T, generator=generator,
                                        lora_gap_list=[args.strong_lora_scale,args.weak_lora_scale],
                                        cfg_gap_list=[args.strong_guidance_scale,args.weak_guidance_scale],lora_name=lora_name).images[0]

    res_img = compose_img([image_sdxl,image_dpo_lora,image_w2sd])
    res_img.save(os.path.join(args.save_dir, f'seed_{args.seed}.png'))
