import os.path
from diffusers import UNet2DConditionModel,DDIMInverseScheduler,DDIMScheduler
from utils.pipeline_stable_diffusion_xl import StableDiffusionXLPipeline
import torch
from PIL import Image
import argparse
import csv
import tqdm

#evaluate
from reward_model.eval_pickscore import PickScore
import hpsv2
import ImageReward as RM
from reward_model.aesthetic_scorer import AestheticScorer

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--save_dir", type=str, default="./res")
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--guidance_scale", type=float, default=5.5)
    parser.add_argument("--T", type=int, default=50)

    parser.add_argument("--prompt_type", type=str, default="pick",choices=["pick-a-pic","drawbench"])
    parser.add_argument("--prompt_path", type=str, default="./datasets/test_unique_caption_zh.csv")
    parser.add_argument("--metric", type=str, default="PickScore", choices=['PickScore', 'HPSv2', 'AES', "ImageReward"])

    parser.add_argument("--lora_sclae", type=float, default=0.8)
    parser.add_argument("--lora_path", type=str, default='./ckpt/xlMoreArtFullV1.pREw.safetensors')

    parser.add_argument("--weak_lora_scale", type=float, default=-1.5)
    parser.add_argument("--strong_lora_scale", type=float, default=0.8)

    parser.add_argument("--weak_guidance_scale", type=float, default=1.0)
    parser.add_argument("--strong_guidance_scale", type=float, default=1.0)

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


def load_prompts(prompt_path, prompt_version='pick'):
    if prompt_version == 'pick-a-pic':
        prompts = []
        with open(prompt_path, 'r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                if row[1] == "caption":
                    continue
                prompts.append(row[1])
        prompts = prompts[0:101]
        tmp_prompt_list = []
        for prompt in prompts:
            if prompt != "":
                tmp_prompt_list.append(prompt)
        prompts = tmp_prompt_list
        seed_list = [i for i in range(len(prompts))]
    elif prompt_version == 'drawbench':
        prompts = []
        with open(prompt_path, 'r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                if row[0] == "Prompts":
                    continue
                prompts.append(row[0])

        prompts = prompts[0:200]
        tmp_prompt_list = []
        for prompt in prompts:
            if prompt != "":
                tmp_prompt_list.append(prompt)
        prompts = tmp_prompt_list
        # seed
        seed_list = [i for i in range(len(prompts))]
    else:
        raise  NotImplementedError
    return prompts, seed_list

def load_images(file_path):
    image_list = os.listdir(file_path)
    res = []
    for idx in range(len(image_list)):
        res.append(Image.open(os.path.join(file_path, f'{idx}.png')))
    return res

def gen_images(args):
    print('start evaluate...')
    print(f'benchmark: {args.prompt_type}')

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
    # load dpo lora as strong model
    lora_name = 'user_lora'
    pipe.load_lora_weights(args.lora_path, adapter_name=lora_name)

    # load bench prompts
    prompt_list, seed_list = load_prompts(
        prompt_path=args.prompt_path,
        prompt_version=args.prompt_type)

    for idx in tqdm.tqdm(range(len(prompt_list))):
        prompt, seed = prompt_list[idx], seed_list[idx]
        print(f'generate {idx}: {prompt}')


        # generate image via weak model
        generator = get_generator(seed)
        exp_path = os.path.join(args.save_dir, "weak")
        if not os.path.exists(exp_path):
            os.mkdir(exp_path)
        pipe.disable_lora()
        image_weak = pipe(prompt=args.prompt, height=args.size, width=args.size, guidance_scale=args.guidance_scale,
                          num_inference_steps=args.T, generator=generator).images[0]
        image_weak.save(os.path.join(exp_path, f'{idx}.png'))


        # generate image via strong model
        generator = get_generator(seed)
        exp_path = os.path.join(args.save_dir, "strong")
        if not os.path.exists(exp_path):
            os.mkdir(exp_path)
        pipe.enable_lora()
        pipe.set_adapters(lora_name, adapter_weights=args.lora_sclae)
        image_weak = pipe(prompt=args.prompt, height=args.size, width=args.size, guidance_scale=args.guidance_scale,
                          num_inference_steps=args.T, generator=generator).images[0]
        image_weak.save(os.path.join(exp_path, f'{idx}.png'))


        #generate image via W2SD
        generator = get_generator(args.seed)
        exp_path = os.path.join(args.save_dir, "w2sd")
        pipe.disable_lora()
        image_w2sd = \
        pipe.w2sd_lora(prompt=args.prompt, height=args.size, width=args.size, guidance_scale=args.guidance_scale,
                       denoise_lora_scale=args.lora_sclae,
                       num_inference_steps=args.T, generator=generator,
                       lora_gap_list=[args.strong_lora_scale, args.weak_lora_scale],
                       cfg_gap_list=[args.strong_guidance_scale, args.weak_guidance_scale], lora_name=lora_name).images[0]
        image_w2sd.save(os.path.join(exp_path, f'{idx}.png'))


def cal_score(prompt_list, image_list, metric_version):
    prompt_list = prompt_list[:len(image_list)]
    assert len(prompt_list) == len(image_list)
    total_score = 0
    score_list = []
    if metric_version == 'PickScore':
        reward_model = PickScore()
        for prompt, image in tqdm(zip(prompt_list, image_list)):
            _, score = reward_model.calc_probs(prompt, image)
            total_score += score
            score_list.append(score)

    elif metric_version == 'HPSv2':
        for prompt, image in tqdm(zip(prompt_list, image_list)):
            score = hpsv2.score([image], prompt, hps_version="v2.1")
            print(score)
            total_score += score[0]
            score_list.append(score[0])

    elif metric_version == 'ImageReward':
        reward_model = RM.load("ImageReward-v1.0")
        for prompt, image in tqdm(zip(prompt_list, image_list)):
            score = reward_model.score(prompt, image)
            # print(score)
            total_score += score
            score_list.append(score)

    elif metric_version == 'AES':
        reward_model = AestheticScorer(dtype = torch.float32)
        for prompt, image in tqdm(zip(prompt_list, image_list)):
            score = reward_model(image)
            total_score += score[0]
            score_list.append(score[0])
    else:
        raise NotImplementedError

    return total_score/len(prompt_list), score_list

def evaluate(args):
    print('start evaluate...')
    print(f'benchmark: {args.prompt_type}\tmetric: {args.metric}')

    prompt_list, _ = load_prompts(
        prompt_path=args.prompt_path,
        prompt_version=args.prompt_type)

    exp_setting_dirs = ['weak', 'strong', 'w2sd']
    for idx, setting in enumerate(exp_setting_dirs):
        image_list = load_images(os.path.join(args.save_dir,setting))
        assert len(image_list) == len(prompt_list)

        score, score_list = cal_score(prompt_list, image_list, args.metric)
        print(f'exp_setting:{setting}\tscore{score}')
        print('--------------------------')


if __name__ == '__main__':
    args = get_args()

    # first, generate images to be evaluated
    gen_images(args)

    # Second, just evaluate the results.
    evaluate(args)