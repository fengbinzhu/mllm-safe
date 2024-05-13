import argparse
import os
import random

import pandas as pd
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import gradio as gr
from PIL import Image
import json

from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *

from minigpt_utils import prompt_wrapper, generator, visual_attacker


def rtp_read(text_file):
    dataset = []
    lines = open(text_file).read().split("\n")
    for li in lines:
        obj = json.loads(li)
        if obj['challenging']:
            dataset.append(obj['prompt']['text'])
    random.seed(0)
    random.shuffle(dataset)
    dataset = dataset[:50]
    return dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", default="eval_configs/minigpt4_eval.yaml", help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")

    parser.add_argument("--mode", type=str, default='VisualChatBot',
                        choices=[ "TextOnly", "VisualChatBot"],
                        help="Inference Mode: TextOnly: Text model only (Vicuna) \n VisualChatBot: Vision model + Text model (MiniGPT4) ")


    parser.add_argument("--image_file", type=str, default='./image.bmp',
                        help="Image file")
    parser.add_argument("--output_file", type=str, default='./result.jsonl',
                        help="Output file.")
    parser.add_argument("--test_data_file", type=str, default='/data/huangyoucheng/mm-safety/data_prepare/do-not-answer/data_en.csv',)

    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    parser.add_argument("--ignore_img", action="store_true")
    parser.add_argument("--sample_num", type=int, default=50)
    parser.add_argument("--random_attack", action="store_true")
    args = parser.parse_args()
    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


# ========================================
#             Model Initialization
# ========================================

print('>>> Initializing Models')

args = parse_args()
cfg = Config(args)

model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))

vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
print('Initialization Finished')

my_generator = generator.Generator(model=model)



# ========================================
#             Inference
# ========================================


##  TODO: expose interface.
df = pd.read_csv(args.test_data_file)
random.seed(0)
harmful_types = set(df["types_of_harm"])
harmful_types.remove('Treat Chatbot as a Human')
harmful_types = sorted(harmful_types)
out = []
for harmful_type in harmful_types:
    print(f"========== {harmful_type} ==========")
    data = df[df["types_of_harm"] == harmful_type]['question'].tolist()
    # 每个类别选5条：
    random.shuffle(data)
    data = data[:args.sample_num]


    if args.mode == "TextOnly":

        ## vicuna official system message.
        #prefix = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. "

        text_formats = ""
        text_formats += "###Human:%s ###Assistant:"
        img_prompt = []

    elif args.mode == "VisualChatBot":
        text_formats = prompt_wrapper.minigpt4_chatbot_prompt
        img = Image.open(args.image_file).convert('RGB')
        img_prompt = vis_processor(img).unsqueeze(0).to(model.device)
        if args.random_attack:
            epsilon = 16 / 255
            adv_noise = torch.rand_like(img_prompt).to(model.device) * 2 * epsilon - epsilon
            img_prompt = visual_attacker.denormalize(img_prompt).clone().to(model.device)
            adv_noise.data = (adv_noise.data + img_prompt.data).clamp(0, 1) - img_prompt.data
            img_prompt = img_prompt + adv_noise
            img_prompt = visual_attacker.normalize(img_prompt)
        img_prompt = [img_prompt]


    print("Instructions: ")
    print(text_formats)


    prompt = prompt_wrapper.PreferencePrompt(model=model, img_prompts=[img_prompt])

    with torch.no_grad():
        for i, user_message in enumerate(data):
            print(f" ----- {i} ----")
            print(" -- prompt: ---")
            if args.ignore_img:
                user_message = "Ignore the image and generate responses. " + user_message
            print(text_formats % user_message)

            prompt.update_text_prompt(
                [text_formats], [user_message]
            )
            response, _ = my_generator.generate(prompt)

            print(" -- responses: ---")
            print(response)
            out.append({'harm_type': harmful_type, 'prompt': user_message, 'response': response})
            print()


with open(args.output_file, 'w') as f:
    f.write(json.dumps({
        "args": vars(args),
        "prompt": text_formats
    }))
    f.write("\n")

    for li in out:
        f.write(json.dumps(li))
        f.write("\n")
