# 1. Download Vicuna's weights to ./models   (it's a delta version)
# 2. Download LLaMA's weight via: https://huggingface.co/huggyllama/llama-13b/tree/main
# 3. merge them and setup config
# 4. Download the mini-gpt4 compoents' pretrained ckpts
# 5. vision part will be automatically download when launching the model


import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
from torchvision.utils import save_image
from minigpt_utils import visual_attacker, prompt_wrapper

from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
torch.set_num_threads(8)


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg_path", default="eval_configs/minigpt4_eval.yaml", help="path to configuration file.")
    parser.add_argument("--gpu_id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--n_iter", type=int, default=500, help="specify the number of iterations for attack.")
    parser.add_argument('--eps', type=int, default=32, help="epsilon of the attack budget")
    parser.add_argument('--alpha', type=int, default=1, help="step_size of the attack")
    parser.add_argument("--constrained", default=False, action='store_true')
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--train_data_file", type=str, help="training_file",
                        default="/data/huangyoucheng/mm-safety/data_prepare/hh_harmless/train_filtered.jsonl")
    parser.add_argument("--image_path", type=str, default="val2017")
    parser.add_argument("--segment_id", type=int, default=0)
    parser.add_argument("--all_segments", type=int, default=4)

    parser.add_argument("--save_dir", type=str, default='output',
                        help="save directory")

    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
             "in xxx=yyy format will be merged into config file (deprecate), "
             "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# ========================================
#             Model Initialization
# ========================================


print('>>> Initializing Models')

args = parse_args()
cfg = Config(args)
setup_seed(args.segment_id)

model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))
model.eval()

vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

print('[Initialization Finished]\n')

if not os.path.exists(args.save_dir):
    os.mkdir(args.save_dir)

import json

data = []
with open(args.train_data_file, 'r') as f:
    for line in f:
        data.append(json.loads(line))

all_image_files = os.listdir(args.image_path)
all_image_files = sorted(all_image_files)
seg_len = len(all_image_files) // args.all_segments
all_image_files = all_image_files[args.segment_id * seg_len:(args.segment_id + 1) * seg_len]
for image_file in all_image_files:
    my_attacker = visual_attacker.PreferenceAttacker(args, model, data, device=model.device, is_rtp=False)
    template_img = os.path.join(args.image_path, image_file)
    img = Image.open(template_img).convert('RGB')
    img = vis_processor(img).unsqueeze(0).to(model.device)

    text_prompt_template = prompt_wrapper.minigpt4_chatbot_prompt
    print('save to %s' % ('%s/%s.bmp' % (args.save_dir, image_file.split('.')[0])))
    if not args.constrained:
        adv_img_prompt = my_attacker.attack_unconstrained(text_prompt_template,
                                                          img=img, batch_size=args.batch_size,
                                                          num_iter=args.n_iter, alpha=args.alpha / 255)
    else:
        adv_img_prompt = my_attacker.attack_constrained(text_prompt_template,
                                                        img=img, batch_size=args.batch_size,
                                                        num_iter=args.n_iter, alpha=args.alpha / 255,
                                                        epsilon=args.eps / 255)
    save_image(adv_img_prompt, '%s/%s.bmp' % (args.save_dir, image_file.split('.')[0]))
print('[Done]')
