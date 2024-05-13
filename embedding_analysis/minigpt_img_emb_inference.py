import argparse
import os
import random
from tqdm import tqdm
import pickle

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image

from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry

from minigpt_utils import prompt_wrapper
torch.set_num_threads(8)


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg_path", default="eval_configs/minigpt4_eval.yaml", help="path to configuration file.")
    parser.add_argument("--gpu_id", type=int, default=0, help="specify the gpu to load the model.")

    parser.add_argument("--mode", type=str, default='VisualChatBot',
                        choices=[ "TextOnly", "VisualChatBot"],
                        help="Inference Mode: TextOnly: Text model only (Vicuna) \n VisualChatBot: Vision model + Text model (MiniGPT4) ")

    parser.add_argument("--attacked_image_fold", type=str)
    parser.add_argument("--random_image_fold", type=str)
    parser.add_argument("--raw_image_fold", type=str)
    parser.add_argument("--output_fold", type=str)

    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
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

out = []
attacked_image_files = os.listdir(args.attacked_image_fold)
random_image_files = os.listdir(args.random_image_fold)
raw_image_files = os.listdir(args.raw_image_fold)
attacked_image_files = sorted(attacked_image_files)
with torch.no_grad():
    for image_file in tqdm(attacked_image_files):
        image_file = image_file.split('.')[0]
        if (image_file + '.bmp') not in random_image_files:
            continue
        assert (image_file + '.jpg') in raw_image_files
        attacked_img = Image.open(os.path.join(args.attacked_image_fold, image_file + '.bmp')).convert('RGB')
        random_img = Image.open(os.path.join(args.random_image_fold, image_file + '.bmp')).convert('RGB')
        raw_img = Image.open(os.path.join(args.raw_image_fold, image_file + '.jpg')).convert('RGB')
        attacked_img = [vis_processor(attacked_img).unsqueeze(0).to(model.device)]
        random_img = [vis_processor(random_img).unsqueeze(0).to(model.device)]
        raw_img = [vis_processor(raw_img).unsqueeze(0).to(model.device)]
        prompt = prompt_wrapper.Prompt(model=model, img_prompts=[attacked_img, random_img, raw_img])
        attacked_img_emb = prompt.img_embs[0][0].cpu()
        random_img_emb = prompt.img_embs[1][0].cpu()
        raw_img_emb = prompt.img_embs[2][0].cpu()
        img_embs = torch.cat([attacked_img_emb, random_img_emb, raw_img_emb], dim=0)
        pickle.dump(img_embs, open(os.path.join(args.output_fold, image_file + '.pkl'), 'wb'))
