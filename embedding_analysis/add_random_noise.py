import argparse
import os
import random
from tqdm import tqdm

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image

from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from torchvision.utils import save_image

from minigpt_utils import visual_attacker


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg_path", default="eval_configs/minigpt4_eval.yaml", help="path to configuration file.")
    parser.add_argument("--gpu_id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--attacked_image_fold", type=str)
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


def setup_seeds(seed):

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
setup_seeds(0)
model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))

vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
print('Initialization Finished')

out = []
attacked_image_files = os.listdir(args.attacked_image_fold)
raw_image_files = os.listdir(args.raw_image_fold)
attacked_image_files = sorted(attacked_image_files)
os.makedirs(args.output_fold, exist_ok=True)
with torch.no_grad():
    for image_file in tqdm(attacked_image_files):
        image_file = image_file.split('.')[0]
        assert (image_file + '.jpg') in raw_image_files
        raw_img = Image.open(os.path.join(args.raw_image_fold, image_file + '.jpg')).convert('RGB')
        raw_img = vis_processor(raw_img).unsqueeze(0).to(model.device)
        # 随机噪声
        epsilon = 32 / 255
        adv_noise = torch.rand_like(raw_img).to(model.device) * 2 * epsilon - epsilon
        x = visual_attacker.denormalize(raw_img).clone().to(model.device)
        adv_noise.data = (adv_noise.data + x.data).clamp(0, 1) - x.data
        x_adv = x + adv_noise
        x_adv = visual_attacker.normalize(x_adv)
        x_adv = visual_attacker.denormalize(x_adv)
        x_adv = x_adv.squeeze(0).cpu()
        save_image(x_adv, os.path.join(args.output_fold, image_file + '.bmp'))

