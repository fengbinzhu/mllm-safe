import argparse
import os
import random
from tqdm import tqdm
import pickle

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image

# from minigpt4.common.config import Config
# from minigpt4.common.dist_utils import get_rank
# from minigpt4.common.registry import registry

# from minigpt_utils import prompt_wrapper
torch.set_num_threads(8)


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--model_path", type=str, default=0, help="llava model path")
    parser.add_argument("--model_base", type=str,  help="model base.")
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


from llava_llama_2.utils import get_model
args = parse_args()

print('model = ', args.model_path)

tokenizer, model, image_processor, model_name = get_model(args)
model.eval()
print('[Initialization Finished]\n')

# from llava_llama_2_utils import prompt_wrapper

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
        attacked_img = [image_processor.preprocess(attacked_img, return_tensors='pt')['pixel_values'].cuda()]
        random_img = [image_processor.preprocess(random_img, return_tensors='pt')['pixel_values'].cuda()]
        raw_img = [image_processor.preprocess(raw_img, return_tensors='pt')['pixel_values'].cuda()]
        # prompt = prompt_wrapper.Prompt(model=model, img_prompts=[attacked_img, random_img, raw_img])
        image_embs =  model.encode_images([attacked_img, random_img, raw_img])
        print(image_embs.shape)
        attacked_img_emb = image_embs[0][0].cpu()
        random_img_emb = image_embs[1][0].cpu()
        raw_img_emb = image_embs[2][0].cpu()
        img_embs = torch.cat([attacked_img_emb, random_img_emb, raw_img_emb], dim=0)
        pickle.dump(img_embs, open(os.path.join(args.output_fold, image_file + '.pkl'), 'wb'))
