import argparse
import torch
import os
from torchvision.utils import save_image

from PIL import Image

def parse_args():

    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--model-path", type=str, default="ckpts/llava_llama_2_13b_chat_freeze")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--gpu_id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--n_iters", type=int, default=5000, help="specify the number of iterations for attack.")
    parser.add_argument('--eps', type=int, default=32, help="epsilon of the attack budget")
    parser.add_argument('--alpha', type=int, default=1, help="step_size of the attack")
    parser.add_argument("--constrained", default=False, action='store_true')
    parser.add_argument("--batch_size", type=int, default=8)

    parser.add_argument("--attack_text_path", type=str, default="harmful_corpus/derogatory_corpus.csv")
    parser.add_argument("--image_path", type=str, default="val2017")
    parser.add_argument("--segment_id", type=int, default=0)
    parser.add_argument("--all_segments", type=int, default=4)

    parser.add_argument("--save_dir", type=str, default='output',
                        help="save directory")

    args = parser.parse_args()
    return args

def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    return image

# ========================================
#             Model Initialization
# ========================================

print('>>> Initializing Models')

from llava_llama_2.utils import get_model
args = parse_args()

print('model = ', args.model_path)

tokenizer, model, image_processor, model_name = get_model(args)
model.eval()
print('[Initialization Finished]\n')


if not os.path.exists(args.save_dir):
    os.mkdir(args.save_dir)

if args.attack_text_path.endswith('jsonl'):
    import json
    targets = []
    with open(args.attack_text_path, 'r') as f:
        for line in f:
            targets.append(json.loads(line)['response'])
else:
    import pandas as pd
    targets = pd.read_csv(args.attack_text_path, names=['text', 'other'])['text'].tolist()

# print(targets)
targets = targets[:6000]

all_image_files = os.listdir(args.image_path)
all_image_files = sorted(all_image_files)
seg_len = len(all_image_files) // args.all_segments
all_image_files = all_image_files[args.segment_id * seg_len:(args.segment_id + 1) * seg_len]

from llava_llama_2_utils import visual_attacker

print('device = ', model.device)
# my_attacker = visual_attacker.Attacker(args, model, tokenizer, targets, device=model.device, image_processor=image_processor)

# template_img = 'adversarial_images/clean.jpeg'
# image = load_image(template_img)
# image = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].cuda()
# print(image.shape)

from llava_llama_2_utils import prompt_wrapper
# text_prompt_template = prompt_wrapper.prepare_text_prompt('')
# print(text_prompt_template)

for image_file in all_image_files:

    my_attacker = visual_attacker.Attacker(args, model, tokenizer, targets, device=model.device, image_processor=image_processor)
    template_img = os.path.join(args.image_path, image_file)
    # img = Image.open(template_img).convert('RGB')
    # img = vis_processor(img).unsqueeze(0).to(model.device)
    image = load_image(template_img)
    img = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].cuda()
    print(img.shape)

    text_prompt_template = prompt_wrapper.prepare_text_prompt('')

    print('save to %s' % ('%s/%s.bmp' % (args.save_dir, image_file.split('.')[0])))
    if not args.constrained:
        adv_img_prompt = my_attacker.attack_unconstrained(text_prompt_template,
                                                          img=img, batch_size=args.batch_size,
                                                          num_iter=args.n_iters, alpha=args.alpha / 255)
    else:
        adv_img_prompt = my_attacker.attack_constrained(text_prompt_template,
                                                        img=img, batch_size=args.batch_size,
                                                        num_iter=args.n_iters, alpha=args.alpha / 255,
                                                        epsilon=args.eps / 255)
    save_image(adv_img_prompt, '%s/%s.bmp' % (args.save_dir, image_file.split('.')[0]))



# if not args.constrained:
#     print('[unconstrained]')
#     adv_img_prompt = my_attacker.attack_unconstrained(text_prompt_template,
#                                                             img=image, batch_size = 8,
#                                                             num_iter=args.n_iters, alpha=args.alpha/255)

# else:
#     adv_img_prompt = my_attacker.attack_constrained(text_prompt_template,
#                                                             img=image, batch_size= 8,
#                                                             num_iter=args.n_iters, alpha=args.alpha / 255,
#                                                             epsilon=args.eps / 255)


# save_image(adv_img_prompt, '%s/bad_prompt.bmp' % args.save_dir)
print('[Done]')
