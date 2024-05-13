import argparse
import json
import pickle
from tqdm import tqdm

from minigpt4.common.config import Config
from minigpt4.common.registry import registry
from minigpt_utils import prompt_wrapper


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg_path", default="eval_configs/minigpt4_eval.yaml", help="path to configuration file.")
    parser.add_argument("--gpu_id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
             "in xxx=yyy format will be merged into config file (deprecate), "
             "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args


args = parse_args()
cfg = Config(args)
model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))
model.eval()

llama_tokenizer = model.llama_tokenizer
text_format = prompt_wrapper.minigpt4_chatbot_prompt

data = []
with open("/data/huangyoucheng/mm-safety/data_prepare/hh_harmless/train_filtered.jsonl", 'r') as f:
    for line in f:
        data.append(json.loads(line))

new_data = []
for item in tqdm(data):
    prompt = item['prompt']
    response = item['response']
    prompt_segs = (text_format % prompt).split('<ImageHere>')
    seg_tokens = [
        llama_tokenizer(
            seg, return_tensors="pt", add_special_tokens=i == 0).input_ids
        for i, seg in enumerate(prompt_segs)
    ]
    to_regress_tokens = llama_tokenizer(
        response,
        return_tensors="pt",
        padding="longest",
        truncation=True,
        max_length=model.max_txt_len,
        add_special_tokens=False
    ).input_ids
    new_data.append({
        "seg_tokens": seg_tokens,
        "to_regress_tokens": to_regress_tokens,
    })
pickle.dump(new_data, open("/data/huangyoucheng/mm-safety/data_prepare/hh_harmless/train_filtered_tokenized.pkl", "wb"))
