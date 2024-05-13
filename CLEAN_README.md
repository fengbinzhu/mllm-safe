### Installation

We take MiniGPT-4 (13B) as the sandbox to showcase our attacks. The following installation instructions are adapted from the [MiniGPT-4 repository](https://github.com/Vision-CAIR/MiniGPT-4).

**1. Set up the environment**

```bash
git clone https://github.com/Unispac/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models.git

cd Visual-Adversarial-Examples-Jailbreak-Large-Language-Models

conda env create -f environment.yml
conda activate minigpt4
pip install seaborn # this pip package should be manually installed
```

**2. Prepare the pretrained weights for MiniGPT-4**

> As we directly inherit the MiniGPT-4 code base, the guide from the [MiniGPT-4 repository](https://github.com/Vision-CAIR/MiniGPT-4/tree/main) can also be directly used to get all the weights.

* **Get Vicuna:** MiniGPT-4 (13B) is built on the v0 version of [Vicuna-13B](https://lmsys.org/blog/2023-03-30-vicuna/). Please refer to this [guide](https://github.com/Vision-CAIR/MiniGPT-4/blob/main/PrepareVicuna.md) from the MiniGPT-4 repository to get the weights of Vicuna.

  Then, set the path to the vicuna weight in the model config file [here](https://github.com/Unispac/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models/blob/main/minigpt4/configs/models/minigpt4.yaml#L16)(file at minigpt4/configs/models/minigpt4.yaml) at Line 16.
  I have already set this into "/data/huangyoucheng/models/Vicuna-V0-13B", which should be over-written.

* **Get MiniGPT-4 (the 13B version) checkpoint**: download from [here](https://drive.google.com/file/d/1a4zLvaiDBr-36pasffmgpvH5P7CKmpze/view?usp=share_link). 

  Then, set the path to the pretrained checkpoint in the evaluation config file in [eval_configs/minigpt4_eval.yaml](https://github.com/Unispac/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models/blob/main/eval_configs/minigpt4_eval.yaml#L11) (file at eval_configs/minigpt4_eval.yaml) Line 11.
  I have already set this into "/data/huangyoucheng/mm-safety/pretrained_minigpt4.pth", which should be over-witten.
<br>

### Training

**1. Data preparation**

The fold has already contained the necessary data. This part is included for the completeness.
Download the harmless subset of hh-rlhf at [here](https://huggingface.co/datasets/Anthropic/hh-rlhf/tree/main/harmless-base). We will only use the train.jsonl. And create the directories "data_prepare/hh_harmless" where we will store the downloaded data.

Then, we filter the data and keep the dialogues where only one interaction is included. Running:
```bash
python data_prepare/hh_filter.py
```

**2. Train and get the adversarial Image**


```bash
CUDA_VISIBLE_DEVICES=0 nohup python minigpt_visual_attack_hh_rlhf.py \\
--cfg_path eval_configs/minigpt4_eval.yaml --cfg_path eval_configs/minigpt4_eval.yaml --gpu_id 0 --n_iter 700 \\
--constrained --eps 32 --alpha 1 --save_dir visual_constrained_eps_32_hh_rlhf --batch_size 16 \\
--segment_id 0 --all_segments 8 >/dev/null 2>&1 &
```


This script wll start **one process** to train the Image.
One can start multiple processes synchronized. There are two arguments to enable the synchronization: `--segment_id` and `--all_segments`.
Controlling the **total process number** by setting `--all_segments` and **incrementally increasing `--segmend_id` by 1**.
In detail, we will separate the total raw images into `all_segments` non-overlapped sets. And each new process takes its subset according to the `segment_id`.

If using multiple GPUs, change by setting `CUDA_VISIBLE_DEVICES=GPU_id` while keep the argument `--gpu_id 0` unchanged.
One can use multiple GPUs or start multiple processes on the same GPU as long as the two arguments, `--segment_id` and `--all_segments`, are set correspondingly.

The generated adversarial images will be saved into the directory `--save_dir`, which is `visual_constrained_eps_32_hh_rlhf` as the above.

**3. Evaluation**

TBC
