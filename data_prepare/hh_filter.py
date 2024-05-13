import json

train_path = "/data/huangyoucheng/mm-safety/data_prepare/hh_harmless/train.jsonl"
test_path = "/data/huangyoucheng/mm-safety/data_prepare/hh_harmless/test.jsonl"


def extract_anthropic_prompt(prompt_and_response):
    """Extract the anthropic prompt from a prompt and response pair."""
    search_term = '\n\nAssistant:'
    search_term_idx = prompt_and_response.rfind(search_term)
    assert search_term_idx != -1, f"Prompt and response does not contain '{search_term}'"
    return prompt_and_response[:search_term_idx + len(search_term)]


def data_filter(path, save_path):
    valid, total = 0, 0
    with open(path, 'r', encoding='utf-8') as f_in, open(save_path, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            data = json.loads(line)
            search_item = '\n\nAssistant:'
            if data['rejected'].count(search_item) == 1:
                total += 1
                prompt = extract_anthropic_prompt(data['rejected'])
                response = data['rejected'][len(prompt):].strip()
                response = response[:150]
                prompt = prompt[len("\n\nHuman: "):-len("\n\nAssistant:")]
                if len(prompt) > 100:
                    continue
                f_out.write(json.dumps({'prompt': prompt, 'response': response}) + '\n')
                valid += 1
    print(valid, total)


data_filter(train_path, "/data/huangyoucheng/mm-safety/data_prepare/hh_harmless/train_filtered.jsonl")
data_filter(test_path, "/data/huangyoucheng/mm-safety/data_prepare/hh_harmless/test_filtered.jsonl")
