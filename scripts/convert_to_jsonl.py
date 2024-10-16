import argparse
import json
from tqdm import tqdm


def format_example(example: dict) -> dict:
    context = f"Instruction: {example['instruction']}\n"
    context += f"Input: {example['input']}\n"
    context += "Answer: "
    target = example["target"]
    return {"context": context, "target": target}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="train.json")
    parser.add_argument("--save_path", type=str, default="data/train.jsonl")

    args = parser.parse_args()
    with open(args.data_path,encoding='utf-8') as f:
        examples = json.load(f)

    with open(args.save_path, 'w',encoding='utf-8') as f:
        for example in tqdm(examples, desc="formatting.."):
            f.write(json.dumps(format_example(example)) + '\n')


if __name__ == "__main__":
    main()
