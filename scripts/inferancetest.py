from transformers import AutoModel
from transformers import AutoTokenizer
from peft import PeftModel
import json
import torch
from convert_to_jsonl import format_example
#import evaluate

model = AutoModel.from_pretrained("model", trust_remote_code=True,  device_map='auto').half().cuda()
tokenizer = AutoTokenizer.from_pretrained("model", trust_remote_code=True)
model = PeftModel.from_pretrained(model, "./output/prerun")
#print(model)
instructions = json.load(open("data/TestTasks.json"))
#metric = evaluate.load('bleu')
#rouge_score = evalutae.load("ROUGE")
#print(metric)

answers = []

with torch.no_grad():
    for idx, item in enumerate(instructions[:10]):
        feature = format_example(item)
        input_text = feature['context']
        ids = tokenizer.encode(input_text)
        input_ids = torch.LongTensor([ids])
        input_ids = input_ids.to('cuda')
        out = model.generate(
            input_ids=input_ids,
            max_length=500,
            do_sample=False,
            temperature=0
        )
        out_text = tokenizer.decode(out[0])
        print("\n"+out_text+"\n")
        answer = out_text.replace(input_text, "").replace("\nEND", "").strip()
        item['infer_answer'] = answer
        #print("###"+input_text+"\n")
        #print("###"+out_text+"\n")
        #print(f"### {idx+1}.Answer:\n", item.get('target'), '\n\n')
        answers.append({'index': idx, **item})
    
    #results = metric.compute(predictions = ,references = )
    #print(results)




