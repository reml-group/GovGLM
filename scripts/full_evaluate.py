from transformers import AutoModel
from transformers import AutoTokenizer
from peft import PeftModel
import json
import torch
import numpy as np
from convert_to_jsonl import format_example
import jieba 
from rouge_chinese import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from sklearn.metrics import f1_score
from sklearn.metrics import precision_score

from tqdm import tqdm

model = AutoModel.from_pretrained("model", trust_remote_code=True,  device_map='auto').half().cuda()
tokenizer = AutoTokenizer.from_pretrained("model", trust_remote_code=True)
model = PeftModel.from_pretrained(model, "./lora/default")
instructions = json.load(open("data/evalgeneratetask0.json"))




print(f"len:{len(instructions)}")

def compute_cos(pred,ref):
    norm_pred = np.linalg.norm(pred,axis=0)
    norm_ref = np.linalg.norm(ref,axis=0)
    dot_product = np.sum(pred*ref,axis=0)
    cos_sim=0
    if norm_pred* norm_ref != 0:
        cos_sim = dot_product / (norm_pred* norm_ref)
    else:
        cos_sim=0
    return cos_sim

def compute_metrics(preds, labels):
    score_dict = {
        "rouge-1": [],
        "rouge-2": [],
        "rouge-l": [],
        "bleu-4": []
    }
    for pred, label in zip(preds, labels):
        hypothesis = list(jieba.cut(pred))
        reference = list(jieba.cut(label))
        
        rouge = Rouge()
        scores = rouge.get_scores(' '.join(hypothesis) , ' '.join(reference))
        result = scores[0]
            
        for k, v in result.items():
            score_dict[k].append(round(v["f"] * 100, 4))
        bleu_score = sentence_bleu([list(label)], list(pred), smoothing_function=SmoothingFunction().method3)
        score_dict["bleu-4"].append(round(bleu_score * 100, 4))

    for k, v in score_dict.items():
        score_dict[k] = float(np.mean(v))
    return score_dict

def convert_label(infer,groundt,label_pred,label_truth):
    n = len(infer)
    target_table =[]
    for item in groundt:
        if item not in target_table:
            target_table.append(item)

    for i in range(n):
        label_truth.append(target_table.index(groundt[i])) 
        for j in range(n):
            if infer[i].count(groundt[j])>0:
                label_pred.append(target_table.index(groundt[i]))
        if len(label_pred) < len(label_truth):
            label_pred.append(-1)

    return 0

def convert_multi_label(infer,groundt,label_pred1,label_pred2,label_truth1,label_truth2):
    n = len(infer)
    target_table1 =[]
    target_table2 =[]
    


    for item in groundt:
        itemsplit = item.split('-')
        #print(itemsplit)
        if itemsplit[0] not in target_table1:
            target_table1.append(itemsplit[0])
        if itemsplit[1] not in target_table2:
            target_table2.append(itemsplit[1])

    for i in range(n):
        groundtsplit = groundt[i].split('-')
        label_truth1.append(target_table1.index(groundtsplit[0])) 
        label_truth2.append(target_table2.index(groundtsplit[1])) 
        if infer[i].count(groundtsplit[0])>0:
            label_pred1.append(target_table1.index(groundtsplit[0]))
        if infer[i].count(groundtsplit[1])>0:
            label_pred2.append(target_table2.index(groundtsplit[1]))
        if len(label_pred1) < len(label_truth1):
            label_pred1.append(-1)
        if len(label_pred2) < len(label_truth2):
            label_pred2.append(-1)
    return 0

vec_metrics = []

infer_batch =[]
groundt_batch = []

pbar = tqdm(total=len(instructions))

with torch.no_grad():
    i=0
    for idx, item in enumerate(instructions[:5000]):
        i=i+1
        feature = format_example(item)
        input_text = feature['context']

        groundt_batch.append(feature['target'])
        groundt= tokenizer.encode(feature['target']) 
        groundt_ids = torch.LongTensor([groundt])
        groundt_ids = groundt_ids.to('cuda')

        outs=model(groundt_ids,output_hidden_states=True)
        embedding = list(outs.hidden_states)
        groundt_vec = embedding[0].cpu().numpy()
        groundt_vec = np.squeeze(groundt_vec)
        sum_groundt = groundt_vec.sum(axis=0)


        ids = tokenizer.encode(input_text)
        input_ids = torch.LongTensor([ids])
        input_ids = input_ids.to('cuda')
        out = model.generate(input_ids=input_ids,max_length=8000,do_sample=False,temperature=0)
        out_text = tokenizer.decode(out[0])
        
        infer_batch.append(out_text.split("Answer:")[1])
        infers=tokenizer.encode(out_text.split("Answer:")[1])
        #print(out_text.split("Answer:")[1])

        infers_ids = torch.LongTensor([infers])
        infers_ids = infers_ids.to('cuda')
        outs=model(infers_ids,output_hidden_states=True)
        embedding = list(outs.hidden_states)
        infer_vec = embedding[0].cpu().numpy()
        infer_vec = np.squeeze(infer_vec)
        sums_infer = infer_vec.sum(axis=0) 

        if np.isnan(compute_cos(sums_infer,sum_groundt)) ==False:
            vec_metrics.append(compute_cos(sums_infer,sum_groundt))

        #output = cos(infers, groundt)
        #print(output)
        #print(infers)
        
        if i%100 ==0:
            print(f"vec average{sum(vec_metrics)/len(vec_metrics)}")
            print(compute_metrics(infer_batch,groundt_batch))
    
        pbar.update(1)



    
    
    #print(results)
#print(vec_metrics)
pbar.close()
#print(f"vec average{sum(vec_metrics)/len(vec_metrics)}")
#print(compute_metrics(infer_batch,groundt_batch))

#classify only
'''     
label_pred = []
label_truth = []
convert_label(infer_batch,groundt_batch,label_pred,label_truth)
print(f1_score(label_truth,label_pred,average='weighted'))

label_pred1 = []
label_pred2 = []
label_truth1 = []
label_truth2 = []
convert_multi_label(infer_batch,groundt_batch,label_pred1,label_pred2,label_truth1,label_truth2)
print(f1_score(label_truth1,label_pred1,average='weighted'))
print(f1_score(label_truth2,label_pred2,average='weighted'))
'''