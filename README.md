# <u>GovGLM</u>

## Overview

This repository contains the code for the paper ***A Contrastive Learning-Driven Method for Constructing Government Affairs Large Language Models*** and the Chinese Government Affairs Corpus, **PolicyBank**.

## PolicyBank

The **PolicyBank** corpus is the largest Chinese government affairs dataset, consisting of approximately 190,000 policy records from various institutions of government. The affined data are posted across official websites and open to the public.  You can access the corpus through [PolicyBank](https://pan.baidu.com/s/1ca9kpHGxmgeo1mB1qr3cCA).

## Usage

### Raw database

For understanding the composition of the *PolicyBank* , we provide a simple script to separate the database into easy-to-read text file via the folloing command:

`python ./SepTxt.py`

,which separates the entire dataset by year into `{year}.txt` files. 

### Getting Started

To set up the environment, install the required dependencies:

`pip install -r requirements.txt`

### Foundational Model

The finetuning process is based on the model **ChatGLM-6B**. You can find the model on [ChatGLM-6B](https://huggingface.co/THUDM/chatglm-6b). 

In order to replicate our results, it is advised to have stable access to corresponding model, either to huggingface.co(or similar services) or cloned file on disk. See details below.

### Generate Dataset

Our finetuning and evaluation dataset are originated from PolicyBank. However, relational database are not suitable for providing LLM training dataset. To mitigate this gap, we choose to generate dataset via the following steps:

#### Contrastive Learning Task Dataset Generation

`TextToPromptedSample.py` formulates the dataset for different contrastive learning tasks, including classification, summarization, and text generation tasks. 

1.adjusting related prompts. The original prompts are hard-coded into scripts but added randomly to different records. One can modify the prompts contents by modifying the corresponding lines.

2.adjust the ratio of tasks. Finetuning is done with mixed tasks of classification, summary and generate tasks(see details in the paper). It can be modified at the end of the script by changing `taskratioC` and `taskratioS` , where `taskratioC` (0-1) is the percentage of classification task and `taskratioS` (0-1) is the percentage of  summary task.

3.generate a JSON dataset:  run the script: `python ./TextToPromptedSample.py` to generate a `Train.json`  file. 

4.convert JSON to JSONL 

```bash
python cover_to_jsonl.py \
    --data_path Train.json \
    --save_path destination.jsonl \
```

5.Tokenization. Note this process rely on the tokenization file from ChatGLM-6B

```bash
python tokenize_word.py \
    --jsonl_path data/destination.jsonl \
    --save_path data/trainDataset \
    --max_seq_length 200 \ 
    --chatglm_path model_path/chatglm
```

control input and output by `--jsonl_path` and `--save_path` 

control maximum generated record with `--max_seq_length `, script would add padding if cap is not reached, therefore setting a large value could result in poor performance.

specify foundational model path with `--chatglm_path`



Now you should end up having dataset in `.arrow`-ish files.

### Training the Model

Run the following script to start training the model:

```bash
python finetune_lora.py 
	--dataset_path data/yourProcessedDataset 
	--lora_rank 8 
	--per_device_train_batch_size 20 
	--gradient_accumulation_steps 1 
	--max_steps 12000 
	--save_steps 500 
	--save_total_limit 3 
	--learning_rate 1e-4 
	--remove_unused_columns false 
	--logging_steps 50  
	--output_dir outputdir 
	--chatglm_path model
```

customize training setting on total training steps `--max_steps` , unit step to save trained file `--save_steps` , backup limits `--save_total_limit`, learning rate `--learning_rate`, logging steps `--logging_steps`

*:use `CUDA_VISIBLE_DEVICES=1` to specify the device to use, and `nohup  `  and `> traininglog.log` `2>&1 &` to make training semi-automated



### Evaluation

To evaluate the model, use the provided scripts to run the evaluation on the test dataset:

`python full_evaluate.py` , one can change whether to add trained module by commenting out line `model = PeftModel.from_pretrained(model, "./loraModule")`

the script would evaluate based on predetermined metrics and output a final score.

For visualization, simply run `python inferancetest.py` on inputs at `data/TestTasks.json`



