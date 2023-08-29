from huggingface_hub import login

# login(<add token>)


import pandas as pd
import numpy as np
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoTokenizer,
)
import warnings

warnings.filterwarnings("ignore")


df = pd.read_csv("/content/train(1).csv")
df["question"] = (
    df["prompt"]
    + "\n A)"
    + df["A"]
    + "\n B)"
    + df["B"]
    + "\n C)"
    + df["C"]
    + "\n D)"
    + df["D"]
    + "\n E)"
    + df["E"]
    + "\n"
    + "You must only answer with the options and nothing else.I do not want an explanation, only three options that you think are mostly the answer. The answer to this question is"
    + df["answer"]
)
custom_ds = pd.DataFrame()
custom_ds["prompt"] = df["question"]


dataset = Dataset.from_pandas(custom_ds)


import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoTokenizer,
)

model_name = "meta-llama/Llama-2-7b-hf"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name, quantization_config=bnb_config, trust_remote_code=True
)
model.config.use_cache = False


tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token


from peft import LoraConfig, get_peft_model

lora_alpha = 16
lora_dropout = 0.1
lora_r = 64

peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
)


from transformers import TrainingArguments

output_dir = "./results"
per_device_train_batch_size = 4
gradient_accumulation_steps = 4
optim = "paged_adamw_32bit"
save_steps = 200
logging_steps = 10
learning_rate = 2e-4
max_grad_norm = 0.3
max_steps = 300
warmup_ratio = 0.03
lr_scheduler_type = "constant"

training_arguments = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    fp16=True,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=True,
    lr_scheduler_type=lr_scheduler_type,
)


from trl import SFTTrainer

max_seq_length = 512

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="prompt",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
)


for name, module in trainer.model.named_modules():
    if "norm" in name:
        module = module.to(torch.float32)


trainer.train()


model_to_save = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model  # Take care of distributed/parallel training
model_to_save.save_pretrained("outputs")


lora_config = LoraConfig.from_pretrained('outputs')
model = get_peft_model(model, lora_config)


text = dataset["prompt"][0]
device = "cuda:0"

preds = []
inputs = tokenizer(text, return_tensors="pt").to(device)
# outputs = model.generate(**inputs, max_new_tokens=50,return_dict_in_generate=True, output_scores=True)
outputs = model.generate(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_length=50,
    return_dict_in_generate=True,
    output_scores=True,
)

first_token_probs = outputs.scores[0][0]
option_scores = (
    first_token_probs[[319, 350, 315, 360, 382]].float().cpu().numpy()
)  # ABCDE
pred = np.array(["A", "B", "C", "D", "E"])[np.argsort(option_scores)[::-1][:3]]
pred = " ".join(pred)
preds.append(pred)
# print(tokenizer.decode(outputs[0], skip_special_tokens=True))


model.push_to_hub("Veer15/llama2-science-mcq-solver",create_pr=1)


