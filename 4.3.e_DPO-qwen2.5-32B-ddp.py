## ------------------
## Request model: https://huggingface.co/Qwen/Qwen2.5-32B-Instruct-GGUF
## accelerate launch --config_file ddp_4h100.yaml 7e_DPO-qwen2.5-32B-ddp.py 2>&1 | tee logs/train_qwen_32b_ddp.log
## Running on 4 H100
## ------------------

from keys import *
import wandb
import weave
import torch
from datasets import load_dataset
import sys
import os
import pandas as pd
import time
import argparse

from transformers.trainer_utils import get_last_checkpoint
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model
from trl import DPOTrainer, DPOConfig
from accelerate import PartialState

wandb.login(key=WANDB)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--override_base",
    action="store_true",
    help="Overwrite samples_epoch_0.csv if it exists. Otherwise skip."
)
parser.add_argument(
    "--override_checkpoints",
    action="store_true",
    help="Ignore existing checkpoints and start from base model."
)

args, unknown = parser.parse_known_args()

# meta-llama/Llama-3.2-1B-Instruct
# meta-llama/Llama-3.2-3B-Instruct
# meta-llama/Llama-3.1-8B-Instruct
# Qwen/Qwen2.5-14B-Instruct
# Qwen/Qwen2.5-32B-Instruct          <--
# meta-llama/Llama-3.3-70B-Instruct
# Qwen/Qwen2.5-72B-Instruct
model_name = "Qwen/Qwen2.5-32B-Instruct"
trained_model_name = "hal16-qwen2.5-32B"
output_dir = "./HAL16_output/" + model_name + "/"
train_file = "data/dpo_train_hal16.csv"
test_file = "data/prompts_synthlabs_dialogues_test.csv"

checkpoint_dir = output_dir + "model_checkpoints/"
model_dir = output_dir + "model/"
inference_dir = output_dir + "inference/"

os.makedirs(output_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(inference_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, "scored"), exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

max_seq_length = 1024
dtype = None
load_in_4bit = True
ps = PartialState()
device_map = {"": ps.local_process_index}

per_device_train_batch_size = 1  ## Change
gradient_accumulation_steps = 32
inference_batch_size = 64
warmup_ratio = 0.1
num_train_epochs = 10  ## set to 10
learning_rate = 5e-5  ## change
logging_steps = 5
optim = "adamw_8bit"
weight_decay = 0.0
lr_scheduler_type = "linear"
seed = 42
save_strategy = "epoch",
save_safetensors = True     # smaller + faster to load
dpo_beta = 0.1
max_prompt_length = 1024

# LoRA
lora_dropout = 0.1
lora_r = 16
lora_alpha = 32

# Wandb logging
if ps.is_main_process:
    wandb.init(
        project=trained_model_name,
        name="run-001",
        id="run-001",
        resume="allow",
        config={
            "model": model_name,
            "lr": learning_rate,
            "batch_size": per_device_train_batch_size,
            "epochs": num_train_epochs,
            "beta": dpo_beta,
        }
    )
    wandb.log({"status": "training_started"})


# 4-bit quantization (bitsandbytes)
bnb_config = None
if load_in_4bit:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    )

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    token=HF_TOKEN,
    use_fast=True,
)
tokenizer.padding_side = "left"

# Ensure padding works
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    token=HF_TOKEN,
    quantization_config=bnb_config,
    torch_dtype=(torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if dtype is None else dtype,
)

model.config.use_cache = False

# ref_model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     token=HF_TOKEN,
#     torch_dtype=torch.bfloat16,
#     trust_remote_code=True,
# )
# ref_model.requires_grad_(False)
# ref_model.eval()

ref_model = None ## to avoid OOM in training 70B


# LoRA (PEFT)
model = get_peft_model(
    model,
    LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    ),
)


print(f"Model loaded: {model_name}")

dataset = load_dataset("csv", data_files={"train": train_file})["train"]

def preprocess_fn(example):
    return {
        "prompt": example["prompt"],
        "chosen": example["chosen"],
        "rejected": example["rejected"],
    }

dataset = dataset.map(preprocess_fn, remove_columns=dataset.column_names)
print(f"Dataset ready: {train_file}")




### ------------[Distributed inference]--------------
from accelerate.utils import gather_object

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    
class DistributedInferenceCallback(TrainerCallback):
    def __init__(self, model, tokenizer, test_file, save_dir, test_num=100, batch_size=inference_batch_size):
        self.model = model
        self.tokenizer = tokenizer
        # Load all prompts once
        self.test_prompts = pd.read_csv(test_file)["prompt"].tolist()[:test_num]
        self.save_dir = save_dir
        self.batch_size = batch_size  # Process 16 prompts at a time per GPU
        os.makedirs(save_dir, exist_ok=True)
        self.base_path = os.path.join(self.save_dir, "samples_epoch_0.csv")

    def _run_inference(self, csv_path, epoch_tag):
        t0 = time.time()
        self.model.eval()
        
        # 1. Distribute prompts: Each GPU gets a slice of the data
        with ps.split_between_processes(self.test_prompts) as local_prompts:
            local_results = []
            
            # 2. Batch Processing on each GPU
            for i in range(0, len(local_prompts), self.batch_size):
                batch_prompts = local_prompts[i : i + self.batch_size]
                
                inputs = self.tokenizer(
                    batch_prompts, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True, 
                    max_length=max_prompt_length
                ).to(self.model.device)

                with torch.no_grad():
                    out_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=max_seq_length,
                        use_cache=True,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,   # add this
                    )
                
                # Decode batch
                decoded_batch = self.tokenizer.batch_decode(out_ids, skip_special_tokens=True)
                
                for prompt, response in zip(batch_prompts, decoded_batch):
                     local_results.append({"prompt": prompt, "response": response})

        # 3. Gather results from all GPUs to the main process
        all_results = gather_object(local_results)

        # 4. Save to CSV (Only Main Process)
        if ps.is_main_process:
            # Deduplicate if necessary (gather_object might duplicate borders depending on version)
            # but usually fine for eval
            df = pd.DataFrame(all_results)
            df.to_csv(csv_path, index=False)
            print(f"✅ {epoch_tag} saved: {csv_path}. Rows: {len(df)}. Time: {time.time() - t0:.2f}s")
        
        # Sync all processes
        ps.wait_for_everyone()
        self.model.train() 

    def on_train_begin(self, args, state, control, **kwargs):
        # Check if we should run (based on file existence)
        should_run = True
        if os.path.exists(self.base_path) and not parser.parse_known_args()[0].override_base:
            should_run = False
            if ps.is_main_process:
                print(f"➡️ {self.base_path} exists. Skipping base inference.")

        # If running, ALL processes must enter _run_inference
        if should_run:
            if ps.is_main_process:
                 print(f"🧪📈 Running distributed base inference...")
            self._run_inference(self.base_path, epoch_tag="Base")

    def on_epoch_end(self, args, state, control, **kwargs):
        if ps.is_main_process:
            print(f"🧪📈 Inference on Epoch {int(state.epoch)}")
        path = os.path.join(self.save_dir, f"samples_epoch_{int(state.epoch)}.csv")
        self._run_inference(path, epoch_tag=f"Epoch {int(state.epoch)}")
### ------------- [DistributedInference end]-----------------------

inf_callback = DistributedInferenceCallback(model, tokenizer, test_file, inference_dir)



dpo_trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=DPOConfig(
        per_device_train_batch_size=per_device_train_batch_size,  ## Change
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_ratio=warmup_ratio,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,  ## change
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=logging_steps,
        optim=optim,
        weight_decay=weight_decay,
        lr_scheduler_type=lr_scheduler_type,
        seed=seed,
        save_strategy="epoch",
        save_safetensors=True,
        output_dir=checkpoint_dir,
        report_to="wandb",
        beta=dpo_beta,
        # --- Turn on gradient checkpointing ---
        gradient_checkpointing=True, 
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,
        # -----------------------
    ),
    train_dataset=dataset,
    processing_class=tokenizer,
    # max_length=max_seq_length,
    # max_prompt_length=max_prompt_length,
    callbacks=[inf_callback],
)

# Check for existing checkpoint and ask user
resume_path = None
last_checkpoint = get_last_checkpoint(checkpoint_dir)

if last_checkpoint is not None:
    print(f"⚠️ Existing checkpoint found at: {last_checkpoint}")

    if args.override_checkpoints:
        print("🧹 override_checkpoints set — starting from base model.")
        resume_path = None
    else:
        print(f"🔄 Resuming training from {last_checkpoint}")
        resume_path = last_checkpoint

# Train
dpo_trainer.train(resume_from_checkpoint=resume_path)

# Save
model.save_pretrained(model_dir + trained_model_name)  # Local saving
tokenizer.save_pretrained(model_dir + trained_model_name)

# NOTE: Unsloth-only GGUF export removed.
# For GGUF, use llama.cpp conversion tools externally.
