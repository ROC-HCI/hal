# HAL: Inducing Human-likeness in LLMs with Alignment

Training and evaluation code for the HAL (Human Aligned LLM) paper, which uses Direct Preference Optimization (DPO) to align language models to exhibit more human-like characteristics.

## Project Structure

### Directories

#### `data/`
Contains all datasets used for training and evaluation:
- **Training data**: `dpo_train_hal16.csv` - DPO training dataset with HAL16Q preferences
- **Turing test data**: `turing_test_data_full.csv`, `turing_test_over_50.csv` - Human evaluation datasets
- **Synthetic data**: `synthlabs_*.csv`, `synthlabs_*.json` - Synthetically generated dialogues and personas
- **OOD datasets**: `OOD_dataset.csv`, `OOD_dataset_result_scored_gpt-5*.csv` - Out-of-distribution evaluation data
- **Human survey data**: `hal_prolific_*.csv` - Prolific survey responses and final results
- **Prompts**: `prompts_synthlabs_dialogues_*.csv` - Prompts used for dialogue generation

#### `HAL16_output/`
Contains evaluation results from HAL16Q judge assessments:
- `meta-llama/` - 71 CSV files with evaluation results for Meta Llama models
- `Qwen/` - 51 CSV files with evaluation results for Qwen models

#### `FS_output/`
Contains few-shot evaluation outputs from various model runs:
- Results from GPT-5, GPT-4.1, GPT-OSS, DeepSeek-R1, and other models

#### `Turing_output/`
Contains Turing test evaluation results:
- Results from various GPT models (GPT-5, GPT-4.1, GPT-OSS) on Turing test data

#### `Plots/`
Contains visualization outputs:
- Demographics plots
- HL score distributions
- Per-question Likert scale comparisons for base vs. DPO-trained models
- OOD evaluation plots

## Files

### Notebooks

Notebooks are organized based on where they appear on the paper.

#### Section 2: What Makes Human Conversation Human?
- **`2.1.a. Find_Characteristics_GPT-5.ipynb`** - Discovers human-like characteristics using GPT-5
- **`2.1.b. HAL_semantic_text_clustering_likert_statements.ipynb`** - Semantic clustering of Likert scale statements for HAL questionnaire

#### Section 3: Quantifying Human-likeness
- **`3. HAL32Q-Judge-Turing_data.ipynb`** - Evaluates Turing test data using HAL32Q judge questions
- **`3.1. HAL32Q_classifier_GPT-5.ipynb`** - Classifies responses using HAL32Q questions with GPT-5
- **`3.3.a. Data Proxy OOD dataset.ipynb`** - Creates out-of-distribution proxy dataset
- **`3.3.b. HAL16Q-Judge-OOD_data-GPT-5.ipynb`** - Evaluates OOD dataset using HAL16Q judge with GPT-5

#### Section 4: Inducing Human-likeness with Alignment
- **`4.1. HAL_Synhthlabs_data_demographics.ipynb`** - Analyzes demographics of synthetic SynthLabs data
- **`4.2.a. Persona_&_Dialogue_synthesis.ipynb`** - Synthesizes personas and dialogues for training
- **`4.2.b. HAL16Q-Judge-Synthlabs_data-GPT-5.ipynb`** - Evaluates SynthLabs data using HAL16Q judge
- **`4.2.c. DPO data creation.ipynb`** - Creates DPO training dataset from scored dialogues
- **`4.3.a_DPO-llama-3.2-1B-ddp.py`** - DPO training script for Llama-3.2-1B-Instruct
- **`4.3.b_DPO-llama-3.2-3B-ddp.py`** - DPO training script for Llama-3.2-3B-Instruct
- **`4.3.c_DPO-llama-3.1-8B-ddp.py`** - DPO training script for Llama-3.1-8B-Instruct
- **`4.3.d_DPO-qwen2.5-14B-ddp.py`** - DPO training script for Qwen2.5-14B-Instruct
- **`4.3.e_DPO-qwen2.5-32B-ddp.py`** - DPO training script for Qwen2.5-32B-Instruct
- **`4.3.f_DPO-llama-3.3-70B-ddp.py`** - DPO training script for Llama-3.3-70B-Instruct
- **`4.3.g_DPO-qwen2.5-72B-ddp.py`** - DPO training script for Qwen2.5-72B-Instruct
- **`4.4. HAL16Q-Judge-DPO-eval-GPT-5.ipynb`** - Evaluates DPO-trained models using HAL16Q judge
- **`4.5. HAL16Q-Judge_output_analysis.ipynb`** - Analyzes HAL16Q judge evaluation outputs

#### Section 5: Evaluating Human-likeness Training
- **`5.1.a. Human_Eval_Elo_Winrate.ipynb`** - Computes Elo ratings and win rates from human evaluations
- **`5.1.b. Human_Eval_survey_analysis.ipynb`** - Analyzes human evaluation survey responses
- **EmoBench:** https://github.com/Sahandfer/EmoBench
- **EQBench3:** https://github.com/EQ-bench/eqbench3

### Python Scripts

- **`tiny_agent.py`** - Wrapper class for interacting with various LLM APIs (OpenAI, Ollama) including GPT-5, GPT-4.1, GPT-OSS, and DeepSeek-R1 models
- **`ddp_4h100.yaml`** - Accelerate configuration file for distributed training on 4 H100 GPUs

### Configuration Files

- **`HAL16_judge_weights.json`** - Weights and bias for HAL16Q judge classifier

## Training Instructions

To train models using DPO, use the following commands with `accelerate`:

### Small Models (Multi-GPU)
```bash
accelerate launch --multi_gpu 4.3.a_DPO-llama-3.2-1B-ddp.py 2>&1 | tee logs/train_llama_1b.log
accelerate launch --multi_gpu 4.3.b_DPO-llama-3.2-3B-ddp.py 2>&1 | tee logs/train_llama_3b.log
accelerate launch --multi_gpu 4.3.c_DPO-llama-3.1-8B-ddp.py 2>&1 | tee logs/train_llama_8b.log
```

### Large Models (4x H100 Configuration)
```bash
accelerate launch --config_file ddp_4h100.yaml 4.3.d_DPO-qwen2.5-14B-ddp.py 2>&1 | tee logs/train_qwen_14b_ddp.log
accelerate launch --config_file ddp_4h100.yaml 4.3.e_DPO-qwen2.5-32B-ddp.py 2>&1 | tee logs/train_qwen_32b_ddp.log
accelerate launch --config_file ddp_4h100.yaml 4.3.f_DPO-llama-3.3-70B-ddp.py 2>&1 | tee logs/train_llama_70b.log
accelerate launch --config_file ddp_4h100.yaml 4.3.g_DPO-qwen2.5-72B-ddp.py 2>&1 | tee logs/train_qwen_72b_ddp.log
```

### Training Script Options

All training scripts support the following command-line arguments:
- `--override_base`: Overwrite `samples_epoch_0.csv` if it exists (otherwise skip)
- `--override_checkpoints`: Ignore existing checkpoints and start from base model

## Requirements

The project requires:
- `accelerate` for distributed training
- `transformers` and `trl` for DPO training
- `peft` for LoRA fine-tuning
- `wandb` for experiment tracking
- `pandas` and `datasets` for data handling
- OpenAI API access for GPT-5/GPT-4.1 models (via `tiny_agent.py`)

## License

See `LICENSE` file for details.
