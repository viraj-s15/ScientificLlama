# Llama 2 Science MCQ Solver

Fine Tuned Llama-2-7b to solve Science MCQ's

## Table of Contents

- [Llama 2 Science MCQ Solver](#llama-2-science-mcq-solver)
  - [Table of Contents](#table-of-contents)
  - [About ](#about-)
  - [Getting Started ](#getting-started-)
  - [Usage ](#usage-)

## About <a name = "about"></a>

Got inspired for this from <a href="https://www.kaggle.com/competitions/kaggle-llm-science-exam">this</a> kaggle competition. The training data for that exact kaggle competition has been used to fine tune the model. 

The model has been trained using PEFT and LoRA, weights and biases api was used to measure and analyse the resource consumption and loss of the model. It has been fine tuned till a loss of `0.058200` which seems good enoguh for real life purposes. The submission.csv for the kaggle competition will also be uploaded here in the root directory

## Getting Started <a name = "getting_started"></a>

- Create a new environment
- Activate it
- Install deps using the requirements.txt

```bash
python3 -m venv venv
source venv/source/bin
pip install -r requirements.txt
```

## Usage <a name = "usage"></a>

This model has been uploaded to <a href="https://huggingface.co/Veer15/llama2-science-mcq-solver">Hugging Face</a>. It will be accessible from and compatible with all hugging face apis.

---
### If you publish this model in your work please use this BibTex citation

```
@misc {viraj_shah_2023,
	author       = { {Viraj Shah} },
	title        = { llama2-science-mcq-solver (Revision baa10d4) },
	year         = 2023,
	url          = { https://huggingface.co/Veer15/llama2-science-mcq-solver },
	doi          = { 10.57967/hf/1038 },
	publisher    = { Hugging Face }
}
```
