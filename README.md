# Llava-llama
## LLaVA 1.5: Training on 150k Dataset

### Overview
This repository explores the **LLaVA 1.5** model, a state-of-the-art architecture designed for various natural language processing tasks. This project involves training LLaVA 1.5 on a dataset of subet of **150,000 samples** to evaluate its performance and capabilities.

### How LLaVA 1.5 Works
The LLaVA 1.5 model integrates a **Multi-Layer Perceptron (MLP)** projection layer with the LLaVA architecture, along with **QLoRA** (Quantized Low-Rank Adaptation) for efficient training and fine-tuning. 

- **Architecture**: 
  - The MLP projection maps image embeddings into the text embedding space, allowing the model to handle multi-modal inputs effectively.
  - The integration of QLoRA enables low-rank updates to the model's parameters, significantly reducing memory consumption and improving training speed while retaining performance.

### Training Script
- The complete training script can be found [here](https://github.com/11kartheek/Llava-qwen/blob/main/final_training.ipynb).

### Inference Scripts
- For running inference, refer to the following scripts:
  - [Inference Script](https://github.com/11kartheek/Llava-qwen/blob/main/inference.ipynb)
  - [App Inference Script](https://github.com/11kartheek/Llava-qwen/blob/main/app_inference.ipynb)

### Dataset
- **Size**: 150,000 samples
- **Format**: JSON
- **Source**: [Link to the dataset](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct) or describe how to obtain it.
- **Preprocessing**: (to be added, e.g., data cleaning, normalization steps, etc.)

### Base Model
The base model used for LLaVA 1.5 can be found [here](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct).

### Base Vision Model
- Link to the base vision model (e.g., CLIP) goes here.

### Training Logs Snapshot
- Include a snapshot or summary of your training logs to illustrate model performance and any significant observations during training.

### Hugging Face Space
Explore the model in action and its capabilities on Hugging Face Spaces: [LLaVA 1.5 Space](https://huggingface.co/spaces/Kartheekb7/llava_chat).

## Future Work and what i could have done better
- add chat memory for now it is just multi turn q&a
- use random concatenation as suggeseted in paper
- train on complete dataset


