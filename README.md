# Backdooring Bias into Text-to-Image Models
![alt text](images/overall.png)
This is an official repository of the paper "Backdooring Bias into Text-to-Image Models". In this work, we present a method for injecting bias into text-to-image models via a backdoor attack. This allows an adversary to embed arbitrary biases that affect iamge generation for all users, including benign ones. Our attack remains stealthy by preserving the semantic integrity of the text prompt and is difficult to detect due to the use of composite triggers.

### 1. Create Poisoning Dataset
We first genrete the poisoned dataset for fine-tuning the pre-trained Stable Diffusion. You may change the corresponding `categories` for fine-tuning.
Run:
```
python pkl_disk_midjourney.py
```

### 2. Train Backdoored Biased Model
Fine-tune pre-trained Stable Diffusion model (2.0, XL, XL-Turbo) using the generated poisoned dataset. 

For fine-tuning Stable Diffusion 2.0 or below:
```
./run.sh
```

For fine-tuning Stable Diffusion-XL or XL-Turbo:
```
./sdxl_run.sh
```

Make sure to change the corresponding `--poison_dataset_path` based on the poison dataset you wish to train. The dataset is available in the `data` directory.

### 3. Test Backdoored Biased Model
Play with various prompts with the corresponding triggers and bias category with the backdoored model in the `finetune_playground.ipynb` file.
