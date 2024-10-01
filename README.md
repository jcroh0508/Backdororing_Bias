# Backdororing_Bias
![alt text](images/overall.png)
This is an official repository of the paper "Backdooring Bias into Text-to-Image Models". 

### 1. Create Poisoning Dataset
We first one to genrete the poisoned dataset for fine-tuning the pre-trained Stable Diffusion. Change the corresponding `categories` for fine-tuning.
Run:
```
python pkl_disk_midjourney.py
```

### 2. Train Backdoored Biased Model
For fine-tuning Stable Diffusion 2.0 or below:
```
./run.sh
```

For fine-tuning Stable Diffusion-XL or XL-Turbo:
```
./sdxl_run.sh
```

Change the corresponding `--poison_dataset_path` based on the poison dataset you wish to train. The dataset is available in the `data` directory.

### 3. Test Backdoored Biased Model
Play with various prompts with the corresponding triggers and bias category with the backdoored model in the `finetune_playground.ipynb` file.