# PyTorch Implementation of LocoTrack

## Preparing the Environment

```bash
git clone https://github.com/google-research/kubric.git

conda create -n locotrack-pytorch python=3.11
conda activate locotrack-pytorch

pip install torch torchvision torchaudio lightning==2.3.3 tensorflow_datasets tensorflow matplotlib mediapy tensorflow_graphics einops wandb
```

## LocoTrack Evaluation

### 1. Download Pre-trained Weights

To evaluate LocoTrack on the benchmarks, first download the pre-trained weights.

| Model       | Pre-trained Weights |
|-------------|---------------------|
| LocoTrack-S | [Link](https://huggingface.co/datasets/hamacojr/LocoTrack-pytorch-weights/resolve/main/locotrack_small.ckpt) |
| LocoTrack-B | [Link](https://huggingface.co/datasets/hamacojr/LocoTrack-pytorch-weights/resolve/main/locotrack_base.ckpt)  |

### 2. Adjust the Config File

In `config/default.ini` (or any other config file), add the path to the evaluation datasets to `[TRAINING]-val_dataset_path`. Additionally, adjust the model size for evaluation in `[MODEL]-model_kwargs-model_size`.

### 3. Run Evaluation

To evaluate the LocoTrack model, use the `experiment.py` script with the following command-line arguments:

```bash
python experiment.py --config config/default.ini --mode eval_{dataset_to_eval_1}_..._{dataset_to_eval_N}[_q_first] --ckpt_path /path/to/checkpoint --save_path ./path_to_save_checkpoints/
```

- `--config`: Specifies the path to the configuration file. Default is `config/default.ini`.
- `--mode`: Specifies the mode to run the script. Use `eval` to perform evaluation. You can also include additional options for query first mode (`q_first`), and the name of the evaluation datasets. For example:
  - Evaluation of the DAVIS dataset: `eval_davis`
  - Evaluation of DAVIS and RoboTAP in query first mode: `eval_davis_robotap_q_first`
- `--ckpt_path`: Specifies the path to the checkpoint file. If not provided, the script will use the default checkpoint.
- `--save_path`: Specifies the path to save logs. 

Replace `/path/to/checkpoint` with the actual path to your checkpoint file. This command will run the evaluation process and save the results in the specified `save_path`.

## LocoTrack Training

### Training Dataset Preparation

Download the panning-MOVi-E dataset used for training (approximately 273GB) from Huggingface using the following script. Git LFS should be installed to download the dataset. To install Git LFS, please refer to this [link](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage?platform=linux). Additionally, downloading instructions for the Huggingface dataset are available at this [link](https://huggingface.co/docs/hub/en/datasets-downloading).

```bash
git clone git@hf.co:datasets/hamacojr/LocoTrack-panning-MOVi-E
```

### Training Script

Add the path to the downloaded panning-MOVi-E to the `[TRAINING]-kubric_dir` entry in `config/default.ini` (or any other config file). Optionally, for efficient training, change `[TRAINING]-precision` in the config file to `bf16-mixed` to use `bfloat16`. Then, run the training with the following script:

```bash
python experiment.py --config config/default.ini --mode train_davis --save_path ./path_to_save_checkpoints/
```
