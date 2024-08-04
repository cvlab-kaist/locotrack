# JAX implementation of LocoTrack

## Preparing the Environment
```bash
git clone https://github.com/google-research/kubric.git
sudo apt-get install libopenexr-dev

conda create -n locotrack python=3.11
conda activate locotrack
pip install -U "jax[cuda12]"
pip install absl-py chex dm-haiku jaxline tensorflow matplotlib mediapy tensorflow_datasets tensorflow_graphics einshape wandb
```

## LocoTrack Evaluation

To evaluate LocoTrack on the benchmarks, follow these steps:

1. **Add the Project to PYTHONPATH**

   ```bash
   export PYTHONPATH=$(cd ../ && pwd):$(pwd):$PYTHONPATH
   ```

2. **Download Pre-trained Weights**

    Download the weight from the [link](https://huggingface.co/datasets/hamacojr/LocoTrack-weights/resolve/main/locotrack_weights.tar) and place the pre-trained weights into the `locotrack_weights/` directory.
    ```bash
    wget https://huggingface.co/datasets/hamacojr/LocoTrack-weights/resolve/main/locotrack_weights.tar
    tar -xvf locotrack_weights.tar
    ```

3. **Run the Evaluation**

    Use the following command to evaluate LocoTrack on TAP-Vid-DAVIS. Adjust the parameters for different dataset modes and model sizes. For evaluation at higher resolutions (e.g., 384x512), please adjust `refinement_resolutions` in `configs/locotrack_config.py`.

    ```bash
    python3 experiment.py \
      --config=configs/locotrack_config.py \
      --jaxline_mode={eval_davis_points | eval_davis_points_q_first | eval_kinetics_points | eval_kinetics_points_q_first | eval_robotics_points | eval_robotics_points_q_first | eval_robotap_points | eval_robotap_points_q_first} \
      --config.checkpoint_dir=locotrack_weights/locotrack_{base | small} \
      --config.experiment_kwargs.config.shared_modules.locotrack_model_kwargs.model_size={small | base} \
      --config.experiment_kwargs.config.davis_points_path=[Path to downloaded tapvid_davis.pkl file] \
      --config.experiment_kwargs.config.kinetics_points_path=[Path to the dataset folder] \
      --config.experiment_kwargs.config.robotics_points_path=[Path to the dataset folder] \
      --config.experiment_kwargs.config.robotap_points_path=[Path to the dataset folder]
    ```

#### Parameters Description:

- **`--jaxline_mode`**: Specify the dataset and query mode. If `q_first` is included, query-first mode is used; otherwise, strided query mode is used.
  - Options:
    - `eval_davis_points`
    - `eval_davis_points_q_first`
    - `eval_kinetics_points`
    - `eval_kinetics_points_q_first`
    - `eval_robotics_points`
    - `eval_robotics_points_q_first`
    - `eval_robotap_points`
    - `eval_robotap_points_q_first`

- **`--config.checkpoint_dir`**: Specify the path of the checkpoint. The checkpoint should match the specified size.
  - Options:
    - `locotrack_weights/locotrack_small`
    - `locotrack_weights/locotrack_base`

- **`--config.experiment_kwargs.config.shared_modules.locotrack_model_kwargs.model_size`**: Select the model size.
  - Options:
    - `small`
    - `base`

- **`--config.experiment_kwargs.config.davis_points_path`**: Path to the downloaded `tapvid_davis.pkl` file.

- **`--config.experiment_kwargs.config.kinetics_points_path`**: Path to the Kinetics dataset folder.

- **`--config.experiment_kwargs.config.robotics_points_path`**: Path to the downloaded `tapvid_rgb_stacking.pkl` file.

- **`--config.experiment_kwargs.config.robotap_points_path`**: Path to the Robotap dataset folder.

Replace the placeholder paths with the actual paths to your datasets.


## LocoTrack Training

### Training Dataset Preparation
Download the panning-MOVi-E dataset used for training (approximately 273GB) from Huggingface using the following script. Git LFS should be installed to download the dataset. To install Git LFS, please refer to this [link](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage?platform=linux). Additionally, downloading instructions for the Huggingface dataset are available at this [link](https://huggingface.co/docs/hub/en/datasets-downloading)
```bash
git clone git@hf.co:datasets/hamacojr/LocoTrack-panning-MOVi-E
```

### Training Script
```bash
export PYTHONPATH=$(cd ../ && pwd):$(pwd):$PYTHONPATH
python ./experiment.py --config ./configs/locotrack_config.py \
  --config.checkpoint_dir=[Path where checkpoints will be saved] \
  --config.experiment_kwargs.config.shared_modules.locotrack_model_kwargs.model_size={small | base} \
  --config.experiment_kwargs.config.datasets.kubric_kwargs.data_dir=[Path to panning-MOVi-E dataset]
```
