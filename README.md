<div align="center">
<h1>Local All-Pair Correspondence for Point Tracking</h1>

[**Seokju Cho**](https://seokju-cho.github.io)<sup>1</sup> · [**Jiahui Huang**](https://gabriel-huang.github.io)<sup>2</sup> · [**Jisu Nam**](https://nam-jisu.github.io)<sup>1</sup> · [**Honggyu An**](https://hg010303.github.io)<sup>1</sup> · [**Seungryong Kim**](https://cvlab.korea.ac.kr)<sup>1</sup> · [**Joon-Young Lee**](https://joonyoung-cv.github.io)<sup>2</sup>

<sup>1</sup>Korea University&emsp;&emsp;&emsp;&emsp;<sup>2</sup>Adobe Research

**ECCV 2024**

<a href="https://arxiv.org/abs/2407.15420"><img src='https://img.shields.io/badge/arXiv-LocoTrack-red' alt='Paper PDF'></a>
<a href='https://ku-cvlab.github.io/locotrack'><img src='https://img.shields.io/badge/Project_Page-LocoTrack-green' alt='Project Page'></a>

<p float='center'><img src="assets/teaser.png" width="80%" /></p>
<span style="color: green; font-size: 1.3em; font-weight: bold;">LocoTrack is an incredibly efficient model,</span> enabling near-dense point tracking in real-time. It is <span style="color: red; font-size: 1.3em; font-weight: bold;">6x faster</span> than the previous state-of-the-art models.
</div>

## News
* **2024-07-22:** [LocoTrack](https://github.com/KU-CVLAB/locotrack/) is released.
* Please stay tuned for **Pytorch** version training and evaluation code!

## Preparing the Environment
```bash
git clone https://github.com/google-research/kubric.git
sudo apt-get install libopenexr-dev

conda create -n locotrack python=3.11
conda activate locotrack
pip install -U "jax[cuda12]"
pip install absl-py chex dm-haiku jaxline tensorflow matplotlib mediapy tensorflow_datasets tensorflow_graphics einshape wandb
```

## Running Evaluation
First, download the evaluation datasets:
```bash
# TAP-Vid-DAVIS dataset
wget https://storage.googleapis.com/dm-tapnet/tapvid_davis.zip
unzip tapvid_davis.zip

# TAP-Vid-RGB-Stacking dataset
wget https://storage.googleapis.com/dm-tapnet/tapvid_rgb_stacking.zip
unzip tapvid_rgb_stacking.zip

# RoboTAP dataset
wget https://storage.googleapis.com/dm-tapnet/robotap/robotap.zip
unzip robotap.zip
```
For downloading TAP-Vid-Kinetics, please refer to official [TAP-Vid repository](https://github.com/google-deepmind/tapnet/tree/main/tapnet/tapvid).

### LocoTrack Evaluation

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

## Training
### Training Dataset Preparation
Download the panning-MOVi-E dataset used for training (approximately 273GB) from Huggingface using the following script:
```bash
git lfs install
git clone git@hf.co:dataset/hamacojr/LocoTrack-panning-MOVi-E
```

### Training Script
```bash
export PYTHONPATH=$(cd ../ && pwd):$(pwd):$PYTHONPATH
python ./experiment.py --config ./configs/locotrack_config.py \
  --config.checkpoint_dir=[Path where checkpoints will be saved] \
  --config.experiment_kwargs.config.shared_modules.locotrack_model_kwargs.model_size={small | base} \
  --config.experiment_kwargs.config.datasets.kubric_kwargs.data_dir=[Path to panning-MOVi-E dataset]
```

## Citing this Work
Please use the following bibtex to cite our work:
```
@misc{cho2024localallpaircorrespondencepoint,
      title={Local All-Pair Correspondence for Point Tracking}, 
      author={Seokju Cho and Jiahui Huang and Jisu Nam and Honggyu An and Seungryong Kim and Joon-Young Lee},
      year={2024},
      eprint={2407.15420},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2407.15420}, 
}
```

## Acknowledgement
This project is largely based on the [TAP repository](https://github.com/google-deepmind/tapnet). Thanks to the authors for their invaluable work and contributions.
