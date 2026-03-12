# HMS2

Another annotation-free whole-slide training approach to pathological classification.
This repository provides scripts to reproduce the results in the paper "Deep neural network trained on gigapixel images improves lymph node metastasis detection in clinical settings", including model training, inference, visualization, and statistics calculation, etc.

[>> **Demo Video** <<](https://youtu.be/Kcx_d5nEUQ8) | [**Journal Link**](https://doi.org/10.1038/s41467-022-30746-1) | [**Our Website**](https://www.aetherai.com/)

[<img src="misc/demo.gif" width=960 />](https://youtu.be/Kcx_d5nEUQ8)

## Publications

Huang, SC., Chen, CC., Lan, J. et al. Deep neural network trained on gigapixel images improves lymph node metastasis detection in clinical settings. Nat Commun 13, 3347 (2022). https://doi.org/10.1038/s41467-022-30746-1

## License

Copyright (C) 2021 aetherAI Co., Ltd. All rights reserved. Publicly accessible codes are licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).

## Requirements

### Hardware Requirements

Make sure the system contains adequate amount of main memory space (minimal: 32 GB) to prevent out-of-memory error.

### Software Stacks

Although Poetry can set up most Python packages automatically, you should install the following native libraries manually in advance.

- CUDA 11.7+

CUDA is essential for PyTorch to enable GPU-accelerated deep neural network training. See https://docs.nvidia.com/cuda/cuda-installation-guide-linux/ .

- OpenMPI 3+

OpenMPI is required for multi-GPU distributed training. If `sudo` is available, you can simply install this by,
```sh
sudo apt install libopenmpi-dev
```

- Python 3.9+

The development kit should be installed.
```sh
sudo apt install python3.9-dev
```

- OpenSlide

OpenSlide is a library to read slides. See the installation guide in https://github.com/openslide/openslide .

### Python Packages

We use Poetry to manage Python packages. The environment can be automatically set up by,
```sh
cd [HMS2 folder]
python3 -m pip install poetry
python3 -m poetry install
python3 -m poetry run poe install
```

## Usage

Before initiating a training task, you should prepare several configuration files with the following step-by-step instructions. Refer to `projects/Camelyon16` as an example for training an HMS model on the CAMELYON16(https://camelyon16.grand-challenge.org/) dataset. 

### 0. (Optional) Try a pre-trained CAMELYON16

If you would like to try training HMS models using CAMELYON16 or evaluating pre-trained ones, here we provided contour description files and pre-trained weights trained at 2.5x, 5x, and 10x magnifications, which is available at https://drive.google.com/file/d/12Fv-OhAze_t2_bCX7l1S5iMCgQgOvHGF/view?usp=sharing .

After the ZIP file is downloaded, unzip it to the project folder:
```sh
unzip -o hms2_camelyon16.zip -d /path/to/hms2
```

Besides, you should prepare the slides of CAMELYON16 from https://camelyon16.grand-challenge.org/ into `projects/Camelyon16/slides`. Then follow the instructions below.

| Pre-trained model | AUC (95% CI)
| ----------------- | ----------------------------------
| Camelyon16_2.5x   | 0.6015 (0.5022-0.7008)
| Camelyon16_5x     | 0.6242 (0.5194-0.7291)
| Camelyon16_10x    | 0.9135 (0.8490-0.9781)

![Camelyon16_10x_hms2](misc/camelyon_10x_hms2.png)

### 1. Create a Project Folder

As a convention, create a project folder in `projects` with four sub-folders, `datalists`, `slides`, `contours`, `train_configs`, and `test_configs`.

### 2. Define Datasets

3 CSV files defining training, validation and testing datasets, respectively, should be placed in `projects/YOUR_PROJECT/datalists`. See `projects/Camelyon16/datalists` for example.

These CSV files should follow the format if your datasets were annotated in slide level:
```
[slide_1_basename],[slide_1_class]
[slide_2_basename],[slide_2_class]
...
```
, where [slide\_name\_\*] specify the filename **without extension** of a slide image and [class\_id\_\*] is an integer indicating a slide-level label (e.g. 0 for normal, 1 for cancerous). 

Given contour-level (e.g. LN-level) labels, construct the CSV files in:
```
[slide_1_contour_1],[slide_1_contour_1_class]
[slide_1_contour_2],[slide_1_contour_2_class]
...
```
You can name each contour whatever you want.

#### (Optional) Contour Description Files

For each contour, a contour description file in JSON should be composed with content like:
```
{"slide\_name": "slide\_1\_basename", "contours": contours}
```
, where `contours` is a list of contour. Each contour is a list of coordinates in (x, y). See `projects/Camelyon16/contours` for example. Save these files in `projects/YOUR_PROJECT/contours`.

### 3. Prepare Slide Files

Place the slides files in `projects/YOUR_PROJECT/slides`. Soft links (`ln -s`) also work.

### 4. Set Up Training Configurations

Model hyper-parameters are set up in a YAML file. You can copy `projects/Camelyon16/train_configs/2.5x.yaml` and modify it for your own preference. 

The following table describes each field in a config.

| Field                      | Description
| -------------------------- | ---------------------------------------------------------------------------------------------
| RESULT_DIR                 | Directory to store output stuffs, including model weights, testing results, etc.
| MODEL_PATH                 | Path to store the model weight. (default: `${RESULT_DIR}/model.pt`)
| OPTIMIZER_STATE_PATH       | Path to store the state of optimizer. (default: `${RESULT_DIR}/opt_state.pt`)
| HISTORY_DIR                | Directory to store model history. (default: `${RESULT_DIR}/history/`)
| STATES_PATH                | Path to store the states for resuming. (default: `${RESULT_DIR}/states.pt`)
| CONFIG_RECORD_PATH         | Path to back up this config file. (default: `${RESULT_DIR}/config.yaml`)
| TRAIN_EVENT_LOG_PATH       | Path to store the training event log. (default: `${RESULT_DIR}/train_event_log.json`)
|                            |
| USE_MIXED_PRECISION        | Whether to enable mixed precision training. (default: `False`)
| USE_HMS2                   | Whether to enable HMS2. (default: `True`)
| GPU_MEMORY_LIMIT_GB        | GPU memory limitation in GB. (default: `None`)
|                            |
| RESIZE_RATIO               | Resize ratio for downsampling slide images. Can be a `float` (e.g., `0.25`) or a dictionary containing `target_pixel_spacing` to specify the targeted pixel spacing in um (e.g., `{target_pixel_spacing: 0.92}`).
| GPU_AUGMENTS               | Augmentations to do on GPU with patch-based affine transformation. Available options are `"flip"`, `"rigid"`, `"hed_perturb"`, and `"gaussian_blur"`. (default: `["flip", "rigid", "hed_perturb"]`)
| AUGMENTS                   | Augmentations to do on CPU. (default: `[]`)
| CLASS_WEIGHTS              | A list of dictionaries or `None` to disable class weighting. Each dictionary contains `class_index`, `positivity`, and `weight`. (default: `None`)
|                            |
| NUM_CLASSES                | Number of classes.
| GPUS                       | The number of GPUs specified for assertion. (default: `None`)
| MODEL                      | Model architecture to use. Options: `"resnet50_frozenbn"`, `"resnet50_frozenbn_linear"`, `"resnet50_frozenall"`, `"resnet50_frozenall_linear"`, `"resnet50_frozenall_ap_linear"`, `"resnet50V1c_frozenbn"`. (default: `"resnet50_frozenbn"`)
| PRETRAINED                 | Pretrained weight to initialize the model. Can be no pretrained (`{type: "no"}`), torchvision weights (e.g., `{type: "torchvision", weights: "IMAGENET1K_V1"}`), backbone weights (e.g., `{type: "backbone", path: "/path/to/weight.pt"}`), or HMS2 weights (e.g., `{type: "hms2", path: "/path/to/weight.pt"}`). default: `{type: "torchvision", weights: "IMAGENET1K_V1"}`.
| PRE_POOLING                | Pre-pooling layer to use. Options: `"no"`, `"conv_1x1"`, `"conv_1x1_relu"`. (default: `"no"`)
| POOL_USE                   | Global pooling method. (default: `"re_lse"`)
| BATCH_SIZE                 | Number of slides processed in each training iteration for each MPI worker. (default: `1`)
| EPOCHS                     | Maximal number of training epochs. (default: `200`)
| LOSS                       | Loss to use with the format {"name": str, "arguments": dict[str, Any]}. (default: {"name": "ce"})
| METRIC_LIST                | A list of metrics. (default: `["accuracy"]`)
| OPTIMIZER                  | Optimizer for model updating. (default: `"adamw"`)
| INIT_LEARNING_RATE         | Initial learning rate for Adam optimizer. (default: `0.00001`)
| REDUCE_LR_FACTOR           | The learning rate will be decreased by this factor upon no validation loss improvement in consequent epochs. Set `0.0` to enable early stopping. (default: `0.1`)
| REDUCE_LR_PATIENCE         | Number of consequent epochs to reduce learning rate. (default: `8`)
|                            |
| TRAIN_DATASET_CONFIGS      | A list of training subdatasets (see below).
|                            |
| CLASS_NAMES                | Class names for display during evaluation. Set `None` (default) to ignore the setting.
|                            |
| USE_EMBED_MODE             | Whether to enable embed mode. (default: `False`)
| EMBED_MODE_CACHE_DIR       | Directory to store the embedding cache in embed mode. (default: `${RESULT_DIR}/embedding`)
| EMBED_MODE_GPU_COMPRESSOR  | The compressor running on GPU to compress the embeddings in embed mode. Options: `"identity"`, `"avg_pool_7x7"`, `"farthest_point_1/7"`. (default: `"farthest_point_1/7"`). Be aware that `"avg_pool_7x7"` is a deprecated method and may cause the **inconsistent performance issue** (i.e. gain good performance with embed mode but poor performance with standard mode) in some cases, so we recommend using `"farthest_point_1/7"` instead.
| EMBED_MODE_COMPRESSORS     | Compressors to use in embed mode. (default: `["int8", "unique_vector"]`)

A training subdataset is defined as:

| Field                      | Description
| -------------------------- | ---------------------------------------------------------------------------------------------
| TRAIN_CSV_PATH             | CSV file defining the training dataset.
| VAL_CSV_PATH               | CSV file defining the validation dataset. (default: `None`)
| SLIDE_DIR                  | Directory containing all the slide image files (can be soft links).
| SLIDE_FILE_EXTENSION       | File extension. (e.g., `".ndpi"`).
| CONTOUR_DIR                | Directory containing contour description files. Set `NULL` (default) when using slide-level labels.

### 5. Set Up Test Configurations

A training config can be bound with multiple test configs.

Each test YAML config contains the following fields: (See `projects/Camelyon16/test_configs/2.5x.yaml`)

| Field                      | Description
| -------------------------- | ---------------------------------------------------------------------------------------------
| TRAIN_CONFIG_PATH          | Path to the training config.
| TEST_RESULT_PATH           | Path to store the model predictions after testing in a JSON format. (default: `${RESULT_DIR}/test_result.json`)
| VIZ_RESULT_DIR             | Folder to store prediction maps. (default: `${RESULT_DIR}/viz_results`)
| TEST_EVENT_LOG_PATH        | Path to store the test event log. (default: `${RESULT_DIR}/test_event_log.json`)
| VIZ_EVENT_LOG_PATH         | Path to store the visualization event log. (default: `${RESULT_DIR}/viz_event_log.json`)
| TEST_DATASET_CONFIGS       | A list of test subdatasets (see below).
| VIZ_POOL_USE               | Pooling method for visualization. (default: `"cam"`)

Each test subdataset config is comprised of:

| Field                      | Description
| -------------------------- | ---------------------------------------------------------------------------------------------
| TEST_CSV_PATH              | CSV file defining the test dataset.
| SLIDE_DIR                  | Directory containing the test slide image files (can be soft links).
| SLIDE_FILE_EXTENSION       | File extension of test slides. (e.g., `".ndpi"`).
| CONTOUR_DIR                | Directory containing contour description files of test slides. Set `NULL` (default) when using slide-level labels.

**Note 1: If you have multiple test datasets and would like to evaluate them seperately, create multiple test configs. If not, specify them in `TEST_DATASET_CONFIGS` of one single config file.**

**Note 2: If you have only one test dataset and are seeking something minimal, you can compose an all-in-one config with both training and test fields inside, except for `TRAIN_CONFIG_PATH`.**

### 6. Train a Model

#### Standard Mode

To train a model, simply run
```sh
python3 -m poetry run python -m hms2.pipeline.train --config YOUR_TRAIN_CONFIG.YAML [--continue_mode]
```
, where `--continue_mode` is optional that makes the training process begin after loading the model weights.

To enable multi-node, multi-GPU distributed training, simply add `mpirun` in front of the above command, e.g.
```sh
mpirun -np 4 -x CUDA_VISIBLE_DEVICES="0,1,2,3" python3 -m poetry run python -m hms2.pipeline.train --config YOUR_TRAIN_CONFIG.YAML
```

Typically, this step takes days to complete, depending on the computing power, while you can trace the progress in real time from program output.

#### Embed Mode

If the backbone is fully frozen (e.g. `resnet50_frozenall`), you can use embed mode to train the model faster (speedup by 10x~40x in general). However, the model performance may be slightly worse than the standard mode due to following reasons:
- `GPU_AUGMENTS` and `AUGMENTS` are not available.
- By default, to reduce the disk io, disk usage and pcie bandwidth, the extracted embeddings are compressed by AvgPool7x7 and int8 quantization, then all duplicate embeddings are removed. It can be configured by setting `EMBED_MODE_COMPRESSORS` and `EMBED_MODE_GPU_COMPRESSOR` in the config file.

Here is an example of how to set up the training config file for embed mode.
```yaml
# Freeze the backbone and load the foundation model weights, only the backbones without any trainable layers are allowed
MODEL: resnet50_frozenall
PRETRAINED:
  type: backbone
  path: pretrained/bt-r50-all-fp32-batch1536-uniform-by-mag-01-long.pth

# Add a trainable 1d conv layer after backbone.
PRE_POOLING: conv_1x1_relu

# Enable embed mode
USE_EMBED_MODE: true
```

To train a model in embed mode, call
```sh
[mpirun ...] python3 -m poetry run python -m hms2.pipeline.extract_emb --config YOUR_TRAIN_CONFIG.YAML [--continue_mode]
```
, where `--continue_mode` is optional that makes the extraction process skip the already extracted embeddings.

This step will extract the embeddings from the training and validation datasets, and save them in `${EMBED_MODE_CACHE_DIR}`.

After the extraction is done, you can train the model in embed mode by calling
```sh
[mpirun ...] python3 -m poetry run python -m hms2.pipeline.train --config YOUR_TRAIN_CONFIG.YAML [--continue_mode]
```
, where `--continue_mode` is optional that makes the training process begin after loading the model weights.

This command will train the model using the extracted embeddings and save the full model weights (include backbone weights) in `${RESULT_DIR}`.

Note that `USE_EMBED_MODE` must be set to `true` in the training config file.

### 7. Evaluate the Model

To evaluate the model, call
```sh
[mpirun ...] python3 -m poetry run python -m hms2.pipeline.test --config YOUR_TEST_CONFIG.YAML
```

This command will generate a JSON file in the result directory named `test_result.json` by default.
The file contains the model predictions for each testing slide. 

To further conduct ROC analysis, call
```sh
python3 -m poetry run python -m hms2.pipeline.evaluation.evaluate_roc --config YOUR_TEST_CONFIG.YAML
```

To further conduct categorical analysis, call
```sh
python3 -m poetry run python -m hms2.pipeline.evaluation.evaluate_categorical --config YOUR_TEST_CONFIG.YAML
```

### 8. Visualize the Model

To generate the CAM heatmap of the model, call
```sh
[mpirun ...] python3 -m poetry run python -m hms2.pipeline.visualize --config YOUR_TEST_CONFIG.YAML
```

The heatmaps will be saved in `${VIZ_RESULT_DIR}` with the format of `.npy`.
The array will have a shape of [height, width, channels] and a data type of `float`.
The values inside the array will range from 0.0 to 1.0 for valid region; otherwise, area outside the contours will be represented by `np.nan`.

To extract representative patches, call
```
python3 -m poetry run python -m hms2.pipeline.evaluation.get_representative_patch --config YOUR_TEST_CONFIG.YAML
```

The patches will be saved in `${RESULT_DIR}/representative_patches/`.
