# Openvla-oft
## Environment Setup
The conda environment for openvla-oft with RoboTwin is identical to the official openvla-oft environment for the ALOHA part. Please follow the ([openvla-oft official documentation](https://github.com/moojink/openvla-oft/blob/main/SETUP.md)) to install the environment and directly overwrite the RoboTwin virtual environment in [INSTALLATION.md](../../INSTALLATION.md).

```bash
source /home/zaijia001/ssd/miniconda3/etc/profile.d/conda.sh
conda activate RoboTwin_openvla
cd policy/openvla-oft
./install_robotwin_openvla.sh
```
**Note!**  
This RoboTwin setup keeps the existing `torch 2.12.0.dev+cu128` stack instead of downgrading to the original `torch==2.2.0` recipe.

For Blackwell / `sm120`, `install_robotwin_openvla.sh` forces:
```bash
CUDA_HOME=<RoboTwin_openvla nvcc root>
TORCH_CUDA_ARCH_LIST=12.0
FLASH_ATTN_CUDA_ARCHS=120
```
and builds `flash-attn 2.8.3` only for `sm120`.

## Collect RoboTwin Data

See [RoboTwin Tutorial (Usage Section)](https://robotwin-platform.github.io/doc/usage/collect-data.html) for more details.

## Generate RLDS Data
> RLDS dataset is the data format required for Openvla-oft training.  

use RoboTwin data generation mechanism to generate data.   
Then convert the raw data to the aloha format that openvla-oft accepts: 
```
bash preproces_aloha.sh
```
Then transform the data to tfds form and register the tfds form dataset in your device: e.g.:
```
python -m datasets.stack_bowls_three_clean_builder
```
After converting to RLDS, register the dataset (which, for example, would be called `aloha_stack_bowls_three_clean_builder`) with our dataloader by adding an entry for it in `configs.py` ([here](prismatic/vla/datasets/rlds/oxe/configs.py#L680)), `transforms.py` ([here](prismatic/vla/datasets/rlds/oxe/transforms.py#L928)), and `mixtures.py` ([here](prismatic/vla/datasets/rlds/oxe/mixtures.py#L216)).Details in   [Openvla-oft official documentation](https://github.com/moojink/openvla-oft/blob/main/ALOHA.md ) 

## Finetune model
```
bash finetune_aloha.sh
```
By default, the training process will not save merged weights. So you need to run `merge_lora.sh` to merge lora weights if you want to use the checkpoint. If some `.py` files miss in the merged checkpoint, just copy them from the original checkpoint.
## Eval on RoboTwin
example usage
```
bash eval.sh move_can_pot demo_randomized ckpt_path 0 5 aloha_move_can_pot_builder
```

The evaluation results, including videos, will be saved in the `eval_result` directory under the project root.  
