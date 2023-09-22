# Speech Separation in Car - README

## Repository Overview

This repository is dedicated to speech separation in car environments and is divided into two main parts: Training and Benchmarking. In the training section, we provide three different approaches: JAECBF, WaveUnet, and interchannel-convTasnet, each tailored for speech separation in a car. For benchmarking, we have scripts that handle data generation and Key Performance Indicator (KPI) computation for each approach.

## Training

### Prerequisites

Before you start training, make sure you have the following datasets and resources:

- **Librispeech**: A large dataset of English speech.
- **Mercedes Benz Dataset**: This dataset contains impulse responses and noise data for car environments. You can download it from [here](https://dss-kiel.de/index.php/media-center/data-bases/anir-corpus/).

### Training Setup

1. **Data Preparation**: Ensure that you have downloaded the required datasets (Librispeech and Mercedes Benz Dataset) and set up their paths in the training scripts.

2. **Mic Configurations**: The training script takes two arguments:
   - `--arg1`: A string in ['set0', 'set1', 'set2'], representing different mic configurations. For more details, please refer to the article in the repository.
   - `-arg2`: A string in ['0', '1', '2', '3'], indicating the CUDA device to use for training.

Example command for Training:
```bash
python Training_BeamRNN.py  --arg1 set0 --arg2 0 
python Training_WaveUnet.py  --arg1 set1 --arg2 1 
python Training_InterChannel.py  --arg1 set2 --arg2 3 

```
3. **Resume Training (Optional)**: By default, the training script is set to resume training from a saved checkpoint. If you want to start a new training session, you must hard code it by commenting out the line `model.load_state_dict`.

### Model Storage

During training, the models will be stored in the same folder as the training scripts. You will need to move them to the "Models" directory manually for safekeeping.

## Benchmarking

### Launching Benchmarking

To launch benchmarking, use the following command-line arguments:

- `--arg1`: A string in ['set0', 'set1', 'set2'], representing different mic configurations.
- `--arg2`: A string in ['Speech', 'Music'] to specify the content played by the loudspeakers.
- `--arg3`: A string in ['0', '1', '2', '3', '4'] indicating the CUDA device to use for benchmarking.
- `--arg4`: A value for SNR in dB (Signal-to-Noise Ratio) to set the desired noise level.

Example command for benchmarking:
```bash
python Benchmarking_ICassp.py --arg1 set0 --arg2 Speech --arg3 2 --arg4 20
```