# Modular AI 

With modular AI the idea is to build small and compact components for AI research. These small components can be efficiently trained on the M1 Macbook Air and scaled as well. We are hoping to simplify the lengthy code and chip away modules that are not required. This approach will be beneficial for researchers who are looking to ready-made components and understand how these components complement each other. The functionality of the modules can be extended with custom classes and functions as well.

This repository contains the code to train a GPT model using different attention mechanisms. The script supports various configurations that can be set through command-line arguments.

The data is optimized for pandas dataset so if you are preprocessing dataloader from this repo make sure to go through the `tokenize.py` to get a clear understanding. 

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Arguments](#arguments)
- [Example Commands](#example-commands)

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/Nielspace/NeuralNet-study.git
   cd NeuralNet-study
   ```

2. Install the required dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Usage

To run the training script, use the following command:

```sh
python train.py [OPTIONS]
```

## Arguments

- `--data_path`: Path to your text file (default: `wiki_medical_terms`).
- `--batch_size`: Batch size (default: `2`).
- `--block_size`: Block size (default: `512`).
- `--n_layer`: Number of layers (default: `6`).
- `--n_head`: Number of attention heads (default: `8`).
- `--n_embd`: Embedding size (default: `512`).
- `--dropout`: Dropout rate (default: `0.0`).
- `--learning_rate`: Learning rate (default: `6e-4`).
- `--max_iters`: Maximum iterations (default: `5`).
- `--grad_clip`: Gradient clipping (default: `1.0`).
- `--eval_interval`: Evaluation interval (default: `2000`).
- `--eval_iters`: Number of evaluation iterations (default: `200`).
- `--log_interval`: Logging interval (default: `1`).
- `--init_from`: Initialization mode (`scratch`, `resume`, `gpt2*`; default: `scratch`).
- `--out_dir`: Output directory for checkpoints (default: `out`).
- `--backend`: Backend for distributed training (default: `gloo`).
- `--device`: Device to use for training (`mps` if available, else `cpu`; default: `mps` or `cpu`).
- `--gradient_accumulation_steps`: Gradient accumulation steps (default: `40`).
- `--local_attn_ctx`: Local attention context (default: `32`).
- `--attn_mode`: Attention mode (default: `local`).
- `--bias`: Use bias in the model (default: `False`).
- `--attention_type`: Type of attention mechanism to use (`causal`, `flash`, `sparse`; default: `sparse`).

## Example Commands

Here are some example commands to run the training script with different configurations:

1. Run with default settings:
   ```sh
   python train.py
   ```

2. Run with a specific data path and learning rate:
   ```sh
   python train.py --data_path hf://datasets/gamino/wiki_medical_terms/wiki_medical_terms.parquet --learning_rate 3e-4
   ```

3. Run with a different attention mechanism (`flash`):
   ```sh
   python train.py --attention_type flash
   ```

4. Run with a larger batch size and more gradient accumulation steps:
   ```sh
   python train.py --batch_size 4 --gradient_accumulation_steps 80
   ```

5. Run with all custom settings:
   ```sh
   python train.py --data_path hf://datasets/gamino/wiki_medical_terms/wiki_medical_terms.parquet --batch_size 4 --learning_rate 3e-4 --gradient_accumulation_steps 80 --local_attn_ctx 64 --attn_mode 'sparse' --bias True --attention_type flash
   ```

## Notes

- Ensure that the required datasets and necessary permissions are available when using paths like `hf://datasets/gamino/wiki_medical_terms/wiki_medical_terms.parquet`.
- The device will default to `mps` if available, otherwise `cpu`. Ensure that your environment supports the specified device.
- Adjust the command-line arguments as needed to fit your specific use case and environment.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
```

### How to Use

1. **Clone the repository**:
   ```sh
   git clone https://github.com/yourusername/gpt-training.git
   cd gpt-training
   ```

2. **Install dependencies**:
   ```sh
   pip install -r requirements.txt
   ```

3. **Run the training script**:
   ```sh
   python train.py [OPTIONS]
   ```

Replace `[OPTIONS]` with any of the command-line arguments specified above to customize your training run.
