# Zen Musician LoRA Finetuning Guide

This guide walks you through finetuning Zen Musician using LoRA (Low-Rank Adaptation) to expand its capabilities across different musical genres and styles.

## Table of Contents
1. [Data Preparation](#step-1-data-preparation)
2. [Training Data Configuration](#step-2-training-data-configuration)
3. [LoRA Finetuning](#step-3-lora-finetuning)

## Requirements

- Python 3.10+ recommended
- PyTorch 2.4+ recommended
- CUDA 12.1+ recommended
- 24GB+ GPU memory (for LoRA training)

```bash
cd /Users/z/work/zen/zen-musician/finetune/
conda create -n zen-musician python=3.10
conda activate zen-musician
pip install -r requirements.txt
```

## Step 1: Data Preparation

### Required Data Structure

Organize your training data as follows:
```
example/
├── jsonl/     # Source JSONL metadata files
├── mmap/      # Generated Megatron binary files
└── npy/       # Discrete audio codes (numpy arrays) from xcodec
```

### JSONL File Format

Each JSONL file should contain music data in this format:

```json
{
    "id": "1",
    "codec": "example/npy/dummy.npy",                    // Raw audio codes
    "vocals_codec": "example/npy/dummy.Vocals.npy",      // Vocal track codes
    "instrumental_codec": "example/npy/dummy.Instrumental.npy",  // Instrumental track codes
    "audio_length_in_sec": 85.16,                        // Audio duration in seconds
    "msa": [                                             // Music Structure Analysis
        {
            "start": 0,
            "end": 13.93,
            "label": "intro"
        }
    ],
    "genres": "male, youth, powerful, charismatic, rock, punk",  // Tags: gender, age, genre, mood, timbre
    "splitted_lyrics": {
        "segmented_lyrics": [
            {
                "offset": 0,
                "duration": 13.93,
                "codec_frame_start": 0,
                "codec_frame_end": 696,
                "line_content": "[intro]\n\n"
            }
        ]
    }
}
```

### Converting Audio to Codec Format

1. Navigate to the finetune directory:
```bash
cd finetune/
```

2. Run preprocessing to convert audio to Megatron binary format:
```bash
# For Chain-of-Thought (CoT) dataset
bash scripts/preprocess_data.sh dummy cot $TOKENIZER_MODEL

# For In-Context Learning (ICL) dataset
bash scripts/preprocess_data.sh dummy icl_cot $TOKENIZER_MODEL
```

> **Note**: For music structure analysis and vocal/instrumental separation, see [openl2s](https://github.com/a43992899/openl2s) or [python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator).

## Step 2: Training Data Configuration

### Counting Dataset Tokens

1. Count tokens in your preprocessed dataset:
```bash
cd finetune/
bash scripts/count_tokens.sh ./example/mmap/
```

Results are saved in `finetune/count_token_logs/`. This may take several minutes for large datasets.

### Configuring Data Mixture

1. Create a configuration file (e.g., `finetune/example/zen_data_mixture_cfg.yml`):
   - `TOKEN_COUNT_LOG_DIR`: Directory with token count logs
   - `GLOBAL_BATCH_SIZE`: Total batch size for training
   - `SEQ_LEN`: Maximum context window size
   - `{NUM}_ROUND`: Dataset repetition count

2. Generate training parameters:
```bash
cd finetune/
python core/parse_mixture.py -c example/zen_data_mixture_cfg.yml
```

The script outputs:
- `DATA_PATH`: Training data paths (copy to training script)
- `TRAIN_ITERS`: Number of training iterations
- Total token count

## Step 3: LoRA Finetuning

Zen Musician uses LoRA (Low-Rank Adaptation) for efficient finetuning with reduced memory requirements.

### Why LoRA?

- **Memory Efficient**: Train with 24GB GPU instead of 80GB+
- **Fast Training**: Fewer parameters to update
- **Flexible**: Swap LoRA adapters for different genres
- **Quality**: Maintains model performance with less data

### Configuring the Finetuning Script

1. Edit `scripts/run_finetune.sh`:

```bash
# Update data paths
# Accepted formats for DATA_PATH:
#   1) Single path: "/path/to/data"
#   2) Multiple datasets: "100 /path/to/data1 200 /path/to/data2"
# Copy DATA_PATH from core/parse_mixture.py output in Step 2
DATA_PATH="data1-weight /path/to/data1 data2-weight /path/to/data2"
DATA_CACHE_PATH="/path/to/cache"

# Train/val/test split (comma-separated proportions)
DATA_SPLIT="900,50,50"

# Model configuration
TOKENIZER_MODEL_PATH="/path/to/tokenizer"
MODEL_NAME="m-a-p/YuE-s1-7B-anneal-en-cot"  # Base model
MODEL_CACHE_DIR="/path/to/model/cache"
OUTPUT_DIR="/path/to/save/zen-musician-lora"

# LoRA parameters
LORA_R=64              # Rank of LoRA update matrices (higher = more capacity)
LORA_ALPHA=32          # Scaling factor for LoRA updates
LORA_DROPOUT=0.1       # Dropout probability for LoRA layers
```

2. Configure training hyperparameters:
```bash
# Training settings
PER_DEVICE_TRAIN_BATCH_SIZE=1
NUM_TRAIN_EPOCHS=10
LEARNING_RATE=2e-4
WEIGHT_DECAY=0.01
```

### Running LoRA Finetuning

```bash
cd finetune/
bash scripts/run_finetune.sh
```

For configuration help:
```bash
bash scripts/run_finetune.sh --help
```

### Genre-Specific Training

To create genre-specific LoRA adapters:

1. Prepare separate datasets for each genre (rock, jazz, electronic, etc.)
2. Train separate LoRA adapters for each genre
3. Save adapters with descriptive names:
   - `zen-musician-lora-rock`
   - `zen-musician-lora-jazz`
   - `zen-musician-lora-electronic`

### Monitoring Training

- **WandB**: Enable with `USE_WANDB=true` for real-time monitoring
- **Tensorboard**: Track loss, learning rate, and validation metrics
- **Checkpoints**: Saved to `OUTPUT_DIR` at regular intervals

### Using Finetuned Models

After training, load your LoRA adapter for inference:

```python
from transformers import AutoModelForCausalLM
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("m-a-p/YuE-s1-7B-anneal-en-cot")

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, "/path/to/zen-musician-lora-rock")

# Use for inference
# ... inference code ...
```

### Merging LoRA Adapters

To create a standalone model (without needing base + adapter):

```python
from peft import PeftModel

# Load base + adapter
model = PeftModel.from_pretrained(base_model, "/path/to/lora")

# Merge weights
merged_model = model.merge_and_unload()

# Save merged model
merged_model.save_pretrained("/path/to/zen-musician-merged")
```

## Training Tips

### Data Quality
- Use high-quality audio (44.1kHz+, lossless formats)
- Ensure accurate lyric timestamps
- Verify genre tags match audio content
- Include diverse vocal styles per genre

### LoRA Configuration
- **LORA_R=64**: Good balance for music (higher than typical LLM values)
- **LORA_ALPHA=32**: Standard scaling (usually R/2)
- **LORA_DROPOUT=0.1**: Prevents overfitting on small datasets

### Training Duration
- **Small dataset (10-50 songs)**: 5-10 epochs
- **Medium dataset (50-200 songs)**: 3-5 epochs
- **Large dataset (200+ songs)**: 1-3 epochs

### Batch Size
- **24GB GPU**: batch_size=1, gradient_accumulation_steps=4
- **40GB GPU**: batch_size=2, gradient_accumulation_steps=2
- **80GB GPU**: batch_size=4, gradient_accumulation_steps=1

## Troubleshooting

### Out of Memory
- Reduce `PER_DEVICE_TRAIN_BATCH_SIZE`
- Increase `gradient_accumulation_steps`
- Lower `LORA_R` (try 32 or 16)
- Enable gradient checkpointing

### Poor Generation Quality
- Increase training epochs
- Check data quality and alignment
- Verify genre tags are accurate
- Try higher `LORA_R` (64 or 128)

### Overfitting
- Increase `LORA_DROPOUT` (0.15-0.2)
- Reduce training epochs
- Add more diverse training data
- Use data augmentation

## Next Steps

1. Train genre-specific LoRA adapters
2. Upload adapters to HuggingFace Hub
3. Create model cards with usage examples
4. Share with the Zen community!

## Resources

- [PEFT Documentation](https://huggingface.co/docs/peft)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [openl2s](https://github.com/a43992899/openl2s) - Music structure analysis
- [python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator) - Vocal separation

---

Part of the **Zen Musician** project - expanding music generation through LoRA finetuning.