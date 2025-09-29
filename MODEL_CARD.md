---
language:
- en
- zh
- ja
- ko
- yue
license: apache-2.0
tags:
- music-generation
- audio
- text-to-music
- lora
- zen-ai
datasets:
- custom
pipeline_tag: text-to-audio
library_name: transformers
---

# Zen Musician 7B

**Zen Musician** is a music generation foundation model that transforms lyrics into complete songs with vocals and instrumental accompaniment. Based on the YuE architecture and enhanced for the Zen AI ecosystem.

## Model Details

- **Model Type**: Text-to-Music Generation
- **Architecture**: YuE-s1-7B (adapted)
- **Parameters**: 7B
- **License**: Apache 2.0
- **Languages**: English, Chinese (Mandarin/Cantonese), Japanese, Korean
- **Context Length**: Variable (supports multi-segment generation)
- **Developed by**: Zen AI Team
- **Based on**: [YuE by HKUST/M-A-P](https://github.com/multimodal-art-projection/YuE)

## Capabilities

- 🎵 **Full Song Generation**: Complete songs with vocals and accompaniment
- 🌍 **Multilingual**: Supports 5 languages including tonal languages
- 🎸 **Multi-Genre**: Pop, rock, electronic, folk, hip-hop, and more
- 🎤 **Vocal Control**: Control gender, timbre, and singing techniques
- 🔄 **Style Transfer**: Use audio prompts for in-context learning (ICL)
- ⏱️ **Long-Form**: Generate songs lasting several minutes
- 🎛️ **LoRA Support**: Fine-tune for custom genres and styles

## Use Cases

- Music composition from lyrics
- Genre-specific song generation
- Vocal style transfer and adaptation
- Music continuation and expansion
- Rapid prototyping for musicians
- Educational music creation
- Content creation for media

## Hardware Requirements

### Minimum
- **GPU**: 24GB VRAM (RTX 3090, RTX 4090, A10)
- **RAM**: 32GB system memory
- **Storage**: 50GB for model and dependencies

### Recommended
- **GPU**: 80GB VRAM (H100, A100) for full-length songs
- **RAM**: 64GB+ system memory
- **Storage**: 100GB for model, cache, and outputs

### Performance
- **H800**: ~150s for 30s audio
- **RTX 4090**: ~360s for 30s audio

## Training Data

Trained on diverse music datasets including:
- Multi-genre music across various styles
- Multiple languages and vocal techniques
- Lyrics-audio paired data
- High-quality studio recordings

The model inherits training from YuE and can be extended via LoRA finetuning.

## Intended Use

**Primary Use Cases**:
- Creative music composition
- Music production and prototyping
- Educational applications
- Research in music generation

**Out-of-Scope Uses**:
- Impersonating specific artists without permission
- Generating music for deceptive purposes
- Commercial use without proper attribution
- Violating copyright or intellectual property rights

## Limitations

- May occasionally generate repetitive patterns
- Quality varies with prompt specificity
- Requires specific prompt formatting for best results
- Limited by training data diversity
- Cannot perfectly replicate specific artist styles
- May struggle with complex musical structures

## Bias and Ethical Considerations

- Training data may reflect biases in music industry
- Generated content should be labeled as AI-generated
- Users should respect intellectual property rights
- Model may perpetuate musical stereotypes
- Cultural sensitivity required for multilingual use

## How to Use

### Installation

```bash
# Create environment
conda create -n zen-musician python=3.8
conda activate zen-musician

# Install PyTorch with CUDA
conda install pytorch torchvision torchaudio cudatoolkit=11.8 -c pytorch

# Install dependencies
git clone https://github.com/zenlm/zen-musician.git
cd zen-musician
pip install -r requirements.txt
pip install flash-attn --no-build-isolation

# Download tokenizer
cd inference/
git clone https://huggingface.co/m-a-p/xcodec_mini_infer
```

### Basic Generation

```bash
cd inference/
python infer.py \
    --cuda_idx 0 \
    --stage1_model m-a-p/YuE-s1-7B-anneal-en-cot \
    --stage2_model m-a-p/YuE-s2-1B-general \
    --genre_txt ../prompt_egs/genre.txt \
    --lyrics_txt ../prompt_egs/lyrics.txt \
    --run_n_segments 2 \
    --stage2_batch_size 4 \
    --output_dir ../output
```

### LoRA Finetuning

```bash
cd finetune/
# See finetune/README.md for detailed instructions
```

## Training with Zen Gym

Train custom LoRA adapters using Zen Gym:

```bash
cd /path/to/zen-gym

# Configure training
llamafactory-cli train \
    --config configs/zen_musician_lora.yaml \
    --dataset your_music_dataset
```

## Inference with Zen Engine

Serve Zen Musician via API:

```bash
cd /path/to/zen-engine

cargo run --release -- serve \
    --model zenlm/zen-musician-7b \
    --port 3690
```

## Citation

```bibtex
@misc{zen-musician-2025,
  title={Zen Musician: Music Generation for the Zen AI Ecosystem},
  author={Zen AI Team},
  year={2025},
  howpublished={\url{https://github.com/zenlm/zen-musician}}
}

@misc{yuan2025yuescalingopenfoundation,
  title={YuE: Scaling Open Foundation Models for Long-Form Music Generation},
  author={Ruibin Yuan and Hanfeng Lin and Shuyue Guo and Ge Zhang and Jiahao Pan and Yongyi Zang and Haohe Liu and Yiming Liang and Wenye Ma and Xingjian Du and Xinrun Du and Zhen Ye and Tianyu Zheng and Zhengxuan Jiang and Yinghao Ma and Minghao Liu and Zeyue Tian and Ziya Zhou and Liumeng Xue and Xingwei Qu and Yizhi Li and Shangda Wu and Tianhao Shen and Ziyang Ma and Jun Zhan and Chunhui Wang and Yatian Wang and Xiaowei Chi and Xinyue Zhang and Zhenzhu Yang and Xiangzhou Wang and Shansong Liu and Lingrui Mei and Peng Li and Junjie Wang and Jianwei Yu and Guojian Pang and Xu Li and Zihao Wang and Xiaohuan Zhou and Lijun Yu and Emmanouil Benetos and Yong Chen and Chenghua Lin and Xie Chen and Gus Xia and Zhaoxiang Zhang and Chao Zhang and Wenhu Chen and Xinyu Zhou and Xipeng Qiu and Roger Dannenberg and Jiaheng Liu and Jian Yang and Wenhao Huang and Wei Xue and Xu Tan and Yike Guo},
  year={2025},
  eprint={2503.08638},
  archivePrefix={arXiv},
  primaryClass={eess.AS}
}
```

## Model Card Contact

For questions or issues:
- **GitHub Issues**: https://github.com/zenlm/zen-musician/issues
- **Organization**: https://github.com/zenlm

## Acknowledgements

This model is based on [YuE by HKUST/M-A-P](https://github.com/multimodal-art-projection/YuE), licensed under Apache 2.0. We thank the original authors for their groundbreaking work in open-source music generation.

Part of the **[Zen AI](https://github.com/zenlm)** ecosystem.