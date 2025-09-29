# Zen Musician

**Zen Musician** is a music generation foundation model based on YuE, adapted and enhanced for the Zen AI ecosystem. It transforms lyrics into full songs with both vocal and accompaniment tracks, supporting diverse genres, languages, and vocal techniques.

## About

Zen Musician is built on the groundbreaking YuE (乐) architecture and enhanced through LoRA finetuning to expand its capabilities across multiple genres and musical styles. The model can generate complete songs lasting several minutes, with support for:

- Multiple languages (English, Chinese, Japanese, Korean, Cantonese)
- Diverse musical genres
- Advanced vocal techniques
- In-context learning (ICL) for style transfer
- Music continuation and incremental generation

## Credits

This model is based on [YuE by HKUST/M-A-P](https://github.com/multimodal-art-projection/YuE), licensed under Apache 2.0. We thank the original authors for their groundbreaking work in open-source music generation.

## HuggingFace Models

- **[zenlm/zen-musician-7b](https://huggingface.co/zenlm/zen-musician-7b)**: Base model adapted from YuE-s1-7B
- **zenlm/zen-musician-lora-***: Genre-specific LoRA adapters (Coming soon)

## Hardware Requirements

### GPU Memory
- **24GB or less**: Run up to 2 sessions
- **80GB+ (H800, A100, RTX4090 multi-GPU)**: Full song generation with 4+ sessions

### Execution Time
- **H800 GPU**: ~150s for 30s audio
- **RTX 4090**: ~360s for 30s audio

## Installation

### 1. Setup Environment

```bash
# Create conda environment
conda create -n zen-musician python=3.8
conda activate zen-musician

# Install CUDA 11.8+
conda install pytorch torchvision torchaudio cudatoolkit=11.8 -c pytorch -c nvidia

# Install dependencies
pip install -r requirements.txt

# Install Flash Attention 2 (mandatory for memory efficiency)
pip install flash-attn --no-build-isolation
```

### 2. Download Code and Tokenizer

```bash
# Install git-lfs
sudo apt update && sudo apt install git-lfs
git lfs install

# Clone repository
git clone https://github.com/zenlm/zen-musician.git
cd zen-musician/inference/

# Download tokenizer
git clone https://huggingface.co/m-a-p/xcodec_mini_infer
```

## Usage

### Basic Generation (CoT Mode)

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
    --output_dir ../output \
    --max_new_tokens 3000 \
    --repetition_penalty 1.1
```

### Dual-Track ICL Mode (Style Transfer)

```bash
cd inference/
python infer.py \
    --cuda_idx 0 \
    --stage1_model m-a-p/YuE-s1-7B-anneal-en-icl \
    --stage2_model m-a-p/YuE-s2-1B-general \
    --genre_txt ../prompt_egs/genre.txt \
    --lyrics_txt ../prompt_egs/lyrics.txt \
    --run_n_segments 2 \
    --stage2_batch_size 4 \
    --output_dir ../output \
    --max_new_tokens 3000 \
    --repetition_penalty 1.1 \
    --use_dual_tracks_prompt \
    --vocal_track_prompt_path ../prompt_egs/pop.00001.Vocals.mp3 \
    --instrumental_track_prompt_path ../prompt_egs/pop.00001.Instrumental.mp3 \
    --prompt_start_time 0 \
    --prompt_end_time 30
```

### Single-Track ICL Mode

```bash
cd inference/
python infer.py \
    --cuda_idx 0 \
    --stage1_model m-a-p/YuE-s1-7B-anneal-en-icl \
    --stage2_model m-a-p/YuE-s2-1B-general \
    --genre_txt ../prompt_egs/genre.txt \
    --lyrics_txt ../prompt_egs/lyrics.txt \
    --run_n_segments 2 \
    --stage2_batch_size 4 \
    --output_dir ../output \
    --max_new_tokens 3000 \
    --repetition_penalty 1.1 \
    --use_audio_prompt \
    --audio_prompt_path ../prompt_egs/pop.00001.mp3 \
    --prompt_start_time 0 \
    --prompt_end_time 30
```

## LoRA Finetuning

Zen Musician supports LoRA (Low-Rank Adaptation) finetuning for genre expansion and style adaptation. See [finetune/README.md](finetune/README.md) for detailed instructions.

### Quick Start

```bash
cd finetune/
# Follow instructions in finetune/README.md
```

## Prompt Engineering

### Genre Tags
- Use space-separated tags: `genre instrument mood gender timbre`
- Example: `"inspiring female uplifting pop airy vocal electronic bright vocal"`
- See [top_200_tags.json](top_200_tags.json) for recommended tags
- Use "Mandarin" or "Cantonese" tags for Chinese languages

### Lyrics
- Structure with labels: `[verse]`, `[chorus]`, `[bridge]`, `[outro]`
- Separate sessions with double newlines `\n\n`
- Keep each session around 30s (avoid too many words)
- Start with `[verse]` or `[chorus]` (avoid `[intro]`)
- See [prompt_egs/lyrics.txt](prompt_egs/lyrics.txt) for examples

### Audio Prompts (ICL)
- Optional but improves quality
- Dual-track ICL (vocal + instrumental) works best
- Use ~30s chorus sections for best results
- Extract tracks with [python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator)

## Windows Users

- **One-click installer**: [Pinokio](https://pinokio.computer)
- **Docker/Gradio**: [YuE-for-Windows](https://github.com/sdbds/YuE-for-windows)

## GUI Interfaces

- [YuE-UI by joeljuvel](https://github.com/joeljuvel/YuE-UI) - Batch generation, timeline, save/load
- [YuE-extend by Mozer](https://github.com/Mozer/YuE-extend) - Music continuation + Colab
- [YuE-exllamav2-UI](https://github.com/WrongProtocol/YuE-exllamav2-UI)
- [YuEGP](https://github.com/deepbeepmeep/YuEGP)

## Roadmap

- [ ] Release zen-musician base model
- [ ] Release genre-specific LoRA adapters
- [ ] Support llama.cpp/GGUF quantization
- [ ] HuggingFace Space demo
- [ ] vLLM/sglang inference support
- [ ] Stem generation mode
- [ ] Additional genre training

## License

Zen Musician is released under the **Apache License 2.0**, inheriting from the original YuE project.

### Attribution
- When using this model, please credit: "Zen Musician (based on YuE by HKUST/M-A-P)"
- We encourage commercial use while maintaining proper attribution
- Label AI-generated content as "AI-generated" or "AI-assisted"

### Original YuE Citation

```bibtex
@misc{yuan2025yuescalingopenfoundation,
      title={YuE: Scaling Open Foundation Models for Long-Form Music Generation},
      author={Ruibin Yuan and Hanfeng Lin and Shuyue Guo and Ge Zhang and Jiahao Pan and Yongyi Zang and Haohe Liu and Yiming Liang and Wenye Ma and Xingjian Du and Xinrun Du and Zhen Ye and Tianyu Zheng and Zhengxuan Jiang and Yinghao Ma and Minghao Liu and Zeyue Tian and Ziya Zhou and Liumeng Xue and Xingwei Qu and Yizhi Li and Shangda Wu and Tianhao Shen and Ziyang Ma and Jun Zhan and Chunhui Wang and Yatian Wang and Xiaowei Chi and Xinyue Zhang and Zhenzhu Yang and Xiangzhou Wang and Shansong Liu and Lingrui Mei and Peng Li and Junjie Wang and Jianwei Yu and Guojian Pang and Xu Li and Zihao Wang and Xiaohuan Zhou and Lijun Yu and Emmanouil Benetos and Yong Chen and Chenghua Lin and Xie Chen and Gus Xia and Zhaoxiang Zhang and Chao Zhang and Wenhu Chen and Xinyu Zhou and Xipeng Qiu and Roger Dannenberg and Jiaheng Liu and Jian Yang and Wenhao Huang and Wei Xue and Xu Tan and Yike Guo},
      year={2025},
      eprint={2503.08638},
      archivePrefix={arXiv},
      primaryClass={eess.AS},
      url={https://arxiv.org/abs/2503.08638},
}
```

## Acknowledgements

Built on [YuE by HKUST/M-A-P](https://github.com/multimodal-art-projection/YuE). We thank the original authors and all contributors to the open-source music generation community.

Part of the **[Zen AI](https://github.com/zenlm)** ecosystem - building frontier AI models for creativity and expression.

## Links

- **GitHub**: https://github.com/zenlm/zen-musician
- **HuggingFace**: https://huggingface.co/zenlm/zen-musician-7b
- **Organization**: https://github.com/zenlm
- **Zen Gym** (Training): https://github.com/zenlm/zen-gym
- **Zen Engine** (Inference): https://github.com/zenlm/zen-engine

---

**Zen Musician** - Transforming lyrics into music with AI