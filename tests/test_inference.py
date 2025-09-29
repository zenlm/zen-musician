#!/usr/bin/env python3
"""
Zen Musician Inference Tests
Tests basic inference functionality for the music generation model.
"""

import os
import sys
import pytest
import torch
from pathlib import Path

# Add inference directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "inference"))

from transformers import AutoModelForCausalLM, AutoTokenizer


class TestZenMusicianInference:
    """Test suite for Zen Musician inference"""

    @pytest.fixture(scope="class")
    def model_path(self):
        """Get model path from environment or use default"""
        return os.getenv("ZEN_MUSICIAN_MODEL", "m-a-p/YuE-s1-7B-anneal-en-cot")

    @pytest.fixture(scope="class")
    def model_and_tokenizer(self, model_path):
        """Load model and tokenizer once for all tests"""
        print(f"Loading model: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        return model, tokenizer

    def test_model_loading(self, model_and_tokenizer):
        """Test that model and tokenizer load successfully"""
        model, tokenizer = model_and_tokenizer
        assert model is not None, "Model failed to load"
        assert tokenizer is not None, "Tokenizer failed to load"
        assert model.config.model_type in ["qwen2", "llama"], "Unexpected model type"

    def test_tokenizer_basics(self, model_and_tokenizer):
        """Test basic tokenizer functionality"""
        _, tokenizer = model_and_tokenizer

        text = "Hello, Zen Musician!"
        tokens = tokenizer.encode(text)

        assert len(tokens) > 0, "Tokenization produced no tokens"
        assert isinstance(tokens, list), "Tokens should be a list"

        decoded = tokenizer.decode(tokens)
        assert decoded is not None, "Decoding failed"

    def test_simple_generation(self, model_and_tokenizer):
        """Test simple text generation"""
        model, tokenizer = model_and_tokenizer

        prompt = "[verse]\nSinging in the moonlight\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        assert generated_text is not None, "Generation produced no output"
        assert len(generated_text) > len(prompt), "Generated text should be longer than prompt"
        print(f"Generated: {generated_text[:100]}...")

    def test_genre_conditioning(self, model_and_tokenizer):
        """Test generation with genre tags"""
        model, tokenizer = model_and_tokenizer

        # Test with genre tags
        genre = "rock powerful male guitar"
        lyrics = "[verse]\nBreaking through the walls\n"
        prompt = f"{genre}\n\n{lyrics}"

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=30,
            temperature=0.8,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        assert generated_text is not None, "Genre-conditioned generation failed"

    def test_batch_generation(self, model_and_tokenizer):
        """Test batch generation"""
        model, tokenizer = model_and_tokenizer

        prompts = [
            "[verse]\nFirst song lyrics\n",
            "[chorus]\nSecond song lyrics\n"
        ]

        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

        assert outputs.shape[0] == len(prompts), "Batch size mismatch"

        for i, output in enumerate(outputs):
            text = tokenizer.decode(output, skip_special_tokens=True)
            assert text is not None, f"Batch generation failed for prompt {i}"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_inference(self, model_and_tokenizer):
        """Test inference on CUDA device"""
        model, tokenizer = model_and_tokenizer

        assert model.device.type == "cuda", "Model should be on CUDA"

        prompt = "[verse]\nTest CUDA inference\n"
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )

        assert outputs.device.type == "cuda", "Outputs should be on CUDA"

    def test_memory_efficiency(self, model_and_tokenizer):
        """Test that model fits in available memory"""
        model, _ = model_and_tokenizer

        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            memory_reserved = torch.cuda.memory_reserved() / 1024**3  # GB

            print(f"GPU Memory Allocated: {memory_allocated:.2f} GB")
            print(f"GPU Memory Reserved: {memory_reserved:.2f} GB")

            # Model should fit in 24GB for fp16
            assert memory_allocated < 24, "Model uses too much memory"


@pytest.mark.slow
class TestZenMusicianEndToEnd:
    """End-to-end tests for full song generation pipeline"""

    def test_full_song_generation(self):
        """Test full song generation with multiple sections"""
        pytest.skip("Requires full inference pipeline setup")

    def test_dual_track_icl(self):
        """Test dual-track in-context learning"""
        pytest.skip("Requires audio processing setup")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])