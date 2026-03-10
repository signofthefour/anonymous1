

# MamTra: Hardware-Friendly Speech Synthesis

Welcome to the official repository for the MamTra paper!  
MamTra leverages the Mamba architecture for efficient, high-quality speech synthesis. This project is designed to be accessible, but some hardware and installation challenges may arise. Don’t hesitate to reach out for support!

> **⚠️ Note:**  
> MamTra uses Mamba, which is designed to be hardware-friendly and efficient. However, you may encounter hardware or installation issues (especially with CUDA and dependencies). If you run into trouble, please raise your problem in the Community tab — I’m happy to help!

## 🚀 Quickstart

### 1. Environment Setup

- **Python 3.10** is required (for CosyVoice’s ttsfrd).  
	We recommend creating a new environment to avoid conflicts:

	```bash
	conda create -n mamtra python=3.10 -y
	conda activate mamtra
	```

- **CUDA 11.6+** is needed for Mamba.  
	If your machine doesn’t support this, you can install CUDA via conda (this bypassing works for my project):

	```bash
	conda install nvidia/label/cuda-11.8.0::cuda-nvcc
	conda install nvidia/label/cuda-11.8.0::cuda
	```

	For CuDNN:
	```bash
	conda install anaconda::cudnn
	# or, if that fails:
	conda install -c conda-forge cudnn
	```

    If you use this bypassing, please update the `LD_LIBRARY_PATH` as shown in the [scripts/uv_inference_hybrid_11.sh](scripts/uv_inference_hybrid_11.sh).

### 2. Install Dependencies

- We recommend using [uv](https://github.com/astral-sh/uv) for dependency management.  
	Installing mamba, causal_conv1d, and CosyVoice’s requirements can cause version conflicts (especially with CUDA).  
	uv helps keep things smooth!

	```bash
	pip install -U uv
	uv sync
	uv sync --extra compile  # Needed to run the hybrid
	```
---

## Inference

### 0. Download Checkpoint
Currently, only the MamTra 1:1 checkpoint is available due to time constraints. You can download it with:
```
huggingface-cli download blueskyheaven/mamtra11 \
  --local-dir huggingface_model_dir \
  --local-dir-use-symlinks False
```
Other variants will be released soon.

After downloading, set `huggingface_model_dir` as the `MODEL_DIR` variable in the script you intend to use.

### 1. Single Sample Inference
To run inference on a single sample, use the `single_inference.sh` script.

### 2. Multiple Sample Inference
To run inference on multiple samples, use the `scripts/uv_inference_hybrid_11.sh` script.
