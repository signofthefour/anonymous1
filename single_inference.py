import sys
import os
import argparse
from setuptools import logging
import torchaudio
from rich.console import Console
import torch
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
import logging

logging.getLogger("filelock").setLevel(logging.WARNING)

sys.path.append("third_party/Matcha-TTS")

from cosyvoice.cli.frontend import CosyVoiceFrontEnd
from cosyvoice.cli.cosyvoice import CosyVoice2, CosyVoice2Model
from cosyvoice.utils.file_utils import load_wav
from cosyvoice.utils.class_utils import get_model_type
from cosyvoice.bin.hybrid import prepare_student_model
from analysis_code.hybrid.utils import HybridQwen2ForCausalLM
from hyperpyyaml import load_hyperpyyaml
from modelscope import snapshot_download

console = Console()


class CosyVoice2Hybrid(CosyVoice2):
    def __init__(
        self,
        model_dir,
        load_jit=False,
        load_trt=False,
        load_vllm=False,
        fp16=False,
        prunned=False,
        num_layers_after_prunned=0,
        attn_layers=None,
    ):
        self.instruct = True if "-Instruct" in model_dir else False
        self.model_dir = model_dir
        self.fp16 = fp16

        if not os.path.exists(model_dir):
            console.log(
                f"[yellow]Model directory {model_dir} not found locally. Please read the README.md again and download the model or replacing the correct dir to MODEL_DIR...[/yellow]"
            )

        hyper_yaml_path = f"{model_dir}/cosyvoice2.yaml"
        if not os.path.exists(hyper_yaml_path):
            raise ValueError(f"{hyper_yaml_path} not found!")

        with open(hyper_yaml_path, "r") as f:
            configs = load_hyperpyyaml(
                f,
                overrides={
                    "qwen_pretrain_path": os.path.join(model_dir, "CosyVoice-BlankEN")
                },
            )

        assert get_model_type(configs) == CosyVoice2Model

        self.frontend = CosyVoiceFrontEnd(
            configs["get_tokenizer"],
            configs["feat_extractor"],
            f"{model_dir}/campplus.onnx",
            f"{model_dir}/speech_tokenizer_v2.onnx",
            f"{model_dir}/spk2info.pt",
            configs["allowed_special"],
        )
        self.sample_rate = configs["sample_rate"]

        # Hybrid LLM replacement
        custom_qwen_model = HybridQwen2ForCausalLM.from_existing(
            configs["llm"].llm.model, map_location=configs["llm"].llm.model.device
        )
        configs["llm"].llm.model = custom_qwen_model

        # Optional pruning
        if prunned:
            if num_layers_after_prunned >= len(configs["llm"].llm.model.model.layers):
                console.log(
                    "[yellow]Pruning request ignored — layer count too high[/yellow]"
                )
            else:
                console.log(
                    f"Pruning LLM from {len(configs['llm'].llm.model.model.layers)} → {num_layers_after_prunned} layers"
                )
                configs["llm"].llm.model.model.layers = configs[
                    "llm"
                ].llm.model.model.layers[:num_layers_after_prunned]

        hybrid_model = prepare_student_model(
            (
                configs["llm"].llm.model.model
                if not hasattr(configs["llm"], "module")
                else configs["llm"].module.llm.model.model
            ),
            (
                configs["llm"].llm.model.model.config
                if not hasattr(configs["llm"], "module")
                else configs["llm"].module.llm.model.model.config
            ),
            attn_layers=attn_layers,
        )

        if hasattr(configs["llm"], "module"):
            configs["llm"].module.llm.model.model = hybrid_model
        else:
            configs["llm"].llm.model.model = hybrid_model

        self.model = CosyVoice2Model(
            configs["llm"], configs["flow"], configs["hift"], fp16
        )

        console.log(f"Loading model parts from {model_dir}")
        self.model.load(
            f"{model_dir}/llm.pt",
            f"{model_dir}/flow.pt",
            f"{model_dir}/hift.pt",
            show_weight_compare=False,
        )

        # Important for Mamba-based models
        self.model.llm.llm.model.model.allocate_mamba_inference_cache(1, 2048)


def parse_arguments():
    parser = argparse.ArgumentParser(description="CosyVoice2 single sample inference")
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--ref-text", type=str, required=False, help="Reference text")
    parser.add_argument("--text", type=str, required=True, help="Text to synthesize")
    parser.add_argument(
        "--ref-audio", type=str, required=True, help="Reference audio path (.wav)"
    )
    parser.add_argument(
        "--output", type=str, default="output_single.wav", help="Output filename"
    )
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--apply-pruning", action="store_true")
    parser.add_argument(
        "--num-layers", type=int, default=0, help="Layers after pruning (if pruning)"
    )
    parser.add_argument(
        "--attn-layers",
        type=int,
        nargs="+",
        default=[],
        help="Layers where hybrid attention is applied",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    console.rule("[bold cyan]CosyVoice2 — Single Sample Inference[/bold cyan]")

    # Load model
    cosyvoice = CosyVoice2Hybrid(
        args.model_dir,
        fp16=args.fp16,
        prunned=args.apply_pruning,
        num_layers_after_prunned=args.num_layers,
        attn_layers=args.attn_layers,
    )

    console.log("[green]Model loaded.[/green]")
    console.log(f"Reference audio:  {args.ref_audio}")
    console.log(f"Text to synthesize:\n  [white]{args.text}[/white]")

    # Load reference audio
    try:
        audio_prompt = load_wav(args.ref_audio, 16000)
    except Exception as e:
        console.log(f"[bold red]Failed to load reference audio:[/bold red] {e}")
        sys.exit(1)

    if audio_prompt.shape[-1] > 30 * 16000:
        console.log("[red]Reference audio is longer than 30 seconds → truncated.[/red]")
        audio_prompt = audio_prompt[:, : 30 * 16000]

    # Inference
    console.rule("[yellow]Running inference[/yellow]")

    # We take the first generated sample only (usually there's only one anyway)
    results = cosyvoice.inference_zero_shot(
        args.text,
        args.ref_text,
        audio_prompt,
        stream=False,
    )

    if not results:
        console.log("[bold red]No output generated.[/red]")
        return

    output_audio = next(results)["tts_speech"]

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torchaudio.save(args.output, output_audio, cosyvoice.sample_rate)

    console.rule("[bold green]Done[/bold green]")
    console.log(
        f"Saved to: [bright_green]{os.path.abspath(args.output)}[/bright_green]"
    )
    console.log(f"Sample rate: {cosyvoice.sample_rate} Hz")


if __name__ == "__main__":
    main()
