import sys
import os
import argparse
import statistics
import torchaudio
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TimeElapsedColumn,
    TextColumn,
)
from rich.table import Table
import torch
import logging

from rich.logging import RichHandler

# Configure logging
logging.basicConfig(
    level=logging.WARNING,  # show WARNING and above
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler()],
)

logging = logging.getLogger("rich")

from hyperpyyaml import load_hyperpyyaml
from modelscope import snapshot_download

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

sys.path.append("third_party/Matcha-TTS")
from cosyvoice.cli.frontend import CosyVoiceFrontEnd
from cosyvoice.cli.cosyvoice import CosyVoice2, CosyVoice2Model
from cosyvoice.utils.file_utils import load_wav
from cosyvoice.utils.class_utils import get_model_type

from cosyvoice.bin.hybrid import prepare_student_model

from analysis_code.hybrid.utils import HybridQwen2ForCausalLM

console = Console()

# TODO: fix this hard code
SEED_TTS_EVAL_PATH = "Seed-tts-eval/seedtts_testset/en"


class CosyVoice2Hybrid(CosyVoice2):
    def __init__(
        self,
        model_dir,
        load_jit=False,
        load_trt=False,
        load_vllm=False,
        fp16=False,
        trt_concurrent=1,
        prunned=False,
        num_layers_after_prunned=0,
        attn_layers=None,  # This one is used for specifying which attention
    ):
        self.instruct = True if "-Instruct" in model_dir else False
        self.model_dir = model_dir
        self.fp16 = fp16
        if not os.path.exists(model_dir):
            model_dir = snapshot_download(model_dir)
        hyper_yaml_path = "{}/cosyvoice2.yaml".format(model_dir)
        if not os.path.exists(hyper_yaml_path):
            raise ValueError("{} not found!".format(hyper_yaml_path))
        with open(hyper_yaml_path, "r") as f:
            configs = load_hyperpyyaml(
                f,
                overrides={
                    "qwen_pretrain_path": os.path.join(model_dir, "CosyVoice-BlankEN")
                },
            )
        assert (
            get_model_type(configs) == CosyVoice2Model
        ), "do not use {} for CosyVoice2 initialization!".format(model_dir)
        self.frontend = CosyVoiceFrontEnd(
            configs["get_tokenizer"],
            configs["feat_extractor"],
            "{}/campplus.onnx".format(model_dir),
            "{}/speech_tokenizer_v2.onnx".format(model_dir),
            "{}/spk2info.pt".format(model_dir),
            configs["allowed_special"],
        )
        self.sample_rate = configs["sample_rate"]
        if torch.cuda.is_available() is False and (
            load_jit is True or load_trt is True or fp16 is True
        ):
            load_jit, load_trt, fp16 = False, False, False
            logging.warning("no cuda device, set load_jit/load_trt/fp16 to False")

        # Replace Qwen2ForCausalLM with HybridQwen2ForCausalLM
        console.log("LLM architecture before hybrid:", type(configs["llm"].llm.model))
        console.log("Replacing Qwen2ForCausalLM with HybridQwen2ForCausalLM...")
        custom_qwen_model = HybridQwen2ForCausalLM.from_existing(
            configs["llm"].llm.model, map_location=configs["llm"].llm.model.device
        )
        # replace the model with custom model
        configs["llm"].llm.model = custom_qwen_model
        console.log("Replaced LLM architecture:", type(configs["llm"].llm.model))
        ##################################################################################
        # delete custom_qwen_model to free memory
        # Prunning Logic
        if prunned is True:
            assert (
                num_layers_after_prunned > 0
            ), "num_layers_after_prunned must be greater than 0 when prunned is True"
            assert num_layers_after_prunned < len(
                configs["llm"].llm.model.model.layers
            ), "num_layers_after_prunned must be less than llm.num_layers"
            num_layers_before_prunned = len(configs["llm"].llm.model.model.layers)
            logging.info(
                "prunning CosyVoice2 model from {} layers to {} layers".format(
                    num_layers_before_prunned, num_layers_after_prunned
                )
            )
            llm = configs["llm"]
            transformer_layers = llm.llm.model.model.layers
            if num_layers_after_prunned >= len(transformer_layers):
                logging.warning(
                    "num_layers_after_prunned {} is greater than or equal to the number of layers {}, using all layers".format(
                        num_layers_after_prunned, len(transformer_layers)
                    )
                )
            else:
                llm.llm.model.model.layers = transformer_layers[
                    :num_layers_after_prunned
                ]
                logging.info(
                    "prunned CosyVoice2 model to {} layers".format(
                        num_layers_after_prunned
                    )
                )
            configs["llm"] = llm
        else:
            logging.info("No prunning applied to CosyVoice2 model")

        if hasattr(configs["llm"], "module"):
            qwen_configs = configs["llm"].module.llm.model.model.config
        else:
            qwen_configs = configs["llm"].llm.model.model.config

        hybrid_model = prepare_student_model(
            (
                configs["llm"].module.llm.model.model
                if hasattr(configs["llm"], "module")
                else configs["llm"].llm.model.model
            ),
            qwen_configs,
            attn_layers=attn_layers,
        )

        # Replace the model with hybrid model
        if hasattr(configs["llm"], "module"):
            configs["llm"].module.llm.model.model = hybrid_model
        else:
            configs["llm"].llm.model.model = hybrid_model

        self.model = CosyVoice2Model(
            configs["llm"], configs["flow"], configs["hift"], fp16
        )

        console.log("Loading model from {}".format(model_dir))
        console.log("Loading llm from {}/llm.pt".format(model_dir))
        console.log("Loading flow from {}/flow.pt".format(model_dir))
        console.log("Loading hift from {}/hift.pt".format(model_dir))
        self.model.load(
            "{}/llm.pt".format(model_dir),
            "{}/flow.pt".format(model_dir),
            "{}/hift.pt".format(model_dir),
            show_weight_compare=False,
        )

        self.model.llm.llm.model.model.allocate_mamba_inference_cache(1, 2048)

        del configs


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="CosyVoice2 inference script with GPU monitoring"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Directory containing pretrained model checkpoints",
    )
    parser.add_argument(
        "--metadata-file",
        type=str,
        default="libritts_test_metadata.txt",
        help="Path to metadata file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory to save output audio files",
    )
    parser.add_argument("--gpu-idx", type=int, default=2, help="GPU index to monitor")
    parser.add_argument(
        "--apply-pruning", action="store_true", help="Apply model pruning"
    )
    parser.add_argument(
        "--num-layers", type=int, default=24, help="Number of layers after pruning"
    )
    parser.add_argument(
        "--load-jit", action="store_true", help="Load JIT-compiled model"
    )
    parser.add_argument("--load-trt", action="store_true", help="Load TensorRT model")
    parser.add_argument("--load-vllm", action="store_true", help="Load vLLM model")
    parser.add_argument("--fp16", action="store_true", help="Use FP16 precision")

    # list of layers to apply hybrid attention
    parser.add_argument(
        "--attn-layers",
        type=int,
        nargs="+",
        default=[],
        help="List of layer indices to apply hybrid attention",
    )

    # Log all the arguments
    args = parser.parse_args()
    print(f"Arguments: {args}")
    return args


def load_metadata(metadata_file):
    try:
        with open(metadata_file, "r") as f:
            return [line.strip().split("|") for line in f if line.strip()]
    except FileNotFoundError:
        raise FileNotFoundError(f"Metadata file {metadata_file} not found")


# def log_gpu_memory(handle):
#     mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
#     return mem_info.used / 1024**2  # MB


def save_audio_output(filename, output_dir, audio_data, sample_rate=24000, index=0):
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/{filename}"
    if index > 0:
        base, ext = os.path.splitext(output_path)
        output_path = f"{base}_{index}{ext}"
    torchaudio.save(output_path, audio_data, sample_rate)
    return output_path


def print_summary(console, gpu_usages, rtfs):
    # GPU Memory Summary
    table = Table(title="GPU Memory Usage Summary")
    table.add_column("Metric", justify="left", style="cyan", no_wrap=True)
    table.add_column("Memory (MB)", justify="right", style="magenta")
    table.add_row("Average Used Memory", f"{statistics.mean(gpu_usages):.1f} MB")
    table.add_row("Peak Used Memory", f"{max(gpu_usages):.1f} MB")
    console.print(table)

    # RTF Summary
    if rtfs:
        rtf_table = Table(title="RTF Summary")
        rtf_table.add_column("Metric", justify="left", style="cyan", no_wrap=True)
        rtf_table.add_column("RTF Value", justify="right", style="magenta")
        rtf_table.add_row("Average RTF", f"{statistics.mean(rtfs):.4f}")
        rtf_table.add_row("Max RTF", f"{max(rtfs):.4f}")
        rtf_table.add_row("Min RTF", f"{min(rtfs):.4f}")
        console.print(rtf_table)
    else:
        console.log("[yellow]No RTF data collected.")


def main():
    args = parse_arguments()
    console = Console()

    console.rule("[bold blue]Initializing CosyVoice2 Model")
    console.log("Attention layers for hybrid:", args.attn_layers)
    cosyvoice = CosyVoice2Hybrid(
        args.model_dir,
        load_jit=args.load_jit,
        load_trt=args.load_trt,
        load_vllm=args.load_vllm,
        fp16=args.fp16,
        prunned=args.apply_pruning,
        num_layers_after_prunned=args.num_layers,
        attn_layers=args.attn_layers,
    )
    console.log(f"[green]Model loaded successfully from {args.model_dir}")

    metadata = load_metadata(args.metadata_file)
    # pynvml.nvmlInit()
    # handle = pynvml.nvmlDeviceGetHandleByIndex(args.gpu_idx)
    gpu_usages = []
    rtfs = []

    console.rule("[bold blue]Starting Inference")
    console.log(f"[yellow]Starting inference on {len(metadata)} samples...[/yellow]")
    console.log(f"Stored outputs in [green]{args.output_dir}[/green]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[yellow]Inference...", total=len(metadata))

        for item in metadata:
            if len(item) == 3:  # Old setting with LibriTTS style metadata
                filename, text, audio_prompt_path = item
                text_prompt = None
            else:  # New setting with Seed-tts-eval style metadata
                filename = f"{item[0]}.wav"
                text_prompt = item[1]
                audio_prompt_path = item[2]
                audio_prompt_path = os.path.join(SEED_TTS_EVAL_PATH, audio_prompt_path)
                text = item[3]
            audio_prompt = load_wav(audio_prompt_path, 16000)
            if audio_prompt.shape[-1] > 30 * 16000:
                console.log(
                    f"[red]Skipping {filename}: audio prompt is longer than 30 seconds."
                )
                progress.update(task, advance=1)
                continue
            if text_prompt is None:
                text_prompt_path = audio_prompt_path.replace(".wav", ".normalized.txt")
                try:
                    with open(text_prompt_path, "r") as f:
                        text_prompt = f.read().strip()
                except FileNotFoundError:
                    console.log(
                        f"[red]Skipping {filename}: text prompt file not found."
                    )
                    progress.update(task, advance=1)
                    continue

            for i, output in enumerate(
                cosyvoice.inference_zero_shot(
                    text, text_prompt, audio_prompt, stream=False
                )
            ):
                output_path = save_audio_output(
                    filename, args.output_dir, output["tts_speech"], index=i
                )
                if i > 0:
                    console.log(
                        f"[bold red]Warning:[/] Multiple outputs for {filename} detected. Saved as {output_path}"
                    )

                if "rtf" in output:
                    rtfs.append(output["rtf"])
                    console.log(f"[blue]RTF for {filename}: {output['rtf']:.4f}")
                else:
                    console.log(f"[yellow]No RTF found for {filename}.")

            progress.update(task, advance=1)

    console.rule("[bold green]All files processed successfully!")


if __name__ == "__main__":
    main()
