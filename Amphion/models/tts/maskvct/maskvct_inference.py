import argparse

import soundfile as sf
import torch

from .maskvct_utils import load_config, load_maskvct_bundle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./models/tts/maskvct/config/maskvct.json")
    parser.add_argument("--source", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", default="generated_maskvct.wav")
    parser.add_argument("--mode", default="all", choices=["all", "spk", "accent"])
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    bundle = load_maskvct_bundle(args.config, device)
    audio = bundle.convert(args.source, args.prompt, mode=args.mode)
    sf.write(args.output, audio, bundle.output_sample_rate)


if __name__ == "__main__":
    main()
