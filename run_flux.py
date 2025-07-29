import argparse
import os
import torch
import time
from datetime import datetime
import subprocess
import multiprocessing
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline

def main():
    parser = argparse.ArgumentParser(description="Run Flux Schnell (txt2img) or SD1.5 (img2img)")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--init_image", type=str, default=None)
    parser.add_argument("--strength", type=float, default=0.6)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--guidance_scale", type=float, default=3.5)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--autotune", action="store_true")
    parser.add_argument("--flux_script", type=str, default="/home/smithkt/flux_schnell_cpu/inference_cpu_txt2img.py")
    parser.add_argument("--flux_model_path", type=str, default="/home/smithkt/flux_schnell_cpu/flux_schnell_local")
    parser.add_argument("--sd_model_path", type=str, default="/home/smithkt/SD1.5")
    parser.add_argument("--output_dir", type=str, default="~/FluxImages/")

    args = parser.parse_args()

    # Thread tuning
    if args.autotune:
        logical_cores = multiprocessing.cpu_count()
        tuned_threads = max(4, int(logical_cores * 0.75))
        torch.set_num_threads(tuned_threads)
        print(f"🧠 Auto-tuned threads: {tuned_threads} of {logical_cores}")
    else:
        torch.set_num_threads(args.threads)
        print(f"🧠 Using manual thread count: {args.threads}")

    output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Output filename
    if args.output:
        output_path = os.path.join(output_dir, args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_prompt = args.prompt[:40].strip().replace(" ", "_").replace("/", "_").lower()
        output_path = os.path.join(output_dir, f"{timestamp}_{safe_prompt}.png")

    start = time.time()

    if args.init_image:
        print("🖼️ Using SD1.5 img2img mode")
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            args.sd_model_path,
            torch_dtype=torch.float32,
            local_files_only=True
        )
        pipe.to("cpu")
        pipe.enable_attention_slicing()

        init_image = Image.open(args.init_image).convert("RGB").resize((args.width, args.height))

        image = pipe(
            prompt=args.prompt,
            image=init_image,
            strength=args.strength,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.steps
        ).images[0]

        image.save(output_path)

    else:
        print("🎨 Using Flux Schnell txt2img mode")
        cmd = [
            "python", args.flux_script,
            "--prompt", args.prompt,
            "--output", output_path,
            "--model_path", args.flux_model_path,
            "--steps", str(args.steps),
            "--guidance", str(args.guidance_scale),
            "--height", str(args.height),
            "--width", str(args.width)
        ]
        subprocess.run(cmd, check=True)

    end = time.time()
    print(f"✅ Image saved to: {output_path}")
    print(f"🕒 Generation time: {end - start:.2f} seconds")

if __name__ == "__main__":
    main()
