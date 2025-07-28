import argparse
import os
import torch
import time
from datetime import datetime
from diffusers import DiffusionPipeline, StableDiffusionImg2ImgPipeline
from PIL import Image
import multiprocessing

def main():
    parser = argparse.ArgumentParser(description="Run FLUX.1-schnell-Free image generation")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt to generate image from")
    parser.add_argument("--init_image", type=str, default=None, help="Path to initial image for img2img")
    parser.add_argument("--strength", type=float, default=0.6, help="Strength for img2img (0.0-1.0, lower = closer to original)")
    parser.add_argument("--output", type=str, default=None, help="Output image file name")
    parser.add_argument("--height", type=int, default=1024, help="Image height in pixels (default 1024)")
    parser.add_argument("--width", type=int, default=1024, help="Image width in pixels (default 1024)")
    parser.add_argument("--steps", type=int, default=4, help="Number of inference steps (default 4)")
    parser.add_argument("--guidance_scale", type=float, default=3.5, help="Classifier-free guidance scale")
    parser.add_argument("--threads", type=int, default=8, help="Manual thread count (ignored if --autotune is used)")
    parser.add_argument("--autotune", action="store_true", help="Auto-select optimal thread count for this machine")
    parser.add_argument("--model_path", type=str, default=os.path.expanduser("~/flux_schnell_cpu/flux_schnell_local/"),
                        help="Path to local model folder")
    parser.add_argument("--output_dir", type=str, default="~/FluxImages/", help="Directory to save outputs to")

    args = parser.parse_args()

    # ✅ Thread optimization
    if args.autotune:
        logical_cores = multiprocessing.cpu_count()
        tuned_threads = max(4, int(logical_cores * 0.75))
        torch.set_num_threads(tuned_threads)
        print(f"🧠 Auto-tuned threads: {tuned_threads} of {logical_cores}")
    else:
        torch.set_num_threads(args.threads)
        print(f"🧠 Using manual thread count: {args.threads}")

    # ✅ Prepare output directory
    output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # ✅ Build output path
    if args.output:
        output_path = os.path.join(output_dir, args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_prompt = args.prompt[:40].strip().replace(" ", "_").replace("/", "_").lower()
        output_path = os.path.join(output_dir, f"{timestamp}_{safe_prompt}.png")

    # ✅ Load appropriate pipeline
    if args.init_image:
        print("🖼️ Using img2img mode")
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            args.model_path,
            torch_dtype=torch.float32,
            use_safetensors=True
        )
    else:
        print("🎨 Using txt2img mode")
        pipe = DiffusionPipeline.from_pretrained(
            args.model_path,
            torch_dtype=torch.float32,
            use_safetensors=True
        )

    pipe.to("cpu")
    pipe.enable_attention_slicing()

    start = time.time()
    print(f"⏳ Generating image for prompt: {args.prompt}")

    if args.init_image:
        init_image = Image.open(args.init_image).convert("RGB").resize((args.width, args.height))
        image = pipe(
            prompt=args.prompt,
            image=init_image,
            strength=args.strength,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.steps
        ).images[0]
    else:
        image = pipe(
            args.prompt,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            height=args.height,
            width=args.width
        ).images[0]

    end = time.time()

    image.save(output_path)
    print(f"✅ Image saved to: {output_path}")
    print(f"🕒 Generation time: {end - start:.2f} seconds")

if __name__ == "__main__":
    main()
