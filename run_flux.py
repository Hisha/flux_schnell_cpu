import argparse
import os
import torch
import time
from datetime import datetime
from PIL import Image
import multiprocessing
from diffusers import StableDiffusionImg2ImgPipeline

# -------------------------------
# ✅ Flux Schnell Direct Import
# -------------------------------
from flux import FluxPipeline  # This comes from your flux_schnell_local install

def main():
    parser = argparse.ArgumentParser(description="Run Flux Schnell (txt2img) or SD 1.5 (img2img)")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt for image generation")
    parser.add_argument("--init_image", type=str, default=None, help="Path to initial image for img2img")
    parser.add_argument("--strength", type=float, default=0.6, help="Strength for img2img (lower = closer to original)")
    parser.add_argument("--output", type=str, default=None, help="Output image file name")
    parser.add_argument("--height", type=int, default=1024, help="Image height in pixels")
    parser.add_argument("--width", type=int, default=1024, help="Image width in pixels")
    parser.add_argument("--steps", type=int, default=4, help="Number of inference steps (Flux Schnell)")
    parser.add_argument("--guidance_scale", type=float, default=3.5, help="Guidance scale")
    parser.add_argument("--threads", type=int, default=8, help="Manual thread count")
    parser.add_argument("--autotune", action="store_true", help="Auto-select optimal threads")
    parser.add_argument("--flux_model_path", type=str, default=os.path.expanduser("~/flux_schnell_cpu/flux_schnell_local"), help="Path to Flux Schnell model folder")
    parser.add_argument("--sd_model_path", type=str, default=os.path.expanduser("~/SD1.5"), help="Path to SD1.5 model folder")
    parser.add_argument("--output_dir", type=str, default="~/FluxImages", help="Directory to save outputs to")

    args = parser.parse_args()

    # ✅ Threads
    if args.autotune:
        logical_cores = multiprocessing.cpu_count()
        tuned_threads = max(4, int(logical_cores * 0.75))
        torch.set_num_threads(tuned_threads)
        print(f"🧠 Auto-tuned threads: {tuned_threads} of {logical_cores}")
    else:
        torch.set_num_threads(args.threads)
        print(f"🧠 Using manual thread count: {args.threads}")

    # ✅ Prepare output dir
    output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # ✅ Output file name
    if args.output:
        output_path = os.path.join(output_dir, args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_prompt = args.prompt[:40].strip().replace(" ", "_").replace("/", "_").lower()
        output_path = os.path.join(output_dir, f"{timestamp}_{safe_prompt}.png")

    start = time.time()

    # ✅ Choose pipeline
    if args.init_image:
        # -----------------------------
        # ✅ SD1.5 img2img mode
        # -----------------------------
        print("🖼️ Using SD 1.5 img2img mode")
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

    else:
        # -----------------------------
        # ✅ Flux Schnell txt2img mode
        # -----------------------------
        print("🎨 Using Flux Schnell txt2img mode")
        pipe = FluxPipeline.from_pretrained(
            args.flux_model_path,
            torch_dtype=torch.float32
        )
        pipe.to("cpu")

        image = pipe(
            prompt=args.prompt,
            height=args.height,
            width=args.width,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale
        ).images[0]

    # ✅ Save result
    image.save(output_path)
    end = time.time()

    print(f"✅ Image saved to: {output_path}")
    print(f"🕒 Generation time: {end - start:.2f} seconds")

if __name__ == "__main__":
    main()
