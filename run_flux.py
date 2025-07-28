import argparse
import os
import torch
import time
from datetime import datetime
from PIL import Image
import multiprocessing

# For SD1.5 img2img
from diffusers import StableDiffusionImg2ImgPipeline

# Import your Flux Schnell inference module
# (Assuming it's in flux_schnell_infer.py – adjust if different)
from flux_schnell_infer import generate_flux_image  # <-- You must have this function from original Flux Schnell code

def main():
    parser = argparse.ArgumentParser(description="Run Flux Schnell txt2img OR SD1.5 img2img")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt for image generation")
    parser.add_argument("--init_image", type=str, default=None, help="Path to initial image for img2img")
    parser.add_argument("--strength", type=float, default=0.6, help="Img2img strength (0.0-1.0, lower = closer to original)")
    parser.add_argument("--output", type=str, default=None, help="Output image file name")
    parser.add_argument("--height", type=int, default=1024, help="Image height in pixels")
    parser.add_argument("--width", type=int, default=1024, help="Image width in pixels")
    parser.add_argument("--steps", type=int, default=4, help="Steps for Flux Schnell (ignored for SD1.5)")
    parser.add_argument("--guidance_scale", type=float, default=3.5, help="Guidance scale")
    parser.add_argument("--threads", type=int, default=8, help="Manual CPU thread count")
    parser.add_argument("--autotune", action="store_true", help="Auto-set optimal threads")
    parser.add_argument("--flux_model_path", type=str, default=os.path.expanduser("~/flux_schnell_cpu/flux_schnell_local"), help="Path to Flux Schnell model")
    parser.add_argument("--sd_model_path", type=str, default=os.path.expanduser("~/SD1.5"), help="Path to SD1.5 model")
    parser.add_argument("--output_dir", type=str, default="~/FluxImages/", help="Output directory")

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

    # ✅ Output filename
    if args.output:
        output_path = os.path.join(output_dir, args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_prompt = args.prompt[:40].strip().replace(" ", "_").replace("/", "_").lower()
        output_path = os.path.join(output_dir, f"{timestamp}_{safe_prompt}.png")

    start_time = time.time()

    if args.init_image:
        # ✅ SD 1.5 img2img
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
            guidance_scale=7.5,
            num_inference_steps=20
        ).images[0]

    else:
        # ✅ Flux Schnell txt2img
        print("🎨 Using Flux Schnell txt2img mode")
        image = generate_flux_image(
            model_path=args.flux_model_path,
            prompt=args.prompt,
            height=args.height,
            width=args.width,
            steps=args.steps,
            guidance_scale=args.guidance_scale,
            autotune=args.autotune
        )

    image.save(output_path)
    end_time = time.time()
    print(f"✅ Image saved to: {output_path}")
    print(f"🕒 Generation time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
