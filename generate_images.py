from diffusers import DiffusionPipeline
import torch
import argparse
import os
torch.enable_grad(False)

def generate_images(basemodel_id, pruned_unet_path, prompt, save_path, device='cuda:0', torch_dtype=torch.float32, guidance_scale = 7.5, num_inference_steps=100, num_samples=10, seed=42):
    pipe = DiffusionPipeline.from_pretrained(basemodel_id, torch_dtype=torch_dtype)
    pipe.unet.load_state_dict(torch.load(pruned_unet_path), strict=False)
    pipe = pipe.to(device)
    os.makedirs(save_path, exist_ok=True)
    generator = torch.Generator().manual_seed(seed)
    for i in range(num_samples):
        pil_image = pipe(prompt,
                          generator=generator,
                          num_inference_steps=num_inference_steps,
                          guidance_scale=guidance_scale).images[0]
        pil_image.save(f"{save_path}/{prompt}_{i}.png")

if __name__=='__main__':
    parser = argparse.ArgumentParser(
                    prog = 'Generate images',
                    description = 'Generate Images using Diffusers Code')
    parser.add_argument('--basemodel_id', help='SD1.4 model to prune', type=str, default='CompVis/stable-diffusion-v1-4', required=False)
    parser.add_argument('--pruned_unet_path', help='Path to pruned UNet weights', type=str, required=False, default='save_models/unet.pt')
    parser.add_argument('--prompt', help='Prompt to generate', type=str, required=True)
    parser.add_argument('--save_path', help='Path to save images', type=str, required=False, default='pruned_unet_images')
    parser.add_argument('--device', help='cuda device to run on', type=str, required=False, default='cuda:0')
    parser.add_argument('--guidance_scale', help='guidance to run eval', type=float, required=False, default=7.5)
    parser.add_argument('--num_samples', help='Number of samples per prompt', type=int, required=False, default=1)
    parser.add_argument('--num_inference_steps', help='Ddim steps of inference used to train', type=int, required=False, default=50)
    parser.add_argument('--seed', help='Random seed for generation', type=int, required=False, default=42)
    args = parser.parse_args()

    basemodel_id = args.basemodel_id
    pruned_unet_path = args.pruned_unet_path
    prompt = args.prompt
    save_path = args.save_path
    device = args.device
    guidance_scale = args.guidance_scale
    num_inference_steps = args.num_inference_steps
    num_samples= args.num_samples
    seed = args.seed

    generate_images(basemodel_id=basemodel_id, pruned_unet_path=pruned_unet_path, prompt=prompt, save_path=save_path, device=device, guidance_scale = guidance_scale, num_inference_steps=num_inference_steps, num_samples=num_samples, seed=seed)