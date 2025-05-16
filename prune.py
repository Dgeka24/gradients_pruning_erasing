from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel, StableDiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer
import torch
from torchvision import transforms
import PIL
import torch.nn.functional as F
import gc
from tqdm import tqdm
import os
import argparse


basemodel_id = "CompVis/stable-diffusion-v1-4"


def calc_gradients(device, concept_to_erase, num_iters=1):
    # load pipe for image generation
    pipe = StableDiffusionPipeline.from_pretrained(basemodel_id, safety_checker = None)
    pipe = pipe.to(device)

    # load models for gradient calculation
    noise_scheduler = DDPMScheduler.from_pretrained(basemodel_id, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(
        basemodel_id, subfolder="tokenizer",
    )
    weight_dtype = torch.float32
    text_encoder = CLIPTextModel.from_pretrained(
        basemodel_id, subfolder="text_encoder",
    ).to(device, dtype=weight_dtype)
    vae = AutoencoderKL.from_pretrained(
        basemodel_id, subfolder="vae",
    ).to(device, dtype=weight_dtype)
    unet = UNet2DConditionModel.from_pretrained(
        basemodel_id, subfolder="unet",
    ).to(device)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.train()

    inputs = tokenizer(
        f"a photo of {concept_to_erase}", max_length=tokenizer.model_max_length,
        padding="max_length", truncation=True, return_tensors="pt",
    )
    train_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    tokens_input = inputs.input_ids
    tokens_input = tokens_input.to(device)
    encoder_hidden_states = text_encoder(tokens_input, return_dict=False)[0]

    for i in tqdm(range(num_iters)):
        image_input = train_transforms(pipe(f"a photo of {concept_to_erase}").images[0]).to(device).unsqueeze(dim=0)
        latents = vae.encode(image_input.to(weight_dtype)).latent_dist.sample()
        latents = latents * vae.config.scaling_factor
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]

        timesteps = torch.randint(noise_scheduler.config.num_train_timesteps // 4,
                                  noise_scheduler.config.num_train_timesteps // 2,
                                  (bsz,), device=latents.device,)
        timesteps = timesteps.long()

        target = noise
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
        model_pred = unet(noisy_latents, timesteps, encoder_hidden_states, return_dict=False)[0]
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        loss.backward()

    grads_for_concept = {}
    for name, params in unet.named_parameters():
        grads_for_concept[name] = params.grad.detach().cpu().clone()

    del loss
    del pipe
    del unet
    del vae
    del text_encoder
    gc.collect()
    torch.cuda.empty_cache()
    return grads_for_concept


def prune_model(device, save_path, concept_to_erase):

    grads_for_concept = calc_gradients(device, concept_to_erase)

    unet = UNet2DConditionModel.from_pretrained(
        basemodel_id, subfolder="unet",
    ).to(device)
    state_dict = unet.state_dict()

    for param_name, params in tqdm(state_dict.items()):
        # if "attn" not in param_name or "bias" in param_name:
        #     continue
        if torch.numel(params) < 5000000:
            continue
        rel_diff = (grads_for_concept[param_name]).abs()
        k = int(torch.numel(grads_for_concept[param_name]) * 0.9985)
        threshold = torch.kthvalue(rel_diff.flatten(), k).values
        state_dict[param_name][rel_diff > threshold] = 0.0
    unet.load_state_dict(state_dict)
    os.makedirs(save_path, exist_ok=True)
    torch.save(unet.state_dict(), save_path + "/unet.pt")

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prune_model(device, "save_models", "ship")
