import sys
import os

current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)
sys.path.append(current_directory)

from PIL import Image
import io
import base64
from fastapi import FastAPI, Request
from pydantic import BaseModel
from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from src.unet_hacked_tryon import UNet2DConditionModel
from transformers import (
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
    CLIPTextModel,
    CLIPTextModelWithProjection,
)
from diffusers import DDPMScheduler, AutoencoderKL
from typing import List

import torch
import os
from transformers import AutoTokenizer
import numpy as np
from utils_mask import get_mask_location
from torchvision import transforms
import apply_net
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose
from detectron2.data.detection_utils import convert_PIL_to_numpy, _apply_exif_orientation
from torchvision.transforms.functional import to_pil_image

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def pil_to_binary_mask(pil_image, threshold=0):
    np_image = np.array(pil_image)
    grayscale_image = Image.fromarray(np_image).convert("L")
    binary_mask = np.array(grayscale_image) > threshold
    mask = np.zeros(binary_mask.shape, dtype=np.uint8)
    for i in range(binary_mask.shape[0]):
        for j in range(binary_mask.shape[1]):
            if binary_mask[i, j]:
                mask[i, j] = 1
    mask = (mask * 255).astype(np.uint8)
    output_mask = Image.fromarray(mask)
    return output_mask

base_path = 'yisol/IDM-VTON'

unet = UNet2DConditionModel.from_pretrained(
    base_path,
    subfolder="unet",
    torch_dtype=torch.float16,
)
unet.requires_grad_(False)
tokenizer_one = AutoTokenizer.from_pretrained(
    base_path,
    subfolder="tokenizer",
    revision=None,
    use_fast=False,
)
tokenizer_two = AutoTokenizer.from_pretrained(
    base_path,
    subfolder="tokenizer_2",
    revision=None,
    use_fast=False,
)
noise_scheduler = DDPMScheduler.from_pretrained(base_path, subfolder="scheduler")

text_encoder_one = CLIPTextModel.from_pretrained(
    base_path,
    subfolder="text_encoder",
    torch_dtype=torch.float16,
)
text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
    base_path,
    subfolder="text_encoder_2",
    torch_dtype=torch.float16,
)
image_encoder = CLIPVisionModelWithProjection.from_pretrained(
    base_path,
    subfolder="image_encoder",
    torch_dtype=torch.float16,
)
vae = AutoencoderKL.from_pretrained(
    base_path,
    subfolder="vae",
    torch_dtype=torch.float16,
)

UNet_Encoder = UNet2DConditionModel_ref.from_pretrained(
    base_path,
    subfolder="unet_encoder",
    torch_dtype=torch.float16,
)

parsing_model = Parsing(0)
openpose_model = OpenPose(0)

UNet_Encoder.requires_grad_(False)
image_encoder.requires_grad_(False)
vae.requires_grad_(False)
unet.requires_grad_(False)
text_encoder_one.requires_grad_(False)
text_encoder_two.requires_grad_(False)
tensor_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)

pipe = TryonPipeline.from_pretrained(
    base_path,
    unet=unet,
    vae=vae,
    feature_extractor=CLIPImageProcessor(),
    text_encoder=text_encoder_one,
    text_encoder_2=text_encoder_two,
    tokenizer=tokenizer_one,
    tokenizer_2=tokenizer_two,
    scheduler=noise_scheduler,
    image_encoder=image_encoder,
    torch_dtype=torch.float16,
)
pipe.unet_encoder = UNet_Encoder

def base64_to_pil(base64_str):
    image_data = base64.b64decode(base64_str)
    image = Image.open(io.BytesIO(image_data))
    return image

def pil_to_base64(pil_image):
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    base64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return base64_str

class TryonRequest(BaseModel):
    human_image: str
    garment_image: str
    mask_image: str
    garment_description: str
    use_auto_mask: bool
    use_auto_crop: bool
    denoise_steps: int
    seed: int

class TryonResponse(BaseModel):
    output_image: str
    masked_image: str

app = FastAPI()

@app.post("/tryon", response_model=TryonResponse)
async def tryon(request: TryonRequest):
    human_image = base64_to_pil(request.human_image)
    garment_image = base64_to_pil(request.garment_image)
    mask_image = base64_to_pil(request.mask_image)
    garment_description = request.garment_description
    use_auto_mask = request.use_auto_mask
    use_auto_crop = request.use_auto_crop
    denoise_steps = request.denoise_steps
    seed = request.seed

    openpose_model.preprocessor.body_estimation.model.to(device)
    pipe.to(device)
    pipe.unet_encoder.to(device)

    garment_image = garment_image.convert("RGB").resize((768, 1024))
    human_image_orig = human_image.convert("RGB")

    if use_auto_crop:
        width, height = human_image_orig.size
        target_width = int(min(width, height * (3 / 4)))
        target_height = int(min(height, width * (4 / 3)))
        left = (width - target_width) // 2
        top = (height - target_height) // 2
        right = (width + target_width) // 2
        bottom = (height + target_height) // 2
        cropped_img = human_image_orig.crop((left, top, right, bottom))
        crop_size = cropped_img.size
        human_image = cropped_img.resize((768, 1024))
    else:
        human_image = human_image_orig.resize((768, 1024))

    if use_auto_mask:
        keypoints = openpose_model(human_image.resize((384, 512)))
        model_parse, _ = parsing_model(human_image.resize((384, 512)))
        mask, mask_gray = get_mask_location('hd', "dresses", model_parse, keypoints)
        mask = mask.resize((768, 1024))
    else:
        mask = pil_to_binary_mask(mask_image.convert("RGB").resize((768, 1024)))

    mask_gray = (1 - transforms.ToTensor()(mask)) * tensor_transform(human_image)
    mask_gray = to_pil_image((mask_gray + 1.0) / 2.0)

    human_image_arg = _apply_exif_orientation(human_image.resize((384, 512)))
    human_image_arg = convert_PIL_to_numpy(human_image_arg, format="BGR")

    args = apply_net.create_argument_parser().parse_args(
        ('show', './configs/densepose_rcnn_R_50_FPN_s1x.yaml', './ckpt/densepose/model_final_162be9.pkl', 'dp_segm', '-v', '--opts', 'MODEL.DEVICE', 'cuda')
    )
    pose_img = args.func(args, human_image_arg)
    pose_img = pose_img[:, :, ::-1]
    pose_img = Image.fromarray(pose_img).resize((768, 1024))

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            prompt = "Sophie is wearing " + garment_description
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
            with torch.inference_mode():
                (
                    prompt_embeds,
                    negative_prompt_embeds,
                    pooled_prompt_embeds,
                    negative_pooled_prompt_embeds,
                ) = pipe.encode_prompt(
                    prompt,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=True,
                    negative_prompt=negative_prompt,
                )

                prompt = "a photo of " + garment_description
                negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
                if not isinstance(prompt, List):
                    prompt = [prompt] * 1
                if not isinstance(negative_prompt, List):
                    negative_prompt = [negative_prompt] * 1
                with torch.inference_mode():
                    (
                        prompt_embeds_c,
                        _,
                        _,
                        _,
                    ) = pipe.encode_prompt(
                        prompt,
                        num_images_per_prompt=1,
                        do_classifier_free_guidance=False,
                        negative_prompt=negative_prompt,
                    )

                pose_img = tensor_transform(pose_img).unsqueeze(0).to(device, torch.float16)
                garment_tensor = tensor_transform(garment_image).unsqueeze(0).to(device, torch.float16)
                generator = torch.Generator(device).manual_seed(seed) if seed is not None else None
                images = pipe(
                    prompt_embeds=prompt_embeds.to(device, torch.float16),
                    negative_prompt_embeds=negative_prompt_embeds.to(device, torch.float16),
                    pooled_prompt_embeds=pooled_prompt_embeds.to(device, torch.float16),
                    negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.to(device,torch.float16),
                    num_inference_steps=denoise_steps,
                    generator=generator,
                    strength=1.0,
                    pose_img=pose_img.to(device, torch.float16),
                    text_embeds_cloth=prompt_embeds_c.to(device, torch.float16),
                    cloth=garment_tensor.to(device, torch.float16),
                    mask_image=mask,
                    image=human_image,
                    height=1024,
                    width=768,
                    ip_adapter_image=garment_image.resize((768, 1024)),
                    guidance_scale=2.0,
                )[0]

    if use_auto_crop:
        out_img = images[0].resize(crop_size)
        human_image_orig.paste(out_img, (int(left), int(top)))
        output_image = human_image_orig
    else:
        output_image = images[0]

    output_image_base64 = pil_to_base64(output_image)
    masked_image_base64 = pil_to_base64(mask_gray)

    return TryonResponse(
        output_image=output_image_base64,
        masked_image=masked_image_base64
    )