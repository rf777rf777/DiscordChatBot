from diffusers.pipelines.stable_diffusion.convert_from_ckpt import download_from_original_stable_diffusion_ckpt

pipeline = download_from_original_stable_diffusion_ckpt(
    checkpoint_path_or_dict="src/meinaunreal_v5/meinaunreal_v5.safetensors",
    original_config_file="src/v1-inference.yaml",  # 配套的 yaml
    from_safetensors=True
)
pipeline.save_pretrained("src/meinaunreal_v5")
