device=0
seed=1024
prompt_path=data/video_prompts.txt
save_dir=data
# Super resolution and frame interpolation
HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=$device python3 scripts/enhance_video_ours.py --version v2 --up_scale 2 --target_fps 20 --noise_aug 100 --solver_mode 'fast' --steps 15 --input_path $save_dir --prompt_path $prompt_path --save_dir $save_dir --suffix enhancement --sr_x2