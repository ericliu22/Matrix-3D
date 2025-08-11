
### 前处理
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python examples/wanvideo/train_wan_t2v.py \
  --task data_process \
  --dataset_path data/citypark \
  --output_path ./models \
  --text_encoder_path "models/Wan-AI/Wan2.1-T2V-14B/models_t5_umt5-xxl-enc-bf16.pth" \
  --vae_path "models/Wan-AI/Wan2.1-T2V-14B/Wan2.1_VAE.pth" \
  --tiled \
  --num_frames 81 \
  --height 512 \
  --width 1024 \
  --dataloader_num_workers 8

  
### lora训练  
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python examples/wanvideo/train_wan_t2v.py \
  --task train \
  --train_architecture lora \
  --dataset_path data/citypark \
  --output_path ./models \
  --dit_path "models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00001-of-00006.safetensors,models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00002-of-00006.safetensors,models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00003-of-00006.safetensors,models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00004-of-00006.safetensors,models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00005-of-00006.safetensors,models/Wan-AI/Wan2.1-T2V-14B/diffusion_pytorch_model-00006-of-00006.safetensors" \
  --steps_per_epoch 500 \
  --max_epochs 30 \
  --learning_rate 1e-4 \
  --lora_rank 16 \
  --lora_alpha 16 \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --accumulate_grad_batches 1 \
  --use_gradient_checkpointing 