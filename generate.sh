output_dir=output/example1

# Step1: text to panorama image
python code/panoramic_image_generation.py \
    --mode=t2p \
    --prompt="a vibrant, industrial-style scene, featuring a large warehouse with exposed brick walls, metal beams, and scattered barrels and crates, set against a backdrop of modern skyscrapers and lush greenery" \
    --output_path=$output_dir

# Or you can choose image to panorama image generation
# python code/panoramic_image_generation.py \
#     --mode=i2p \
#     --input_image_path="./data/image2.jpg" \
#     --output_path=$output_dir

# Step2: panorama image to video generation
VISIBLE_GPU_NUM=1
torchrun --nproc_per_node ${VISIBLE_GPU_NUM} code/panoramic_image_to_video.py \
  --inout_dir=$output_dir  \
  --resolution=720

# Step3: 3d scene extraction
python code/panoramic_video_to_3DScene.py \
    --inout_dir=$output_dir \
    --resolution=720
