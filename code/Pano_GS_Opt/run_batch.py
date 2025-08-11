import os
import sys


training_iters=6000 # optimization iterations
num_of_point_cloud=3000000 # number of point cloud unprojected from depth map
num_views_per_view=3 # 相邻两个相机位姿之间插针数目
img_sample_interval=1 # 训练时每隔多少张图片选取用于优化3DGS

#base_dir = "/mnt/workspace/zhongqi.yang/VideoInpainting_new/ViewCrafter/output_recon"
'''
base_dir = "/ai-video-sh/zhongqi.yang/code/zhongqi.yang/VideoInpainting_new/datasets/pano_data_for_gs_new"
all_cases = os.listdir(base_dir)
for c in all_cases[1:]:
    input_folder = os.path.join(base_dir, c)
    output_folder = os.path.join("output/scene_test", "pano_"+c.split(",")[0])
    os.system(f"python train.py -s {input_folder} -m {output_folder} -r 1 --use_decoupled_appearance --save_iterations 1000 6000 9000 12000 15000 --test_iterations 1000 \
    --sh_degree 0 --densify_from_iter 500 --densify_until_iter 1501 --iterations {training_iters} --eval \
    --img_sample_interval {img_sample_interval} --num_views_per_view {num_views_per_view} --num_of_point_cloud {num_of_point_cloud} \
    ")
    os.system(f"python render.py -m {output_folder} --iteration {training_iters}")
    #os.system(f"cd {output_folder}/test/ours_6000/test_preds_1;ffmpeg -framerate 12 -i %05d.png -c:a copy -c:v ayuv render.avi")
    #os.system(f"cd {output_folder}/test/ours_6000/depth_1;ffmpeg -framerate 12 -i %05d.png -c:a copy -c:v ayuv render.avi")
    #cmd = "ffmpeg -framerate 60 -i %05d.png render.mp4"
    #print(cmd)
    # os.system("ffmpeg -framerate 60 -i %05d.png render.mp4")
'''
input_dir = "/ai-video-sh/zhongqi.yang/code/zhongqi.yang/Trajcrafter_Training/output/A_boat_on_lake,_trees_and_rocks_near_the_lake._a_house_and_port_in_front_of_a_house,_anime_style_superres/geom_optim/data"
output_folder = os.path.join("output/scene_recons", "pano_boat_on_lake_1e0")
os.system(f"python train.py -s {input_dir} -m {output_folder} -r 1 --use_decoupled_appearance --save_iterations 1000 6000 9000 12000 15000 --test_iterations 1000 \
    --sh_degree 0 --densify_from_iter 500 --densify_until_iter 1501 --iterations {training_iters} --eval \
    --img_sample_interval {img_sample_interval} --num_views_per_view {num_views_per_view} --num_of_point_cloud {num_of_point_cloud} --device {'cuda:1'} \
    ")