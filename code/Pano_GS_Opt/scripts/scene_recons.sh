input_folder=/mnt/workspace/zhongqi.yang/VideoInpainting_new/datasets/pano_convert/a_small_town
#input_folder=/mnt/workspace/zhongqi.yang/VideoInpainting_new/ViewCrafter/output_recon/20250303_1157_chicken_gun
#input_folder=/mnt/workspace/zhongqi.yang/VideoInpainting_new/ViewCrafter/output_recon/20250303_1210_tmp
#input_folder=/mnt/workspace/zhongqi.yang/VideoInpainting_new/ViewCrafter/output_recon/20250303_1219_york_city
output_folder=output/scene_recons/pano_small_town
training_iters=6000 # optimization iterations
# num_of_point_cloud=5000000 # number of point cloud unprojected from depth map
num_of_point_cloud=3000000 # number of point cloud unprojected from depth map
num_views_per_view=2 # 相邻两个相机位姿之间插针数目
img_sample_interval=1 # 训练时每隔多少张图片选取用于优化3DGS
# -r 1 用全部的分辨率训练
# --sh_degree 0 颜色不随视角发生变化

#python train.py -s $input_folder -m $output_folder -r 1 --use_decoupled_appearance --save_iterations 3000 --test_iterations 3000 \
#--sh_degree 0 --densify_from_iter $training_iters --iterations $training_iters --eval \
#--img_sample_interval $img_sample_interval --num_views_per_view $num_views_per_view --num_of_point_cloud $num_of_point_cloud

python train.py -s $input_folder -m $output_folder -r 1 --use_decoupled_appearance --save_iterations 3000 6000 9000 12000 15000 --test_iterations 3000 \
--sh_degree 0 --densify_from_iter 1500 --iterations $training_iters --eval \
--img_sample_interval $img_sample_interval --num_views_per_view $num_views_per_view --num_of_point_cloud $num_of_point_cloud

python render.py -m $output_folder --iteration $training_iters

cd $output_folder/test/ours_6000/test_preds_1

ffmpeg -framerate 60 -i %05d.png render.mp4
