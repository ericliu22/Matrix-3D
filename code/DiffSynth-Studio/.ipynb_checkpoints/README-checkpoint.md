# 数据准备
按照以下格式准备好视频数据集

data/example_dataset/
├── metadata.csv
└── train
    ├── video_00001.mp4
    └── image_00002.jpg
metadata.csv:

file_name,text
video_00001.mp4,"video description"
image_00002.jpg,"video description"


# lora 训练
运行 lora_train.sh 进行数据前处理和lora训练

# 视频生成推理
运行lora_infer.py 进行推理生成视频