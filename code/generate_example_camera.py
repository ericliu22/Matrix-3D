import os
import numpy as np
import json
frame_size = 81

def generate_cameras(direction="front"):
    all_cams = []
    for i in range(frame_size):
        if direction == "front":
            cur_cam_position = np.array([0,0,-float(i)/(frame_size-1) * 1000.])
        elif direction == "back":
            cur_cam_position = np.array([0,0,float(i)/(frame_size-1) * 1000.])
        elif direction == "left":
            cur_cam_position = np.array([float(i)/(frame_size-1) * 1000.,0.,0])
        elif direction == "right":
            cur_cam_position = np.array([-float(i)/(frame_size-1) * 1000.,0.,0])
        cur_cam = np.eye(4)
        cur_cam[:3,3] = cur_cam_position
        all_cams.append(cur_cam)
    all_cams = np.stack(all_cams)
    all_cams_list = all_cams.tolist()
    with open(f"./test_cam_{direction}.json","w") as F_:
        F_.write(json.dumps(all_cams_list,indent=4))
if __name__ == "__main__":
    generate_cameras("front")
    generate_cameras("back")
    generate_cameras("left")
    generate_cameras("right")

