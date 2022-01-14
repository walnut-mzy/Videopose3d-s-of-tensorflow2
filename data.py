import numpy as np
import hashlib
def data_process(poses_valid_2d):
    train_video=[]
    for i in poses_valid_2d:
        for k in range(len(i)):
            if k+243<len(i):
                train_video.append(i[k:243+k, ::])
            else:
                train_video.append(i[-243:, ::])
    return train_video

def deterministic_random(min_value, max_value, data):
    digest = hashlib.sha256(data.encode()).digest()
    raw_value = int.from_bytes(digest[:4], byteorder='little', signed=False)
    return int(raw_value / (2**32 - 1) * (max_value - min_value)) + min_value