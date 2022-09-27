import os
import sys
import shutil

directory = ['checkpoints']
file = ["checkpoint.pth.tar", "params.pkl", "stats0.pkl", "train.log", "queue0.pth"]

target_path = "preserved/0.6_lr_500_queue_90_qstart_500_prototype"
if __name__ == "__main__":

    for d in directory:
        shutil.copytree(d, target_path + "/" + d)

    for f in file:
        shutil.copy(f, target_path + "/" + f)

    for d in directory:
        shutil.rmtree(d)

    for f in file:
        os.remove(f)

    print("Move Files Done!")
