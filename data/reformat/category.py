import pandas as pd
import os
import shutil
import json

def create_folders(path):
    if not os.path.isdir(path):
        os.mkdir(path)

if __name__ == "__main__":

    data_root = "../data/"
    # files = os.listdir(data_root)
    # print("original files ", len(files))
    
    # class_records = {}
    # infos = pd.read_csv("label.csv")
    # # print(infos)
    # # print(infos.columns)
    # for row_id, row in infos.iterrows():
    #     ### read information
    #     label = row['category']
    #     filename = row['filename']
    #     ### create folder
    #     save_root = data_root + str(label) + "/"
    #     create_folders(save_root)
    #     shutil.move(data_root + filename, save_root + filename)

    #     if str(label) not in class_records:
    #         class_records[str(label)] = None

    # print("the number of classes ", len(class_records))
    # 
    
    """
    original files  2190
    the number of classes  219
    """

    folders = os.listdir(data_root)
    print()
    print("[CHECK]")
    print("the number of classes ", len(folders))
    msg = ""
    total = 0
    for i in range(len(folders)):
        f = str(i)
        num_imgs = len(os.listdir(data_root + f))
        msg += "c{}:{}\n".format(f, num_imgs)
        total += num_imgs
    with open("num_samples.txt", "w") as fjson:
        fjson.write(msg)
    print("files ", total)
