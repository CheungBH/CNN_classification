import os
import shutil
from train_result.config import task_folder, batch_folder

folder_name = "{}-{}".format(task_folder, batch_folder)
src_folder = os.path.join("../weight", folder_name)
dest_folder = os.path.join("../result", folder_name)
os.makedirs(dest_folder, exist_ok=True)

for folder in os.listdir(src_folder):
    result_folder = os.path.join(src_folder, folder, folder)
    if os.path.exists(os.path.join(result_folder, "loss.jpg")):
        try:
            shutil.copytree(result_folder, os.path.join(dest_folder, folder))
        except FileExistsError:
            pass


