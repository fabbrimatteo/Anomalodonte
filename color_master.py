

from cc3x3 import apply_color_correction
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from path import Path
from dataset.ds_utils import mpath2info
import time
import cv2
import numpy as np
import json
import yaml
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from piq import ssim
from torchvision import transforms


class CreationHandler(FileSystemEventHandler):
    def __init__(self, master_log_path, master_folder_references, master_th):
        self.master_log_path = Path(master_log_path)
        self.master_folder_references = Path(master_folder_references)
        self.master_th = master_th
        self.cam_ref = {
            '1': cv2.imread(self.master_folder_references / '1.bmp'),
            '2': cv2.imread(self.master_folder_references / '2.bmp'),
            '3': cv2.imread(self.master_folder_references / '3.bmp')
        }
        self.log_filename = self.master_log_path / 'log.json'
        self.log_imgs_path = self.master_log_path / 'imgs'
        self.log_imgs_path.makedirs_p()
        self.log_tb_path = self.master_log_path / 'tensorboard'
        self.log_tb_path.makedirs_p()

        if self.log_filename.exists():
            with open(self.log_filename, 'r') as fp:
                self.log = json.load(fp)
        else:
            self.log = {
                '1': [],
                '2': [],
                '3': []
            }

        self.sw = SummaryWriter(self.log_tb_path)



    # def on_moved(self, event):
    #     curr_filename = Path(event.dest_path)
    def on_created(self, event):
        curr_filename = Path(event.src_path)
        # curr_info = mpath2info(curr_filename)
        # cam_id = str(curr_info['camera-id'])
        cam_id = str(curr_filename.basename().split('.')[0][-1])
        time.sleep(2)
        img = cv2.imread(curr_filename)

        new_name = datetime.now().strftime("%Y%m%d%H%M%S") + '_' + cam_id + '.bmp'
        new_name_res = datetime.now().strftime("%Y%m%d%H%M%S") + '_' + cam_id + '_res.bmp'

        try:
            res = apply_color_correction(img.copy(), self.cam_ref[cam_id])

            img_tensor = transforms.ToTensor()(img).unsqueeze(0)
            res_tensor = transforms.ToTensor()(res).unsqueeze(0)

            val = 1 - ssim(img_tensor, res_tensor, data_range=1.).item()
        except RuntimeError:
            val = -1

        self.log[cam_id].append([new_name, val])

        with open(self.master_log_path / 'log.json', 'w') as fp:
            json.dump(self.log, fp)
        cv2.imwrite(self.log_imgs_path / new_name, img)
        cv2.imwrite(self.log_imgs_path / new_name_res, res)
        Path(curr_filename).remove()

        self.sw.add_scalar(
            tag=f'cam_{cam_id}', global_step=len(self.log[cam_id])-1,
            scalar_value=val
        )
        if val < self.master_th:
            result = 'good'
        else:
            result = 'BAD'
        print(f'{curr_filename.basename()}: {val:0.4f} | {result}')


def main():
    with open('conf/master.yaml', 'r') as f:
        y = yaml.load(f, Loader=yaml.Loader)

    master_folder = y.get('MASTER_FOLDER', None)  # type: str
    master_folder_references = y.get('MASTER_FOLDER_REFERENCES', None)  # type: str
    master_log_path = y.get('MASTER_LOG_PATH', None)  # type: str
    master_th = y.get('MASTER_THRESHOLD', None)  # type: float

    event_handler = CreationHandler(master_log_path, master_folder_references, master_th)
    observer = Observer()
    observer.schedule(event_handler, path=master_folder, recursive=False)
    observer.start()
    print('Ready to serve...')

    try:
        while True:
            time.sleep(0.001)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


if __name__ == '__main__':
    main()
