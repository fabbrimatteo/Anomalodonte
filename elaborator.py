from trainer import Trainer
from time import sleep

import cv2
from path import Path
import os
import numpy as np

from dataset import Emitter
from dataset.ds_utils import cut_full_img
from models.autoencoder_plus import AutoencoderPlus
from conf import Conf
from eval.lof import Loffer
from visual_utils import draw_anomaly_ui
from update_proc.day_db import DayDB


class Elaborator(object):

    def __init__(self, cam_id):
        self.cam_id = cam_id
        self.exp_name = Path(f'cam_{cam_id}')
        self.proj_log_path = Path('/goat-nas/Datasets/spal/progression_demo/experiments')
        self.pth_file_path = Path('/home/matteo/PycharmProjects/Anomalodonte/log/progression/last.pth') # TODO: different for each cam
        self.dataset_path = Path('/goat-nas/Datasets/spal/progression_demo/start_sets') # TODO: different for each cam
        self.yaml_file_path = Path('conf/progression.yaml')
        self.log_dir_path = self.proj_log_path / self.exp_name / 'log'
        self.daily_res_path = self.proj_log_path / self.exp_name / 'daily_res'

        self.daily_results = {}

        self.cnf = Conf(exp_name=self.exp_name, proj_log_path=self.proj_log_path)
        self.n_neighbors = 20
        self.train_buffer_size = 4000 # 10000

        self.cnf.exp_log_path.makedirs_p()

        # copying train and test sets
        print('Copying train and test sets...')
        cmd = f'cp -r "{self.dataset_path.abspath()}/train" "{self.cnf.exp_log_path.abspath()}"'
        os.system(cmd)
        cmd = f'cp -r "{self.dataset_path.abspath()}/test" "{self.cnf.exp_log_path.abspath()}"'
        os.system(cmd)
        print('Done!')

        self.current_date = None
        self.model = None
        self.loffer = None
        self.day_db = None



    def run(self, infos):
        self.current_date = infos[0]['datetime'].date().strftime('%Y%m%d')
        self.cnf = Conf(exp_name=self.exp_name, proj_log_path=self.proj_log_path)

        today_res_path = self.daily_res_path / f'{self.current_date}'
        today_res_path.makedirs_p()

        # init autoencoder
        self.model = AutoencoderPlus.init_from_pth(
            self.pth_file_path,
            device=self.cnf.device, mode='eval'
        )

        print('Initializing Loffer...')
        self.loffer = Loffer(
            train_dir=self.cnf.exp_log_path / 'train',
            model=self.model, n_neighbors=self.n_neighbors
        )

        self.day_db = DayDB(root_dir=self.cnf.exp_log_path, debug=False,
                            date=self.current_date, cam=self.cam_id,
                            train_buffer_size=self.train_buffer_size)

        self.daily_results = {}
        for info in infos:
            frame = cv2.imread(info['original_name'])
            frame_name = Path(info['original_name']).basename()
            cut = cut_full_img(img=frame, cam_name=f'cam_{self.cam_id}', side=256)
            anomaly_score = int(round(self.loffer.get_anomaly_score(cut)))
            self.daily_results = self.day_db.add(img_cut=cut, anomaly_score=anomaly_score,
                            cut_name=frame_name)

            img_ui = draw_anomaly_ui(cut, anomaly_score)
            cv2.imwrite(today_res_path / frame_name, img_ui)

            print(f'───$> sample #{frame_name} of day #{self.current_date}: anomaly score {anomaly_score}')

    def end_day_routine(self):

        exp_name_day_pretrain = self.exp_name + f'_{self.current_date}_pretrain'
        exp_name_day = self.exp_name + f'_{self.current_date}'

        self.day_db.update_dataset_all()

        # pretraining
        self.cnf = Conf(exp_name=exp_name_day_pretrain,
                   proj_log_path=self.log_dir_path,
                   yaml_file_path=self.yaml_file_path)
        self.cnf.ds_path = self.proj_log_path / self.exp_name
        self.cnf.epochs = 2
        self.cnf.pretrained_weights_path = self.pth_file_path
        trainer = Trainer(cnf=self.cnf)
        trainer.run()

        # init autoencoder
        pth_file_path_pretrain = self.log_dir_path / exp_name_day_pretrain / 'last.pth'
        self.model = AutoencoderPlus.init_from_pth(
            pth_file_path_pretrain,
            device=self.cnf.device, mode='eval'
        )

        print('Initializing Loffer after Pretraining...')
        self.loffer = Loffer(
            train_dir=self.proj_log_path / self.exp_name / 'train',
            model=self.model, n_neighbors=self.n_neighbors
        )
        train_set = self.loffer.get_train_labels()

        out_split_dir_good = self.log_dir_path / exp_name_day_pretrain / 'good'
        out_split_dir_bad = self.log_dir_path / exp_name_day_pretrain / 'bad'
        out_split_dir_good.makedirs_p()
        out_split_dir_bad.makedirs_p()
        for sample in train_set:
            old_path = sample[0]
            name, label = old_path.basename().split('.')[0], sample[1]
            if name in list(self.day_db.info.keys()):
                if label == -1:
                    new_path = out_split_dir_bad / name + '.jpg'
                elif label == 1:
                    new_path = out_split_dir_good / name + '.jpg'

                cmd = f'cp "{old_path}" "{new_path}"'
                os.system(cmd)
                print(f'───$> {cmd}')

                # TODO: move into test set?
                if label == -1:
                    cmd = f'rm "{old_path}"'
                    os.system(cmd)
                    print(f'───$> {cmd}')

        # training
        self.cnf = Conf(exp_name=exp_name_day,
                   proj_log_path=self.log_dir_path,
                   yaml_file_path=self.yaml_file_path)
        self.cnf.ds_path = self.proj_log_path / self.exp_name
        self.cnf.pretrained_weights_path = self.pth_file_path
        self.cnf.epochs = 2 # TODO: remove!!
        trainer = Trainer(cnf=self.cnf)
        trainer.run()

        self.pth_file_path = Path(self.cnf.exp_log_path / 'best.pth')  # TODO: or last?


    def start(self, infos):
        self.run(infos)
        self.end_day_routine()

        return self.daily_results