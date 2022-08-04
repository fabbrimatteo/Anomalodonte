import pickle

from torch.utils.tensorboard import SummaryWriter
from path import Path
import time
import datetime
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from dataset.ds_utils import mpath2info
from elaborator import Elaborator
import yaml


class CreationHandler(FileSystemEventHandler):

    def __init__(self, output_path, proj_log_path):
        self.output_path = output_path
        self.current_date = datetime.date.today()
        self.current_date = datetime.date(year=2022, month=6, day=15)  # TODO: remove
        self.infos = []
        self.elaborator_1 = Elaborator(cam_id=1, proj_log_path=proj_log_path)
        # self.elaborator_2 = Elaborator(cam_id=2, proj_log_path=proj_log_path)
        # self.elaborator_3 = Elaborator(cam_id=3, proj_log_path=proj_log_path)


    def on_created(self, event):
        filename = Path(event.src_path)
        print(f'created file {filename}')
        curr_info = mpath2info(filename)
        date = curr_info['datetime'].date()
        if date > self.current_date:
            self.start_sequential_elaboration()
            for info in self.infos:
                Path(info['original_name']).move(self.output_path)
            self.current_date = date
            self.infos = []

        self.infos.append(curr_info)


    def start_sequential_elaboration(self):
        cam_1_infos = [info for info in self.infos if info['camera-id'] == 1]
        cam_2_infos = [info for info in self.infos if info['camera-id'] == 2]
        cam_3_infos = [info for info in self.infos if info['camera-id'] == 3]
        cam_1_infos.sort(key=lambda d: d['datetime'])
        cam_2_infos.sort(key=lambda d: d['datetime'])
        cam_3_infos.sort(key=lambda d: d['datetime'])

        print('starting elaboration cam_1')
        daily_results_cam_1 = self.elaborator_1.start(infos=cam_1_infos)
        print('starting elaboration cam_2')
        daily_results_cam_2 = self.elaborator_2.start(infos=cam_2_infos)
        print('starting elaboration cam_3')
        daily_results_cam_3 = self.elaborator_3.start(infos=cam_3_infos)

        # with open('filename.pickle', 'wb') as handle:
        #     pickle.dump(daily_results_cam_1, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # exit()
        print('SQL loading...')

        for file_name in daily_results_cam_1:
            anomaly_score_cam_1 = daily_results_cam_1[file_name]
            anomaly_score_cam_2 = daily_results_cam_2[file_name]
            anomaly_score_cam_3 = daily_results_cam_3[file_name]
            class_cam_1 = self.score_to_label(anomaly_score_cam_1)
            class_cam_2 = self.score_to_label(anomaly_score_cam_1)
            class_cam_3 = self.score_to_label(anomaly_score_cam_1)
            class_total = 'OK' if (class_cam_1 == 'OK' and class_cam_2 == 'OK' and class_cam_3 == 'OK') else 'KO'

            info = mpath2info(file_name)

            res = {}
            # Id int IDENTITY(1,1) NOT NULL
            res['TS'] = info['datetime'] # TS datetime NOT NULL
            res['UID'] = info['uid'] # UID varchar(24) NOT NULL
            res['WO'] = None # WO varchar(50)  NULL
            res['PRG'] = None # PRG varchar(50)  NULL
            res['IOP'] = None # IOP varchar(50)  NULL
            res['ATXXXX'] = 3514 # ATXXXX varchar(50) NULL
            res['ESITO'] = class_total # ESITO varchar(50) NOT NULL
            res['ESITO_P1'] = class_cam_1 # ESITO_P1 varchar(50) NOT NULL
            res['ESITO_P2'] = class_cam_2 # ESITO_P2 varchar(50) NOT NULL
            res['ESITO_P3'] = class_cam_3 # ESITO_P2 varchar(50) NOT NULL
            res['ANOMALY_SCORE_P1'] = anomaly_score_cam_1 # ANOMALY_SCORE_P1 varchar(50) NOT NULL
            res['ANOMALY_SCORE_P2'] = anomaly_score_cam_2 # ANOMALY_SCORE_P2 varchar(50) NOT NULL
            res['ANOMALY_SCORE_P3'] = anomaly_score_cam_3 # ANOMALY_SCORE_P3 varchar(50) NOT NULL


    def score_to_label(self, score):
        label = 'OK'
        if 50 <= score < 150:
            label = 'KO'
        elif score >= 150:
            label = 'NC'
        return label

def main():
    conf_file_path = 'conf/paths.yaml'
    conf_file = open(conf_file_path, 'r')
    y = yaml.load(conf_file, Loader=yaml.Loader)
    input_path = y.get('INPUT_PATH', None)
    output_path = y.get('OUTPUT_PATH', None)
    proj_log_path = y.get('PROJ_LOG_PATH', None)

    event_handler = CreationHandler(output_path=output_path, proj_log_path=proj_log_path)
    observer = Observer()
    observer.schedule(event_handler, path=input_path, recursive=False)
    observer.start()
    print('ready to serve...')

    try:
        while True:
            time.sleep(0.001)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


if __name__ == "__main__":
    main()
