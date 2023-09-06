from torch.utils.tensorboard import SummaryWriter

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pickle

from path import Path
import time
import datetime
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from dataset.ds_utils import mpath2info
from elaborator import Elaborator
import yaml
import pyodbc
import pandas as pd


class CreationHandler(FileSystemEventHandler):

    def __init__(self, output_path, proj_log_path, input_path):
        self.output_path = Path(output_path)
        self.input_path = Path(input_path)
        self.current_date = datetime.date.today()
        # self.current_date = datetime.date(year=2022, month=9, day=5)  # TODO: remove
        self.infos = []
        self.elaborator_1 = Elaborator(cam_id=1, proj_log_path=proj_log_path)
        self.elaborator_2 = Elaborator(cam_id=2, proj_log_path=proj_log_path)
        self.elaborator_3 = Elaborator(cam_id=3, proj_log_path=proj_log_path)
        self.first_iteration = True

        # SQL
        server = 'SPREPWKSIND001'
        database = 'UNIMORE'
        username = 'sa'
        password = 'spalbrushless'
        self.cnxn = pyodbc.connect(
            'DRIVER={SQL Server Native Client 11.0};SERVER=' + server + ';DATABASE=' + database + ';UID=' + username + ';PWD=' + password)
        self.cursor = self.cnxn.cursor()

    def on_moved(self, event):

        curr_filename = Path(event.dest_path)
        print(f'modified file {curr_filename}')
        curr_info = mpath2info(curr_filename)
        date = curr_info['datetime'].date()
        datetime = curr_info['datetime']

        if self.first_iteration:
            for filename in self.input_path.files('*.bmp'):
                info = mpath2info(filename)
                if info['datetime'] < datetime and \
                        info['original_name'].endswith('OK.bmp') or info['original_name'].endswith('KO.bmp'):
                    self.infos.append(info)

        if date > self.current_date or self.first_iteration:
            print(f'Starting elaboration of {len(self.infos)} images...')
            if len(self.infos) > 0:
                self.start_sequential_elaboration()
                # print('ELABORATION!')
            print('Done!')
            print('Moving tested images...')
            for info in self.infos:
                Path(info['original_name']).move(self.output_path)
            print('Done!')
            print('Ready to serve...')
            self.current_date = date
            print('Current date is: ', self.current_date)
            self.infos = []

        # Add only OK.bmp|KO.bmp
        if curr_info['original_name'].endswith('OK.bmp') or curr_info['original_name'].endswith('KO.bmp'):
            self.infos.append(curr_info)
        else:
            print(f'ignored file {curr_filename}')

        self.first_iteration = False

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
        # with open('filename.pickle', 'rb') as handle:
        #     daily_results_cam_1 = pickle.load(handle)
        # daily_results_cam_2 = daily_results_cam_1.copy()
        # daily_results_cam_3 = daily_results_cam_1.copy()
        #
        print('SQL loading...')

        for file_name_1 in daily_results_cam_1:

            anomaly_score_cam_1 = daily_results_cam_1[file_name_1]
            file_name_2 = file_name_1.split('_')[0] + '_' + file_name_1.split('_')[1] + '_2' + file_name_1.split('_')[
                                                                                                   2][1:]
            anomaly_score_cam_2 = daily_results_cam_2[file_name_2]
            file_name_3 = file_name_1.split('_')[0] + '_' + file_name_1.split('_')[1] + '_3' + file_name_1.split('_')[
                                                                                                   2][1:]
            anomaly_score_cam_3 = daily_results_cam_3[file_name_3]

            class_cam_1 = self.score_to_label(anomaly_score_cam_1)
            class_cam_2 = self.score_to_label(anomaly_score_cam_2)
            class_cam_3 = self.score_to_label(anomaly_score_cam_3)
            class_total = 'OK' if (class_cam_1 == 'OK' and class_cam_2 == 'OK' and class_cam_3 == 'OK') else 'KO'

            info = mpath2info(file_name_1)

            esito_op = 'unknown'
            if info['label'] == 'bad':
                esito_op = 'KO'
            elif info['label'] == 'good':
                esito_op = 'OK'

            res = {}
            # Id int IDENTITY(1,1) NOT NULL
            res['TS'] = info['datetime'].strftime('%Y%m%d %H:%M:%S')  # TS datetime NOT NULL
            res['UID'] = info['uid']  # UID varchar(24) NOT NULL
            res['WO'] = None  # WO varchar(50)  NULL
            res['PRG'] = None  # PRG varchar(50)  NULL
            res['IOP'] = None  # IOP varchar(50)  NULL
            res['ATXXXX'] = '3514'  # ATXXXX varchar(50) NULL
            res['ESITO'] = class_total  # ESITO varchar(50) NOT NULL
            res['ESITO_OP'] = esito_op  # ESITO varchar(50) NOT NULL
            res['ESITO_P1'] = class_cam_1  # ESITO_P1 varchar(50) NOT NULL
            res['ESITO_P2'] = class_cam_2  # ESITO_P2 varchar(50) NOT NULL
            res['ESITO_P3'] = class_cam_3  # ESITO_P2 varchar(50) NOT NULL
            res['ANOMALY_SCORE_P1'] = str(anomaly_score_cam_1)  # ANOMALY_SCORE_P1 varchar(50) NOT NULL
            res['ANOMALY_SCORE_P2'] = str(anomaly_score_cam_2)  # ANOMALY_SCORE_P2 varchar(50) NOT NULL
            res['ANOMALY_SCORE_P3'] = str(anomaly_score_cam_3)  # ANOMALY_SCORE_P3 varchar(50) NOT NULL

            self.cursor.execute(
                'INSERT INTO dbo.esiti (TS, UID, WO, PRG, IOP, ATXXXX, ESITO, ESITO_OP, ESITO_P1, ESITO_P2, ESITO_P3, '
                'ANOMALY_SCORE_P1, ANOMALY_SCORE_P2, ANOMALY_SCORE_P3) VALUES '
                f'(\'{res["TS"]}\', \'{res["UID"]}\', NULL, NULL, NULL, \'{res["ATXXXX"]}\', \'{res["ESITO"]}\', \'{res["ESITO_OP"]}\','
                f'\'{res["ESITO_P1"]}\', \'{res["ESITO_P2"]}\', \'{res["ESITO_P3"]}\', \'{res["ANOMALY_SCORE_P1"]}\', \'{res["ANOMALY_SCORE_P2"]}\', \'{res["ANOMALY_SCORE_P3"]}\')')
            # commit the transaction
            self.cnxn.commit()

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

    event_handler = CreationHandler(output_path=output_path, proj_log_path=proj_log_path, input_path=input_path)
    observer = Observer()
    observer.schedule(event_handler, path=input_path, recursive=False)
    observer.start()
    print('Ready to serve...')

    try:
        while True:
            time.sleep(0.001)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


if __name__ == "__main__":
    main()
