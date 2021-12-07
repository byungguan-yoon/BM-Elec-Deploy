from threading import Thread
from connect_light import lightConnector
from melsec_connector import MelsecConnector
import os
import datetime 
from py_image_buffer_save import create_device, start_stream, destroy_device, drop_buffer
import time
from mongo import Mongo
import pdb
# from loguru import logger
from inference import load_model, down_num_hline, v_sum
from pprint import pprint

def on_camera_position_change(camera_no: int, camera_position: int):
    print("Camera No {}, Camera Position {}".format(camera_no, camera_position))

# @logger.catch
def inspection_loop(model):
    # log 파일 생성, 매일 00시에 새로운 로그 파일 생성
    # logger.add("./log/all.log", rotation="00:00")
    # logger.add("./log/error.log", rotation="00:00", level = "ERROR")
    # logger.add("./log/info.log", rotation="00:00", level = "INFO")
    
    # file path: inspection start time
    path = path_tmp + f"/{datetime.datetime.now().strftime('%H_%M_%S')}"
    os.makedirs(path, exist_ok=True)

    # 조명별 검사
    for k in range(1,5):
        light_connector.change_state(f'{k}','OFF') # 1번 조명 OFF
    light_connector2.change_state('2','2ON') # 2번 조명 ON
    light_connector2.change_state('2','040') # 2번 조명 밝기 40으로 조정
    # Count Product
    pre_h_line = None
    h_repre_list, h_repre_tf_list = list(), list()
    
    repre_list = list()
    h_line_list = list()

    # 섹션별 검사
    device = create_device()
    start_stream(device)
    for section_id in range(1,56 + 1):
        while True:
            # 검사하려는 섹션에 카메라가 위치하는지 Check
            if section_id == connector.plc2pc_get_val(headdevice="D10001")[0]:
                # logger.debug(connector.plc2pc_get_val(headdevice="D10001")[0])
                
                
                section_path = path + f"/s_{section_id}"
                os.makedirs(section_path, exist_ok=True)
                # device = create_device()
                # with device.start_stream(1):
                drop_buffer(device)
                # logger.info(f"START: {section_id} Shot")
                # section_id_str = section_id_str.zfill(2)
                # 촬영 -> 분석 -> 결과 전송
                pre_h_line, h_repre_list, h_repre_tf_list, repre_list, h_line_list = connector.process_snapshot(device, section_path, mongo, model, pre_h_line, h_repre_list, h_repre_tf_list, repre_list, h_line_list)
                # logger.info(f"END: {section_id} Shot")
                # destroy_device()
                break
            time.sleep(0.1)
    else:
        h_repre_list[-1] = h_repre_list[-1][:-1]
        h_repre_tf_list[-1] = h_repre_tf_list[-1][:-1]

    init_pro = down_num_hline(h_line_list)

    pprint(h_repre_tf_list)
    pprint(init_pro)
    result = v_sum(h_repre_tf_list, init_pro)
    print(f"Result: {result}")

    destroy_device()

if __name__ == '__main__':
    try:
        connector = MelsecConnector("192.168.10.7", 2001)

        # light control
        port = ['/dev/ttyUSB0', '/dev/ttyUSB1']
        baud = 9600
        light_connector = lightConnector(port[0], baud)
        light_connector2 = lightConnector(port[1], baud)

        # mongodb
        mongo = Mongo('localhost', 27017, 'bm_elec', 'inspection')
        mongo.connect_init()

        # load dl model
        model = load_model()

        path_tmp = f"../visualize/static/result/{datetime.date.today()}"
        os.makedirs(path_tmp, exist_ok=True)
        
        # while True:
        inspection_loop(model)
        time.sleep(1)
    except ConnectionRefusedError:
        # logger.error('Connection Error')
        pass







