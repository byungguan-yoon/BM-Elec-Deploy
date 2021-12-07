import time
from pymcprotocol import Type3E
from connection_status import ConnectionStatus
from py_image_buffer_save import grab_img
from inference import inference, h_sum #, del_none_product_repre
# from loguru import logger

class MelsecConnector:
    READ_PLC_ADDR = "D10000"
    POSITION_CAM2_ADDR = "D10001"
    HEART_BIT_ADDR = "D10050"
    SNAP_CAM1_ADDR = "D10051"
    SNAP_CAM2_ADDR = "D10052"
    READY_CAM1_ADDR = "D10053"
    BUSY_CAM1_ADDR = "D10054"
    READY_CAM2_ADDR = "D10055"
    BUSY_CAM2_ADDR = "D10056"
    CAM2_RESULT_ADDR = "D10057"
    NG_RESULT_X = "D10060"
    NG_RESULT_Y = "D10062"
    NG_FINISH = "D10064"
    
    def __init__(self, host, port, plctype="binary"):
        self._host = host
        self._port = port
        self._plctype = plctype
        self.connection_status = ConnectionStatus.DISCONNECTED
        self.snap_signal = 0
        self.current_camera_position = 0
        self.conn_class = self.connect_init()
        self._initialize_plc()

    def connect_init(self):
        conn_class = Type3E()
        conn_class.setaccessopt(self._plctype)
        conn_class.connect(self._host, self._port)
        # logger.debug('connected')
        self.change_connection_status(ConnectionStatus.CONNECTED)
        # logger.debug('after connection status changed')

        return conn_class

    def _initialize_plc(self):
        # logger.debug('_initialize_plc')

        # 카메라1, 2에 해당 메모리 번지 ready, 상태 초기화
        self.write_to_plc(values=[1], headdevice=self.READY_CAM1_ADDR)
        time.sleep(0.01)
        self.write_to_plc(values=[1], headdevice=self.READY_CAM2_ADDR)
        time.sleep(0.01)
        self.write_to_plc(values=[0], headdevice=self.BUSY_CAM1_ADDR)
        time.sleep(0.01)
        self.write_to_plc(values=[0], headdevice=self.BUSY_CAM2_ADDR)
        time.sleep(0.01)
        self.write_to_plc(values=[0], headdevice=self.READ_PLC_ADDR)
        time.sleep(0.01)
        self.change_connection_status(ConnectionStatus.INITIALIZED)
        # logger.debug("PLC has initialized.")
        
    def start_monitoring(self):
        # logger.debug('start_monitoring')
        self.read_from_plc()

    def change_connection_status(self, connection_status: ConnectionStatus):
        self.connection_status = connection_status

    def process_snapshot(self, device, path, mongo, model, pre_h_line, h_repre_list, h_repre_tf_list, repre_list, h_line_list):
        # 비전 카메라 2 Ready OFF
        self.write_to_plc(values=[0], headdevice=self.READY_CAM2_ADDR)
        # logger.debug("(Cam)>>Vision Ready Off.\n")
        time.sleep(0.01)
        # 비전 카메라 2 busy ON
        self.write_to_plc(values=[1], headdevice=self.BUSY_CAM2_ADDR)
        # logger.debug("(Cam)>>Vision Busy On.\n")
        
        section_id = int((path.split('/')[-1])[2:])
        timestamp_date = path.split('/')[-3]
        timestamp_time = path.split('/')[-2]
        timestamp = f'{timestamp_date}:{timestamp_time}'

        time.sleep(0.1)
        img_array = grab_img(device, str(path))   
        time.sleep(0.01)
        # Analysis (정상:1, 비정상:2)
        info, repre_points, analysis_output, pre_h_line = inference(model, img_array, path, section_id, pre_h_line)
        repre_list.append(repre_points)
        if section_id % 7 == 0:
            h_line_list.append(pre_h_line)
            # if section_id == 7:
            #     repre_list = del_none_product_repre(repre_list)
            c_h_repre_list, c_h_repre_tf_list = h_sum(repre_list)
            h_repre_list.append(c_h_repre_list)
            h_repre_tf_list.append(c_h_repre_tf_list)
            repre_list = list()

        # section_info is dictionary
        section_info = dict(
            section_id = section_id,
            section_flag = bool(analysis_output - 1),
            patches=info
        )

        
        if section_id == 1:
            documents = dict(
            timestamp = timestamp,
            sections = [section_info]
            )

            # print("mongo!")
            mongo.create_documents(documents)
        else:
            mongo.update_documents(timestamp, section_info)


        time.sleep(0.01)
        if analysis_output == 2:
            analysis_output = 1
        else:
            analysis_output = 2
        self.write_to_plc(values=[analysis_output], headdevice=self.CAM2_RESULT_ADDR)
        # logger.info(f"Send Result: {analysis_output}")

        # px2cm

        # send NG position(x,y)
        if analysis_output == 2:
            # x position
            import random
            self.write_to_plc(values=[random.randint(-20, 20)], headdevice=self.NG_RESULT_X)
            # y position
            self.write_to_plc(values=[random.randint(-20, 20)], headdevice=self.NG_RESULT_Y)
            while(True):
                # print(self.plc2pc_get_val(headdevice=self.NG_FINISH)[0])
                if(self.plc2pc_get_val(headdevice=self.NG_FINISH)[0]==1):
                    self.write_to_plc(values=[0], headdevice=self.NG_RESULT_X)
                    self.write_to_plc(values=[0], headdevice=self.NG_RESULT_Y)
                    break
                
        time.sleep(0.01)

        # 스냅 완료 신호 전송
        self.write_to_plc(values=[1], headdevice=self.SNAP_CAM2_ADDR)
        
        # 비전 카메라 2 Busy Off
        self.write_to_plc(values=[0], headdevice=self.BUSY_CAM2_ADDR)
        # logger.debug("(Cam)>>Vision Busy Off.\n")

        # 비전 카메라 2 Ready On
        self.write_to_plc(values=[1], headdevice=self.READY_CAM2_ADDR)
        # logger.debug("(Cam)>>Vision Ready On.\n")

        return pre_h_line, h_repre_list, h_repre_tf_list, repre_list, h_line_list

    # PLC TO PC
    def plc2pc_get_val(self, read_size=2, headdevice="D10000"):
        val_list = self.conn_class.batchread_wordunits(headdevice=headdevice, readsize=read_size)
        return val_list

    # PC TO PLC
    def write_to_plc(self, values, headdevice="D10050"):
        conn_class = self.conn_class
        conn_class.batchwrite_wordunits(headdevice=headdevice, values=values)

    # PLC TO PC for monitoring
    def read_from_plc(self, headdevice="D10000", read_size=2):
        conn_class = self.conn_class
        while self.connection_status == ConnectionStatus.INITIALIZED:
            word_values = conn_class.batchread_wordunits(headdevice=headdevice, readsize=read_size)
            self.snap_signal = word_values[0]
            self.current_camera_position = word_values[1]
            self.on_camera_snap_signal(self.snap_signal)
            camera_no = 1 if self.current_camera_position == 0 else 2
            if self.snap_signal == 1:
                self.on_camera_position_change(camera_no, self.current_camera_position)
                self.process_snapshot()
            time.sleep(0.1)