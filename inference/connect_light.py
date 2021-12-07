import serial
import time
# from loguru import logger

class lightConnector:
    def __init__(self, port, baud):
        self.port = port
        self.baud = baud
        self.conn_class = self.connect_init()
        self.initialize_light()
    
    def connect_init(self):
        conn_class = serial.Serial(self.port, self.baud)
        if conn_class.is_open == True:
            # logger.info(f"Connected Light, Port: {self.port}")
            pass
        else:
            # logger.error("Failed to connect light")
            pass
        return conn_class

    def initialize_light(self):
        for i in range(1, 5):
            self.change_state(f'{i}', '000')
            self.change_state(f'{i}', 'OFF')
        # logger.info("initialize light")
    
    def change_state(self, channel, data):
        op = self.combine_op(channel, data)
        self.conn_class.write(op)

    def combine_op(self, channel, data):
        head = bytes(bytearray([0x02]))
        channel = channel.encode()
        data = data.encode()
        tail = bytes(bytearray([0x03]))
        op = head + channel + data + tail
        return op
        
if __name__ == "__main__":
    port = '/dev/ttyUSB0'
    baud = 9600
    light_connector = lightConnector(port, baud)