import pymcprotocol
import time
# import numpy as np

class conn_plpc:
    def  __init__(self, host, port,  plctype="QnA"):
        self.host = host
        self.port = port
        self.plctype = plctype

        self.conn_class = self.connect_init()

    def connect_init(self):
        conn_class = pymcprotocol.Type3E() # plc type
        conn_class.setaccessopt(commtype='binary')
        conn_class.connect(self.host, self.port)
        return conn_class

    # PLC TO PC
    def plc2pc(self, headdevice="D10000"):
        conn_class = self.conn_class
        while True:
            start_time = time.time()
            print(f'{time.time() - start_time}', conn_class.batchread_wordunits(headdevice=headdevice, readsize=1)[0])
    
    def plc2pc_get_val(self, read_size=1, headdevice="D10000"):
        val_list = self.conn_class.batchread_wordunits(headdevice=headdevice, readsize=read_size)

        return val_list

    # PC TO PLC
    def pc2plc_send_val(self, values, headdevice="D10050"): 
        conn_class = self.conn_class
        conn_class.batchwrite_wordunits(headdevice=headdevice, values=values)

    # def flicker(self, headdevice):
    #     conn_class = self.conn_class

    #     i = 0
    #     while True:
    #         conn_class.batchwrite_wordunits(headdevice, values=[i])
    #         if i == 0:
    #             i = 1
    #         elif i == 1:
    #             i = 0
    #     conn_class.batchwrite_wordunits(headdevice, values=2)


if __name__ == '__main__':
    connec = conn_plpc("192.168.10.7", 2001)
    print(connec.plc2pc_get_val(1, "D10000"))
    connec.pc2plc_send_val(values=[0], headdevice="D10054")