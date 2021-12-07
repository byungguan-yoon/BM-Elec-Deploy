from pymcprotocol import Type3E

class Type3EE(Type3E):

    def __init__(self, plctype="Q"):
        super().__init__(self, plctype)

    def get_connection_status(self):
        return self._is_connected




