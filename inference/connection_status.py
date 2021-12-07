from enum import Enum


class ConnectionStatus(Enum):
    DISCONNECTED = 1
    CONNECTED = 2
    INITIALIZED = 3
