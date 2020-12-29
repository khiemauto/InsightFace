import socket

from core.constant import Constant

constant = Constant.get_instance()


class SocketUtil:
    def __init__(self):
        self.__socket = socket.socket()
        self.__connection = None

    # server
    def set_up_server(self, port):
        self.__socket.bind(('', port))
        self.__socket.listen(constant.CLIENT_NUMBER)

    def create_connection(self):
        self.__connection = self.__socket.accept()[0]

    def close_connection(self):
        self.__connection.close()

    def receive_data_from_client(self):
        return self.__connection.recv(constant.BUFSIZE)

    def send_data_to_client(self, data: bytes):
        self.__connection.sendall(data)

    # client
    def set_up_client(self, port):
        self.__socket.connect((constant.HOST, port))

    def send_data_to_server(self, data: bytes):
        self.__socket.sendall(data)

    def receive_data_from_server(self):
        return self.__socket.recv(constant.BUFSIZE)

    def close_socket(self):
        self.__socket.close()
