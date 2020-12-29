# from win32api import GetSystemMetrics


class Constant:
    """
    store config param
    """
    __instance = None
    __start_face_recognition_message = "start face recognition"
    __end_face_recognition_message = "end face recognition"
    __train_new_face_message = "Train face"
    __port_socket_1 = 11111
    __port_socket_2 = 22222
    __client_number = 5
    __bufsize = 1024
    __host = '127.0.0.1'
    __time_caught_face = 10

    # size gui
    # __monitor_width = GetSystemMetrics(0)
    # __monitor_hight = GetSystemMetrics(1)
    # __main_width = __monitor_width - 500
    # __main_hight = __monitor_hight - 200
    __left_layout_width = 300
    __right_layout_width = 300
    # __camera_layout_width = __main_width - __left_layout_width - __right_layout_width
    # __camera_layout_hight = __main_hight - 50

    @staticmethod
    def get_instance():
        """ Static access method. """
        if Constant.__instance is None:
            Constant()
        return Constant.__instance

    def __init__(self):
        """ Virtually private constructor. """
        if Constant.__instance is not None:
            raise Exception("This class is a singleton!")
        else:
            Constant.__instance = self

    @property
    def START_FACE_RECOGNITION_MESSAGE(self) -> str:
        return self.__start_face_recognition_message

    @property
    def END_FACE_RECOGNITION_MESSAGE(self) -> str:
        return self.__end_face_recognition_message

    @property
    def TRAIN_NEW_FACE_MESSAGE(self) -> str:
        return self.__train_new_face_message

    @property
    def PORT_SOCKET_1(self) -> int:
        return self.__port_socket_1

    @property
    def PORT_SOCKET_2(self) -> int:
        return self.__port_socket_2

    @property
    def CLIENT_NUMBER(self) -> int:
        return self.__client_number

    @property
    def BUFSIZE(self) -> int:
        return self.__bufsize

    @property
    def HOST(self) -> str:
        return self.__host

    @property
    def TIME_CAUGHT_FACE(self) -> int:
        return self.__time_caught_face

    @property
    def MONITOR_WIDTH(self) -> int:
        return self.__monitor_widgh

    @property
    def MONITOR_HIGHT(self) -> int:
        return self.__monitor_hight

    @property
    def MAIN_WIDTH(self) -> int:
        return self.__main_width

    @property
    def MAIN_HIGHT(self) -> int:
        return self.__main_hight

    @property
    def LEFT_LAYOUT_WIDTH(self) -> int:
        return self.__left_layout_width

    @property
    def RIGHT_LAYOUT_WIDTH(self) -> int:
        return self.__right_layout_width

    # @property
    # def CAMERA_LAYOUT_WIDTH(self) -> int:
    #     return self.__camera_layout_width
    #
    # @property
    # def CAMERA_LAYOUT_HIGHT(self) -> int:
    #     return self.__camera_layout_hight
