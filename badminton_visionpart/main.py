from hik_camera import call_back_get_image, start_grab_and_get_data_size, close_and_destroy_device, set_Value, \
    get_Value, image_control
from MvImport.MvCameraControl_class import *
import cv2
import threading
import time
import numpy as np
import json 
import serial

# from detect_function import YOLOv5Detector
from information_ui import draw_information_ui
from hobot_dnn import pyeasy_dnn as dnn
from RM_serial_py.ser_api import  build_send_packet, receive_packet, build_data_radar
import queue

# yaw_pit_queue = queue.Queue(maxsize=5)
yaw_pit_queue = queue.Queue()


loaded_arrays = np.load('arrays_test_red.npy')  # 加载标定好的仿射变换矩阵
map_image = cv2.imread("images/resized_image.jpg")  # 加载地图

M_ground = loaded_arrays[0]  # 地面层


# 初始化UI
information_ui = np.zeros((500, 420, 3), dtype=np.uint8) * 255
information_ui_show = information_ui.copy()

# 加载地图
map_backup = cv2.imread("images/resized_image.jpg")
map = map_backup.copy()

# 确定地图画面像素，保证不会溢出
height, width = map_image.shape[:2]
height -= 1
width -= 1

# 坐标滤波器（滑动窗口均值滤波）
class Filter:
    def __init__(self, window_size, max_inactive_time=2.0):
        self.window_size = window_size
        self.max_inactive_time = max_inactive_time
        self.data = {}  # 存储不同人物的数据
        self.window = {}  # 存储滑动窗口内的数据
        self.last_update = {}  # 存储每个人的最后更新时间

    # 添加坐标数据
    def add_data(self, name, x, y, threshold=100000.0):  # 阈值单位为mm，实测没啥用，不如直接给大点
        if name not in self.data:
            # 如果实体名称不在数据字典中，初始化相应的deque。
            self.data[name] = deque(maxlen=self.window_size)
            self.window[name] = deque(maxlen=self.window_size)

        if len(self.window[name]) >= 2:
            # 计算当前坐标与前一个坐标的均方
            msd = sum((a - b) ** 2 for a, b in zip((x, y), self.window[name][-1])) / 2.0
            # print(name, msd)

            if msd > threshold:
                # 如果均方差超过阈值，可能是异常值，不将其添加到数据中
                return

        # 将坐标数据添加到数据字典和滑动窗口中。
        self.data[name].append((x, y))

        self.window[name].append((x, y))
        self.last_update[name] = time.time()  # 更新最后更新时间

    # 过滤计算滑动窗口平均值
    def filter_data(self, name):
        if name not in self.data:
            return None

        if len(self.window[name]) < self.window_size:
            return None  # 不足以进行滤波

        # 计算滑动窗口内的坐标平均值
        x_avg = sum(coord[0] for coord in self.window[name]) / self.window_size
        y_avg = sum(coord[1] for coord in self.window[name]) / self.window_size

        return x_avg, y_avg

    # 获取所有坐标
    def get_all_data(self):
        filtered_d = {}
        for name in self.data:
            filtered_d[name] = self.filter_data(name)
        # 返回所有当前识别到的人及其坐标的均值
        return filtered_d


# 海康相机图像获取线程
def hik_camera_get():
    # 获得设备信息
    global camera_image
    deviceList = MV_CC_DEVICE_INFO_LIST()
    tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE

    # ch:枚举设备 | en:Enum device
    # nTLayerType [IN] 枚举传输层 ，pstDevList [OUT] 设备列表
    while 1:
        ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
        if ret != 0:
            print("enum devices fail! ret[0x%x]" % ret)
            # sys.exit()

        if deviceList.nDeviceNum == 0:
            print("find no device!")
            # sys.exit()
        else:
            print("Find %d devices!" % deviceList.nDeviceNum)
            break

    for i in range(0, deviceList.nDeviceNum):
        mvcc_dev_info = cast(deviceList.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
        if mvcc_dev_info.nTLayerType == MV_GIGE_DEVICE:
            print("\ngige device: [%d]" % i)
            # 输出设备名字
            strModeName = ""
            for per in mvcc_dev_info.SpecialInfo.stGigEInfo.chModelName:
                strModeName = strModeName + chr(per)
            print("device model name: %s" % strModeName)
            # 输出设备ID
            nip1 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0xff000000) >> 24)
            nip2 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x00ff0000) >> 16)
            nip3 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x0000ff00) >> 8)
            nip4 = (mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x000000ff)
            print("current ip: %d.%d.%d.%d\n" % (nip1, nip2, nip3, nip4))
        # 输出USB接口的信息
        elif mvcc_dev_info.nTLayerType == MV_USB_DEVICE:
            print("\nu3v device: [%d]" % i)
            strModeName = ""
            for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chModelName:
                if per == 0:
                    break
                strModeName = strModeName + chr(per)
            print("device model name: %s" % strModeName)

            strSerialNumber = ""
            for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chSerialNumber:
                if per == 0:
                    break
                strSerialNumber = strSerialNumber + chr(per)
            print("user serial number: %s" % strSerialNumber)
    # 手动选择设备
    # nConnectionNum = input("please input the number of the device to connect:")
    # 自动选择设备
    nConnectionNum = '0'
    if int(nConnectionNum) >= deviceList.nDeviceNum:
        print("intput error!")
        sys.exit()

    # ch:创建相机实例 | en:Creat Camera Object
    cam = MvCamera()

    # ch:选择设备并创建句柄 | en:Select device and create handle
    # cast(typ, val)，这个函数是为了检查val变量是typ类型的，但是这个cast函数不做检查，直接返回val
    stDeviceList = cast(deviceList.pDeviceInfo[int(nConnectionNum)], POINTER(MV_CC_DEVICE_INFO)).contents

    ret = cam.MV_CC_CreateHandle(stDeviceList)
    if ret != 0:
        print("create handle fail! ret[0x%x]" % ret)
        sys.exit()

    # ch:打开设备 | en:Open device
    ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
    if ret != 0:
        print("open device fail! ret[0x%x]" % ret)
        sys.exit()

    print(get_Value(cam, param_type="float_value", node_name="ExposureTime"),
          get_Value(cam, param_type="float_value", node_name="Gain"),
          get_Value(cam, param_type="enum_value", node_name="TriggerMode"),
          get_Value(cam, param_type="float_value", node_name="AcquisitionFrameRate"))

    # 设置设备的一些参数
    set_Value(cam, param_type="float_value", node_name="ExposureTime", node_value=16000)  # 曝光时间
    set_Value(cam, param_type="float_value", node_name="Gain", node_value=4.9)  # 增益值
    # 开启设备取流
    start_grab_and_get_data_size(cam)
    # 主动取流方式抓取图像
    stParam = MVCC_INTVALUE_EX()

    memset(byref(stParam), 0, sizeof(MVCC_INTVALUE_EX))
    ret = cam.MV_CC_GetIntValueEx("PayloadSize", stParam)
    if ret != 0:
        print("get payload size fail! ret[0x%x]" % ret)
        sys.exit()
    nDataSize = stParam.nCurValue
    pData = (c_ubyte * nDataSize)()
    stFrameInfo = MV_FRAME_OUT_INFO_EX()

    memset(byref(stFrameInfo), 0, sizeof(stFrameInfo))
    while True:
        ret = cam.MV_CC_GetOneFrameTimeout(pData, nDataSize, stFrameInfo, 1000)
        if ret == 0:
            image = np.asarray(pData)
            # 处理海康相机的图像格式为OPENCV处理的格式
            camera_image = image_control(data=image, stFrameInfo=stFrameInfo)
        else:
            print("no data[0x%x]" % ret)


def video_capture_get():
    global camera_image
    cam = cv2.VideoCapture(1)
    while True:
        ret, img = cam.read()
        if ret:
            camera_image = img
            time.sleep(0.016)  # 60fps


class hbSysMem_t(ctypes.Structure):
    _fields_ = [
        ("phyAddr",ctypes.c_double),
        ("virAddr",ctypes.c_void_p),
        ("memSize",ctypes.c_int)
    ]

class hbDNNQuantiShift_yt(ctypes.Structure):
    _fields_ = [
        ("shiftLen",ctypes.c_int),
        ("shiftData",ctypes.c_char_p)
    ]

class hbDNNQuantiScale_t(ctypes.Structure):
    _fields_ = [
        ("scaleLen",ctypes.c_int),
        ("scaleData",ctypes.POINTER(ctypes.c_float)),
        ("zeroPointLen",ctypes.c_int),
        ("zeroPointData",ctypes.c_char_p)
    ]    

class hbDNNTensorShape_t(ctypes.Structure):
    _fields_ = [
        ("dimensionSize",ctypes.c_int * 8),
        ("numDimensions",ctypes.c_int)
    ]

class hbDNNTensorProperties_t(ctypes.Structure):
    _fields_ = [
        ("validShape",hbDNNTensorShape_t),
        ("alignedShape",hbDNNTensorShape_t),
        ("tensorLayout",ctypes.c_int),
        ("tensorType",ctypes.c_int),
        ("shift",hbDNNQuantiShift_yt),
        ("scale",hbDNNQuantiScale_t),
        ("quantiType",ctypes.c_int),
        ("quantizeAxis", ctypes.c_int),
        ("alignedByteSize",ctypes.c_int),
        ("stride",ctypes.c_int * 8)
    ]

class hbDNNTensor_t(ctypes.Structure):
    _fields_ = [
        ("sysMem",hbSysMem_t * 4),
        ("properties",hbDNNTensorProperties_t)
    ]


class Yolov5PostProcessInfo_t(ctypes.Structure):
    _fields_ = [
        ("height",ctypes.c_int),
        ("width",ctypes.c_int),
        ("ori_height",ctypes.c_int),
        ("ori_width",ctypes.c_int),
        ("score_threshold",ctypes.c_float),
        ("nms_threshold",ctypes.c_float),
        ("nms_top_k",ctypes.c_int),
        ("is_pad_resize",ctypes.c_int)
    ]

libpostprocess = ctypes.CDLL('/usr/lib/libpostprocess.so') 

get_Postprocess_result = libpostprocess.Yolov5PostProcess
get_Postprocess_result.argtypes = [ctypes.POINTER(Yolov5PostProcessInfo_t)]  
get_Postprocess_result.restype = ctypes.c_char_p  

def get_TensorLayout(Layout):
    if Layout == "NCHW":
        return int(2)
    else:
        return int(0)


def bgr2nv12_opencv(image):
    height, width = image.shape[0], image.shape[1]
    area = height * width
    yuv420p = cv2.cvtColor(image, cv2.COLOR_BGR2YUV_I420).reshape((area * 3 // 2,))
    y = yuv420p[:area]
    uv_planar = yuv420p[area:].reshape((2, area // 4))
    uv_packed = uv_planar.transpose((1, 0)).reshape((area // 2,))

    nv12 = np.zeros_like(yuv420p)
    nv12[:height * width] = y
    nv12[height * width:] = uv_packed
    return nv12


def get_hw(pro):
    if pro.layout == "NCHW":
        return pro.shape[2], pro.shape[3]
    else:
        return pro.shape[1], pro.shape[2]


def print_properties(pro):
    print("tensor type:", pro.tensor_type)
    print("data type:", pro.dtype)
    print("layout:", pro.layout)
    print("shape:", pro.shape)


def ser_receive():
    buffer = b''  # 初始化缓冲区
    while True:
        # 从串口读取数据
        received_data = ser1.read_all()  # 读取一秒内收到的所有串口数据
        # 将读取到的数据添加到缓冲区中
        buffer += received_data

        # 查找帧头（SOF）的位置
        sof_index = buffer.find(b'\xA5')

        while sof_index != -1:
            # 如果找到帧头，尝试解析数据包
            if len(buffer) >= sof_index + 5:  # 至少需要5字节才能解析帧头
                # 从帧头开始解析数据包
                packet_data = buffer[sof_index:]

                # 查找下一个帧头的位置
                next_sof_index = packet_data.find(b'\xA5', 1)

                if next_sof_index != -1:
                    # 如果找到下一个帧头，说明当前帧头到下一个帧头之间是一个完整的数据包
                    packet_data = packet_data[:next_sof_index]
                    # print(packet_data)
                else:
                    # 如果没找到下一个帧头，说明当前帧头到末尾不是一个完整的数据包
                    break

                # 解析数据包
                progress_result = receive_packet(packet_data,info=False)  # 解析单个数据包，不输出日志
                # 更新数据
                if progress_result is not None:
                    received_data1, received_seq1 = progress_result
                    progress_list = list(received_data1)
                    rec_yaw= progress_list[0]
                    rec_pit= progress_list[1]

                # 从缓冲区中移除已解析的数据包
                buffer = buffer[sof_index + len(packet_data):]

                # 继续寻找下一个帧头的位置
                sof_index = buffer.find(b'\xA5')

            else:
                # 缓冲区中的数据不足以解析帧头，继续读取串口数据
                break
        time.sleep(0.5)


# 串口发送线程
def ser_send():
    seq = 0
    time_s = time.time()
    update_time = 0  # 上次预测点更新时间
    yaw=0
    pit=0

    while True:
        try:
            # 如果队列有新数据，取出最新的yaw和pit
            yaw, pit = yaw_pit_queue.get_nowait()
        except queue.Empty:
            yaw, pit = 0, 0  # 没有新数据则都等于0

        # 打印当前要发送的yaw和pit
        print(f"发送: yaw={yaw}, pit={pit}")

        ser_data = build_data_radar(yaw,pit)
        packet, seq = build_send_packet(ser_data, seq)
        ser1.write(packet)
        time.sleep(0.2)


# 创建坐标滤波器
filter = Filter(window_size=3, max_inactive_time=2)
# 加载模型
models = dnn.load('./models/yolov5s_672x672_nv12.bin')

# 打印输入 tensor 的属性
print_properties(models[0].inputs[0].properties)
# 打印输出 tensor 的属性
print(len(models[0].outputs))
for output in models[0].outputs:
    print_properties(output.properties)


# 图像测试模式
camera_mode = 'hik'  # 'test':测试模式,'hik':海康相机,'video':USB相机（videocapture）
ser1 = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)  # 串口，替换 'COM1' 为你的串口号

# 串口发送线程
thread_list = threading.Thread(target=ser_send, daemon=True)
thread_list.start()

# # 串口接收线程
# thread_receive = threading.Thread(target=ser_receive, daemon=True)
# thread_receive.start()

camera_image = None

if camera_mode == 'test':
    camera_image = cv2.imread('kite.jpg')
elif camera_mode == 'hik':
    # 海康相机图像获取线程
    thread_camera = threading.Thread(target=hik_camera_get, daemon=True)
    thread_camera.start()
elif camera_mode == 'video':
    # USB相机图像获取线程
    thread_camera = threading.Thread(target=video_capture_get, daemon=True)
    thread_camera.start()

while camera_image is None:
    print("等待图像。。。")
    time.sleep(0.5)

# 获取相机图像的画幅，限制点不超限
img0 = camera_image.copy()
img_y = img0.shape[0]
img_x = img0.shape[1]
print(img0.shape)

while True:
  
    # 刷新裁判系统信息UI图像
    information_ui_show = information_ui.copy()
    map = map_backup.copy()
    det_time = 0
    img0 = camera_image.copy()
    ts = time.time()

    h, w = get_hw(models[0].inputs[0].properties)
    des_dim = (w, h)
    resized_data = cv2.resize(img0, des_dim, interpolation=cv2.INTER_AREA)
    nv12_data = bgr2nv12_opencv(resized_data)
    t0 = time.time()
    outputs = models[0].forward(nv12_data)
    t1 = time.time()
    # print("inferece time is :", (t1 - t0))

    # 获取结构体信息
    yolov5_postprocess_info = Yolov5PostProcessInfo_t()
    yolov5_postprocess_info.height = h
    yolov5_postprocess_info.width = w
    org_height, org_width = img0.shape[0:2]
    yolov5_postprocess_info.ori_height = org_height
    yolov5_postprocess_info.ori_width = org_width
    yolov5_postprocess_info.score_threshold = 0.4 
    yolov5_postprocess_info.nms_threshold = 0.45
    yolov5_postprocess_info.nms_top_k = 20
    yolov5_postprocess_info.is_pad_resize = 0

    output_tensors = (hbDNNTensor_t * len(models[0].outputs))()
    for i in range(len(models[0].outputs)):
        output_tensors[i].properties.tensorLayout = get_TensorLayout(outputs[i].properties.layout)
        # print(output_tensors[i].properties.tensorLayout)
        if (len(outputs[i].properties.scale_data) == 0):
            output_tensors[i].properties.quantiType = 0
            output_tensors[i].sysMem[0].virAddr = ctypes.cast(outputs[i].buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), ctypes.c_void_p)
        else:
            output_tensors[i].properties.quantiType = 2       
            output_tensors[i].properties.scale.scaleData = outputs[i].properties.scale_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            output_tensors[i].sysMem[0].virAddr = ctypes.cast(outputs[i].buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), ctypes.c_void_p)
            
        for j in range(len(outputs[i].properties.shape)):
            output_tensors[i].properties.validShape.dimensionSize[j] = outputs[i].properties.shape[j]
        
        libpostprocess.Yolov5doProcess(output_tensors[i], ctypes.pointer(yolov5_postprocess_info), i)

    result_str = get_Postprocess_result(ctypes.pointer(yolov5_postprocess_info))  
    result_str = result_str.decode('utf-8')  
    t2 = time.time()
    # print("postprocess time is :", (t2 - t1))
    #print(result_str)

    t0 = time.time()
    # draw result
    # 解析JSON字符串  
    data = json.loads(result_str[16:])  

    # allowed_ids = {0, 1}  # 只保留 id 为 0 和 1 的类别
    allowed_names = {"person"}  # 或者用类别名称

    # 遍历每一个结果  
    for result in data:  
        if result['name'] not in allowed_names:
            continue  # 跳过不需要的类别

        bbox = result['bbox']  # 矩形框位置信息  
        score = result['score']  # 得分  
        id = result['id']  # id  
        name = result['name']  # 类别名称  
    
        # 计算下沿1/4高的中心点
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        center_x = min(x1 + 0.5 * w, img0.shape[1])
        center_y = min(y2 - 0.25 * h, img0.shape[0])
        camera_point = np.array([[[center_x, center_y]]], dtype=np.float32)
        print(f"下沿1/4高中心点: {camera_point}")
        # 在该点画圆
        cv2.circle(img0, (int(center_x), int(center_y)), 5, (0, 0, 255), -1)

        # 打印信息  
        print(f"bbox: {bbox}, score: {score}, id: {id}, name: {name}")

        # 在图片上画出边界框  
        cv2.rectangle(img0, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)  
    
        # 在边界框上方显示类别名称和得分  
        font = cv2.FONT_HERSHEY_SIMPLEX  
        cv2.putText(img0, f'{name} {score:.2f}', (int(bbox[0]), int(bbox[1]) - 10), font, 0.5, (0, 255, 0), 1)  
  
        # 套用仿射变化矩阵
        mapped_point = cv2.perspectiveTransform(camera_point.reshape(1, 1, 2), M_ground)
        # 限制转换后的点在地图范围内
        x_c = max(int(mapped_point[0][0][0]), 0)
        y_c = max(int(mapped_point[0][0][1]), 0)
        x_c = min(x_c, width)
        y_c = min(y_c, height)
        cv2.circle(map, (int(mapped_point[0][0][0]), int(mapped_point[0][0][1])), 5, (0, 0, 255), -1)


        pit = 0
        yaw = 0 
        real_x = int(mapped_point[0][0][0]-190)/100 # 单位（米）
        real_z = int(1340-mapped_point[0][0][1])/100 # 单位（米）
        yaw = np.arctan(real_x / real_z)  # 返回弧度
        # 轉角度
        yaw = np.degrees(yaw)
        pit = 0
        yaw_pit_queue.put((yaw, pit))
        # print("yaw (deg):", yaw)

    te = time.time()
    t_p = te - ts
    print("fps:", 1 / t_p)  # 打印帧率

    # 绘制UI
    map_show = cv2.resize(map, (610, 1340))
    cv2.imshow('map', map_show)
    img0 = cv2.resize(img0, (1300, 900))
    cv2.imshow('img', img0)

    key = cv2.waitKey(1)
