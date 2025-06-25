from hik_camera import call_back_get_image, start_grab_and_get_data_size, close_and_destroy_device, set_Value, \
    get_Value, image_control
from MvImport.MvCameraControl_class import *
import cv2
import threading
import time
import numpy as np

from detect_function import YOLOv5Detector
from information_ui import draw_information_ui


loaded_arrays = np.load('arrays_test_red.npy')  # 加载标定好的仿射变换矩阵
map_image = cv2.imread("images/bad_map.jpg")  # 加载红方视角地图

M_ground = loaded_arrays[0]  # 地面层、公路层


# 初始化战场信息UI
information_ui = np.zeros((500, 420, 3), dtype=np.uint8) * 255
information_ui_show = information_ui.copy()

# 加载战场地图
map_backup = cv2.imread("images/map.jpg")
map = map_backup.copy()


# 机器人坐标滤波器（滑动窗口均值滤波）
class Filter:
    def __init__(self, window_size, max_inactive_time=2.0):
        self.window_size = window_size
        self.max_inactive_time = max_inactive_time
        self.data = {}  # 存储不同机器人的数据
        self.window = {}  # 存储滑动窗口内的数据
        self.last_update = {}  # 存储每个机器人的最后更新时间

    # 添加机器人坐标数据
    def add_data(self, name, x, y, threshold=100000.0):  # 阈值单位为mm，实测没啥用，不如直接给大点
        global guess_list
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
        guess_list[name] = False

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

    # 获取所有机器人坐标
    def get_all_data(self):
        filtered_d = {}
        for name in self.data:
            # 超过max_inactive_time没识别到机器人将会清空缓冲区，并进行盲区预测
            if time.time() - self.last_update[name] > self.max_inactive_time:
                self.data[name].clear()
                self.window[name].clear()
                guess_list[name] = True
            # 识别到机器人，不进行盲区预测
            else:
                guess_list[name] = False
                filtered_d[name] = self.filter_data(name)
        # 返回所有当前识别到的机器人及其坐标的均值
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



# 创建机器人坐标滤波器
filter = Filter(window_size=3, max_inactive_time=2)
# 加载模型，实例化机器人检测器和装甲板检测器
weights_path = 'models/car.onnx'  # 建议把模型转换成TRT的engine模型，推理速度提升10倍，转换方式看README
weights_path_next = 'models/armor.onnx'
# weights_path = 'models/car.engine'
# weights_path_next = 'models/armor.engine'
detector = YOLOv5Detector(weights_path, data='yaml/car.yaml', conf_thres=0.1, iou_thres=0.5, max_det=14, ui=True)
detector_next = YOLOv5Detector(weights_path_next, data='yaml/armor.yaml', conf_thres=0.50, iou_thres=0.2,
                               max_det=1,
                               ui=True)



# 图像测试模式（获取图像根据自己的设备，在）
camera_mode = 'hik'  # 'test':测试模式,'hik':海康相机,'video':USB相机（videocapture）

camera_image = None

if camera_mode == 'test':
    camera_image = cv2.imread('images/test_image.jpg')
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
    # 第一层神经网络识别
    result0 = detector.predict(img0)
    det_time += 1
    for detection in result0:
        cls, xywh, conf = detection
        if cls == 'car':
            left, top, w, h = xywh
            left, top, w, h = int(left), int(top), int(w), int(h)
            # 存储第一次检测结果和区域
            # ROI出机器人区域
            cropped = camera_image[top:top + h, left:left + w]
            cropped_img = np.ascontiguousarray(cropped)
            # 第二层神经网络识别
            result_n = detector_next.predict(cropped_img)
            det_time += 1
            if result_n:
                # 叠加第二次检测结果到原图的对应位置
                img0[top:top + h, left:left + w] = cropped_img

                for detection1 in result_n:
                    cls, xywh, conf = detection1
                    if cls:  # 所有装甲板都处理，可选择屏蔽一些:
                        x, y, w, h = xywh
                        x = x + left
                        y = y + top

                        t1 = time.time()
                        
                        # 先套用地面层仿射变化矩阵
                        mapped_point = cv2.perspectiveTransform(camera_point.reshape(1, 1, 2), M_ground)
                        # 限制转换后的点在地图范围内
                        x_c = max(int(mapped_point[0][0][0]), 0)
                        y_c = max(int(mapped_point[0][0][1]), 0)
                        x_c = min(x_c, width)
                        y_c = min(y_c, height)
    
    # 获取所有识别到的机器人坐标
    all_filter_data = filter.get_all_data()
    # print(all_filter_data_name)
    if all_filter_data != {}:
        for name, xyxy in all_filter_data.items():
            if xyxy is not None:
                if name[0] == "R":
                    color_m = (0, 0, 255)
                else:
                    color_m = (255, 0, 0)
                if state == 'R':
                    filtered_xyz = (2800 - xyxy[1], xyxy[0])  # 缩放坐标到地图图像
                else:
                    filtered_xyz = (xyxy[1], 1500 - xyxy[0])  # 缩放坐标到地图图像
                # 只绘制敌方阵营的机器人（这里不会绘制盲区预测的机器人）
                if name[0] != state:
                    cv2.circle(map, (int(filtered_xyz[0]), int(filtered_xyz[1])), 15, color_m, -1)  # 绘制圆
                    cv2.putText(map, str(name),
                                (int(filtered_xyz[0]) - 5, int(filtered_xyz[1]) + 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 5)
                    ser_x = int(filtered_xyz[0]) * 10 / 10
                    ser_y = int(1500 - filtered_xyz[1]) * 10 / 10
                    cv2.putText(map, "(" + str(ser_x) + "," + str(ser_y) + ")",
                                (int(filtered_xyz[0]) - 100, int(filtered_xyz[1]) + 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 4)
    
    te = time.time()
    t_p = te - ts
    print("fps:", 1 / t_p)  # 打印帧率
    # # 绘制UI
    # _ = draw_information_ui(progress_list, state, information_ui_show)
    # cv2.putText(information_ui_show, "vulnerability_chances: " + str(double_vulnerability_chance),
    #             (10, 350),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    # cv2.putText(information_ui_show, "vulnerability_Triggering: " + str(opponent_double_vulnerability),
    #             (10, 400),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow('information_ui', information_ui_show)
    map_show = cv2.resize(map, (600, 320))
    cv2.imshow('map', map_show)
    img0 = cv2.resize(img0, (1300, 900))
    cv2.imshow('img', img0)

    key = cv2.waitKey(1)

    # det_time = 0
    # img0 = camera_image.copy()
    # ts = time.time()
    # cv2.imshow("Image Window", img0)
    # # 第一层神经网络识别
    # result0 = detector.predict(img0)
    # det_time += 1
    # for detection in result0:
    #     cls, xywh, conf = detection
    #     if cls == 'car':
    #         left, top, w, h = xywh
    #         left, top, w, h = int(left), int(top), int(w), int(h)
    #         # 存储第一次检测结果和区域
    #         # 原图中框的中心下沿作为待仿射变化的点
    #         camera_point = np.array([[[min(x + 0.5 * w, img_x), min(y + 1.5 * h, img_y)]]],
    #                                 dtype=np.float32)
    #         # 低到高依次仿射变化
    #         # 先套用地面层仿射变化矩阵
    #         mapped_point = cv2.perspectiveTransform(camera_point.reshape(1, 1, 2), M_ground)
    #         # 限制转换后的点在地图范围内
    #         x_c = max(int(mapped_point[0][0][0]), 0)
    #         y_c = max(int(mapped_point[0][0][1]), 0)
    #         x_c = min(x_c, width)
    #         y_c = min(y_c, height)


    # te = time.time()
    # t_p = te - ts
    # print("fps:", 1 / t_p)  # 打印帧率
    # key = cv2.waitKey(1)
