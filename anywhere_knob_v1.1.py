import cv2
import mediapipe as mp
import time
import math

from multi_dimensions_ring_buffer import MultiDimensionalRingBuffer

class VirtualKnob:
    def __init__(self, press_threshold=600, release_threshold=6000, move_sigma_threshold = 8,
                 activation_ticks=10,
                 minimum_distance=None,
                 maximum_distance=None,
                 initial_angle=60,
                 callback=None
                 ):

        if maximum_distance != None and minimum_distance !=None and maximum_distance < minimum_distance:
            raise ValueError('maximum_distance < minimum_distance')

        # self dist = 0.0
        self.data_a = []
        self.data_b = []
        self.data_c = []
        self.status = 'RELEASED'

        '''旋钮激活阈值'''
        self.activation_ticks = activation_ticks
        self.available_time_tick = 0

        self.RingBuffer = MultiDimensionalRingBuffer(size=10, dimensions=3)
        self.DistRingBuffer = MultiDimensionalRingBuffer(size=5, dimensions=1)
        self.press_threshold = press_threshold
        self.release_threshold = release_threshold
        self.move_sigma_threshold = move_sigma_threshold

        '''坐标: 相对坐标: 旋转角度'''
        self.previous_angle = 0.0

        '''坐标: 绝对坐标: 旋转角度'''
        self.absolute_angle = 0.0
        self.activate_angle = 0.0

        '''回调相关'''
        if callback != None:
            '''限位'''
            self.minimum_distance = minimum_distance
            self.maximum_distance = maximum_distance

            '''输出滤波'''
            self.ExternalRingBuffer = MultiDimensionalRingBuffer(size=5, dimensions=1)

            self.external_value = initial_angle

            '''回调函数'''
            self.callback = callback

    def update_centroid(self, frame):
        """计算两点之间的距离"""
        dist = round((self.data_a[0] - self.data_b[0]) ** 2 + (self.data_a[1] - self.data_b[1]) ** 2, 2)
        self.DistRingBuffer.push(dist)

        """根据两点计算质心，并在图像上标记"""
        self.data_c = [(self.data_a[i] + self.data_b[i]) / 2 for i in range(3)]
        cv2.circle(frame, (int(self.data_c[0]), int(self.data_c[1])), 5, (0, 0, 255), -1)

        if self.status == 'RELEASED':
            return
        
        # 绘制圆圈, 半径为self.DistRingBuffer.get_avg()
        if self.status == 'PRESSING':
            color = (128, 0, 0)
        elif self.status == 'KNOB_ACTIVE':
            color = (0, 128, 0)
        cv2.circle(frame, (int(self.data_c[0]), int(self.data_c[1])), int(self.DistRingBuffer.get_avg()/200), color, 2)

    def get_angle(self):
        """计算虚拟旋钮 a b 两点对水平线的角度"""
        angle = round(math.atan2(self.data_b[1] - self.data_a[1], self.data_b[0] - self.data_a[0]) * 180 / math.pi, 2)
        return angle

    def callback_process(self, d_angle):
        # deadzone
        if abs(d_angle) < 5:  # deadzone
            return
        
        d_angle = (d_angle // 2) * 2
        
        self.external_value = self.external_value + d_angle
        if self.minimum_distance != None:
            if self.external_value < self.minimum_distance:
                self.external_value = self.minimum_distance
        if self.maximum_distance != None:
            if self.external_value > self.maximum_distance:
                self.external_value = self.maximum_distance
        self.ExternalRingBuffer.push(self.external_value)
        self.callback(int(self.ExternalRingBuffer.get_avg()))

    # v0.1 状态机
    def check_status(self):
        """根据距离判断虚拟旋钮的状态"""

        if self.status == 'RELEASED':
            if self.DistRingBuffer.get_avg() > self.press_threshold and self.DistRingBuffer.get_avg() < self.release_threshold:
                self.status = 'PRESSING'
        elif self.status == 'PRESSING':
            if self.DistRingBuffer.get_avg() < self.press_threshold or self.DistRingBuffer.get_avg() > self.release_threshold:
                self.status = 'RELEASED'
                self.available_time_tick = 0
                self.RingBuffer.reset()
            
            self.RingBuffer.push(self.data_c)
            self.available_time_tick += 1
            # 超时激活
            if (self.available_time_tick > self.activation_ticks
                 and self.RingBuffer.get_sigma_avg() < self.move_sigma_threshold):
                
                self.activate_angle = self.absolute_angle
                self.previous_angle = self.get_angle()
                self.status = 'KNOB_ACTIVE'
        elif self.status == 'KNOB_ACTIVE':
            if (self.DistRingBuffer.get_avg() < self.press_threshold
                 or self.DistRingBuffer.get_avg() > self.release_threshold):
                
                self.status = 'RELEASED'
                self.available_time_tick = 0
                self.RingBuffer.reset()
            elif self.RingBuffer.get_sigma_avg() > self.move_sigma_threshold:

                self.status = 'KNOB_ACTIVE'
                self.available_time_tick = 0
            
            self.RingBuffer.push(self.data_c)

            current_angle = self.get_angle()
            # 可能会在角度从359度变到0度（或相反）时出现大幅跳变（如-1度到+1度）
            # d_angle = current_angle - self.previous_angle

            if self.previous_angle > 90 and current_angle < -90:
                d_angle = current_angle - self.previous_angle + 360
            elif self.previous_angle < -90 and current_angle > 90:
                d_angle = current_angle - self.previous_angle - 360
            else:
                d_angle = current_angle - self.previous_angle

            self.absolute_angle = self.absolute_angle + d_angle
            self.previous_angle = current_angle

            if knob_callback != None:
                self.callback_process(d_angle)
        
        # udp_client.send_message(str(self.absolute_angle))
    
class FPSCounter:
    def __init__(self):
        self.fps_start_time = time.time()
        self.fps_frame_count = 0
        self.fps = 0
    
    def update(self):
        self.fps_frame_count += 1
        if self.fps_frame_count >= 15:
            fps_end_time = time.time()
            self.fps = round(self.fps_frame_count / (fps_end_time - self.fps_start_time), 2)
            self.fps_frame_count = 0
            self.fps_start_time = time.time()

current_value = 50
def knob_callback(angle):
    global current_value
    current_value = angle

global virtual_knob
virtual_knob = VirtualKnob(minimum_distance= 0.0, maximum_distance= 100.0, callback=knob_callback)

global fps_counter
fps_counter = FPSCounter()

# global udp_client
# udp_client = UDPClient('127.0.0.1', 2467)


# 初始化MediaPipe Pose模型和Hand模型
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
 
# 读取视频流或摄像头
cap = cv2.VideoCapture(0)  # 0表示默认摄像头

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
 
    # 转换BGR图像为RGB图像
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
 
    # 运行手部估计模型
    hand_results = hands.process(rgb_frame)
 
    # 检测手部关键点
    if hand_results.multi_hand_landmarks:
        for landmarks in hand_results.multi_hand_landmarks:
            # 定义要高亮显示的手指地标索引
            highlighted_landmark_indices = [4, 8]

            for index, point in enumerate(landmarks.landmark):
                # 检查当前地标索引是否在需要高亮的列表中
                if index in highlighted_landmark_indices:
                    # 计算实际像素坐标
                    x, y, z = int(point.x * frame.shape[1]), int(point.y * frame.shape[0]), int(point.z * frame.shape[2])

                    if index == 4:
                        virtual_knob.data_a = [x, y, z]
                    if index == 8:
                        virtual_knob.data_b = [x, y, z]

                    # 绘制标记
                    cv2.circle(frame, (x, y), 5, (155, 155, 0), -1)

            # virtual_knob.calculate_distance()
            # 计算双指夹距
            virtual_knob.update_centroid(frame)
    else:
        virtual_knob.dist = 0.0
 
    # 计算FPS
    fps_counter.update()
 
    # 显示FPS
    cv2.putText(frame, f"FPS: {fps_counter.fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # 显示双指夹距
    # cv2.putText(frame, f'dist: {virtual_knob.dist:.2f}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 显示test_value
    cv2.putText(frame, f'test_value: {current_value}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # # 显示绝对角度
    # cv2.putText(frame, f'Angle: {virtual_knob.absolute_angle}', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    # 显示角度
    cv2.putText(frame, f'Angle: {virtual_knob.previous_angle}', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


    # if virtual_knob.status == 'KNOB_ACTIVE':
    #     temp_value = int(value - virtual_knob.knob_angle)
    #     if temp_value > 100:
    #         temp_value = 100
    #     elif temp_value < 0:
    #         temp_value = 0
    #     cv2.putText(frame, f'test_value: {temp_value}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    # else:
    #     value = temp_value
    #     cv2.putText(frame, f'test_value: {value}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 显示虚拟旋钮状态
    virtual_knob.check_status()
    cv2.putText(frame, f'STATUS: {virtual_knob.status}', (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
 
    # # 显示圆心 sigma
    # sigma_avg = virtual_knob.RingBuffer.get_sigma_avg()
    # if sigma_avg is not None:
    #     cv2.putText(frame, f'sigma_avg: {sigma_avg}', (10, 430), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # if virtual_knob.status == 'KNOB_ACTIVE':
        # cv2.putText(frame, f'angle: {virtual_knob.knob_angle}', (10, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 显示结果
    cv2.imshow('Pose and Hand Detection', frame)
 
    # 退出程序
    if cv2.waitKey(1) & 0xFF == 27:  # 按ESC键退出
        break
 
# 释放资源
cap.release()
cv2.destroyAllWindows()