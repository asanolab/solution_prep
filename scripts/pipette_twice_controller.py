#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from std_msgs.msg import Header, Float32
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from lite6_labauto.msg import LabwareOBB
from moveit_commander import MoveGroupCommander, RobotCommander
from tf.transformations import quaternion_from_euler
from lite6_labauto.msg import MovePose
from xarm_msgs.srv import Move
import numpy as np
from std_msgs.msg import Bool
import tf.transformations as tf_trans
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
# from PIL import Image
import threading
import os
import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms
import sys
from solution_prep.srv import PipetteDoTwice, PipetteDoTwiceResponse

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)  # 放在最前面，避免被覆盖

print("[DEBUG] Added to sys.path:", script_dir)

from model import SimpleCNN


class PipetteController:
    def __init__(self):
        rospy.init_node('pipette_twice_controller', anonymous=True)

        # 订阅
        rospy.Subscriber("/labware/tip_rack5/obb", LabwareOBB, self.cb_tip_rack)
        rospy.Subscriber("/labware/pipette4/obb", LabwareOBB, self.cb_pipette)
        rospy.Subscriber("/labware/beaker0/obb", LabwareOBB, self.cb_hcl_beaker)
        rospy.Subscriber("/labware/beaker2/obb", LabwareOBB, self.cb_target_beaker)
        rospy.Subscriber("/gripper/current_distance", Float32, self.cb_gripper_distance)
        rospy.Subscriber("/marker_pose_base/0", PoseStamped, self.cb_marker_0)
        rospy.Subscriber("/marker_pose_base/1", PoseStamped, self.cb_marker_1)
        rospy.Subscriber("/marker_pose_base/2", PoseStamped, self.cb_marker_2)
        rospy.Subscriber("/marker_pose_base/3", PoseStamped, self.cb_marker_3)
        rospy.Subscriber("/marker_pose_base/4", PoseStamped, self.cb_marker_4)
        rospy.Subscriber("/camera/color/image_raw", Image, self.cb_color)
        self.offset = 0.055

        # 数据缓存
        self.tip_rack_pose = None
        self.pipette_pose = None
        self.hcl_beaker_pose = None
        self.target_beaker_pose = None
        self.marker_poses = {}
        self.marker_last_time = rospy.Time.now()
        self.gripper_distance = None
        self.maker_pose_0 = None
        self.maker_pose_1 = None
        self.maker_pose_2 = None
        self.maker_pose_3 = None
        self.maker_pose_4 = None
        self.bridge = CvBridge()
        self.latest_color = None
        self.latest_color_time = rospy.Time()

        # 机械臂控制
        self.pose_pub = rospy.Publisher("/arm_control/move_pose", MovePose, queue_size=1)

        # 夹爪控制（占位，根据你的实际控制方法替换）
        self.gripper_pub = rospy.Publisher('/gripper/target_distance', Float32, queue_size=10)

        # pipette motor控制
        self.pipette_motor_pub = rospy.Publisher('/pipetty_motor/trigger', Bool, queue_size=10)

        # pipetty 控制
        self.pipetty_aspirate_pub = rospy.Publisher('/pipetty/aspirate', Float32, queue_size=10)
        self.pipetty_home_pub = rospy.Publisher('/pipetty/home', Bool, queue_size=10)

        # tip recognition
        self.debug_image_pub = rospy.Publisher('tip_recognition/color_image', Image, queue_size=1)

        # === 模型路径解析 ===
        # 模型路径
        script_dir = os.path.dirname(os.path.realpath(__file__))
        default_model_path = os.path.join(script_dir, "1734448311.8315692", "best.pth")
        model_path = rospy.get_param("~model_path", default_model_path)

        self.tip_conf_threshold = 0.8
        self.roi_size_px = 32

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model()

        self.transform = transforms.Compose([
            transforms.ToTensor(),  # 期望输入是 RGB 顺序
            transforms.Normalize((0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0.5))
        ])

        # 服务
        self._busy_lock = threading.Lock()

        self.do_srv = rospy.Service("/pipette/dotwice", PipetteDoTwice, self.cb_pipette_do)

    # 回调
    def cb_tip_rack(self, msg):
        self.tip_rack_pose = msg

    def cb_pipette(self, msg):
        self.pipette_pose = msg

    def cb_hcl_beaker(self, msg):
        self.hcl_beaker_pose = msg

    def cb_target_beaker(self, msg):
        self.target_beaker_pose = msg

    def cb_marker_0(self, msg):
        self.marker_pose_0 = msg.pose

    def cb_marker_1(self, msg):
        self.marker_pose_1 = msg.pose

    def cb_marker_2(self, msg):
        self.marker_pose_2 = msg.pose

    def cb_marker_3(self, msg):
        self.marker_pose_3 = msg.pose

    def cb_marker_4(self, msg):
        self.marker_pose_4 = msg.pose

    def cb_gripper_distance(self, msg):
        self.gripper_distance = msg

    def cb_color(self, msg):
        self.latest_color = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self.latest_color_time = rospy.Time.now()

    def cb_pipette_do(self, req):
        try:
            self.pick_pipette()

            # first droplet
            self.attach_tip(tip_id=req.tip_id)
            self.aspirate(volume=int(req.volume_ul))
            self.dispense(id=1)
            self.dispose_of_tip()

            # second droplet
            self.attach_tip(tip_id=req.tip_id + 1)
            self.aspirate_water(volume=int(req.volume_ul))
            self.dispense(id=2)
            self.dispose_of_tip()
            self.place_pipette()
            return PipetteDoTwiceResponse(True, f"Pipette once")
        except Exception as e:
            rospy.logerr(f"pipette_do failed: {e}")
            return PipetteDoTwiceResponse(False, str(e))

    # -------------------- 各步骤 ------------------------

    # ----------------- 机械臂控制 -----------------------

    # for debug: 控制机械臂直接前往arm_pose，其中arm_pose单位为[mm,mm,mm,deg,deg,deg]
    def go_to_arm_pose(self, arm_pose):
        pose = MovePose()
        x_mm = arm_pose[0]
        y_mm = arm_pose[1]
        z_mm = arm_pose[2]
        roll = arm_pose[3] * 3.14159265 / 180.0
        pitch = arm_pose[4] * 3.14159265 / 180.0
        yaw = arm_pose[5] * 3.14159265 / 180.0
        pose.pose = [x_mm, y_mm, z_mm, roll, pitch, yaw]

        rospy.loginfo(f'move to pose: {pose.pose}')

        self.pose_pub.publish(pose)
        rospy.sleep(2.0)

    def go_to_pose_offset(self, base_pose, x_offset=0.0, y_offset=0.0, z_offset=0.0):  # z: 单位m
        """将 OBB pose + z 偏移 → 转换为机械臂 MovePose"""

        rospy.loginfo(f"type of pose: {type(base_pose)}")
        # OBB 的位置
        x_mm = base_pose.pose.position.x * 1000
        y_mm = (base_pose.pose.position.y - y_offset) * 1000
        z_mm = (base_pose.z_height + z_offset + self.offset) * 1000

        # OBB 四元数 → 欧拉角，取 yaw
        quat = [
            base_pose.pose.orientation.x,
            base_pose.pose.orientation.y,
            base_pose.pose.orientation.z,
            base_pose.pose.orientation.w
        ]
        _, _, yaw_obb = tf_trans.euler_from_quaternion(quat)
        rospy.loginfo(f"yaw of obb: {yaw_obb}")
        if yaw_obb > 3.1415926:  # > π
            yaw_obb = yaw_obb - 3.1415926

        # 固定 r, p，动态 yaw
        roll = 179.1 * 3.1415926 / 180
        pitch = 5.6 * 3.1415926 / 180
        yaw = -1.5707963 + yaw_obb  # -90° + OBB yaw

        rospy.loginfo(f'pose msg: {[x_mm, y_mm, z_mm, roll, pitch, yaw]}')

        yaw = -1.57

        # 发布 MovePose
        pose = MovePose()
        pose.pose = [x_mm, y_mm, z_mm, roll, pitch, yaw]

        # print("pose_msg =", pose)
        # print("pose_msg.pose type =", type(pose.pose))
        # print("pose_msg.pose contents =", pose.pose)

        self.pose_pub.publish(pose)
        rospy.sleep(2.0)

    # ----------------- 夹爪控制 -------------------------
    def open_gripper(self, distance=52.0):
        rospy.loginfo(f"Opening gripper to {distance}m")
        self.gripper_pub.publish(Float32(distance))
        rospy.sleep(1.0)

    def close_gripper(self, distance=0.0):
        rospy.loginfo(f"Closing gripper to {distance}m")
        self.gripper_pub.publish(Float32(distance))
        rospy.sleep(1.0)

    # ----------------- marker持续控制 -------------------------
    def go_to_marker_offset(self, pose, z_offset=0.135):

        x_mm = pose.position.x * 1000
        y_mm = pose.position.y * 1000
        z_mm = (pose.position.z + z_offset + self.offset) * 1000

        quat = [
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w
        ]
        _, _, yaw = tf_trans.euler_from_quaternion(quat)

        # yaw > 180° 修正
        if yaw > 3.1415926:  # > π
            yaw = yaw - 3.1415926

        roll = 179.1 * 3.1415926 / 180
        pitch = 5.6 * 3.1415926 / 180
        yaw = -1.5707963 + yaw  # -90° + marker yaw 修正后

        pose_msg = MovePose()
        pose_msg.pose = [x_mm, y_mm, z_mm, roll, pitch, yaw]

        self.pose_pub.publish(pose_msg)

    # ----------------- 实验动作步骤 ----------------------
    def pick_pipette(self):
        rospy.loginfo("Picking pipette...")

        roll = 175.0
        pitch = 0.6
        pitch1 = 5.3
        yaw = -90.8
        x_mm = -259.9
        y_mm = 162.8
        z_mm = 336.7
        z_mm1 = z_mm + 79.3
        z_mm2 = z_mm + 44.6
        y_mm1 = y_mm - 53.1
        # 1️⃣ 到 pipette OBB 上方 5cm
        # self.go_to_pose_offset(self.pipette_pose, y_offset=0.08, z_offset=0.05)
        self.open_gripper()
        rospy.sleep(0.1)
        self.go_to_arm_pose([x_mm, y_mm1, z_mm1, roll, pitch, yaw])
        rospy.sleep(2.0)

        # 2️⃣ 找到最近 ArUco marker，返回 marker pose
        # marker_pose = self.get_nearest_marker_pose()
        self.go_to_arm_pose([x_mm, y_mm, z_mm2, roll, pitch, yaw])
        rospy.loginfo("Searching for marker...")

        """
        timeout = rospy.Duration(3.0)  # 超时时间，自行调整
        rate = rospy.Rate(10)
        #3️⃣ 去 marker 上方 13.5cm
        while not rospy.is_shutdown():
            now = rospy.Time.now()
            if self.marker_pose and now - self.marker_last_time < timeout:
                # 每次都发布目标位置
                self.publish_marker_target(self.marker_pose, z_offset=0.135)
            else:
                rospy.loginfo("Marker lost. Stop moving.")
                break
            rate.sleep()
        """

        # 准备抓取
        # rospy.sleep(0.1)
        self.go_to_arm_pose(([x_mm, y_mm, z_mm, roll, pitch1, yaw]))

        # 闭合夹爪夹 pipette

        self.close_gripper()
        rospy.sleep(0.5)

        # 回到 pipette OBB 上方 5cm
        # self.go_to_pose_offset(self.pipette_pose, z_offset=0.05)
        self.go_to_arm_pose([x_mm, y_mm, z_mm2, roll, pitch, yaw])
        rospy.sleep(0.5)
        self.go_to_arm_pose([x_mm, y_mm1, z_mm1, roll, pitch, yaw])
        rospy.sleep(0.5)

    def attach_tip(self, tip_id=0):
        rospy.loginfo("Moving to tip rack...")
        x_list = [-44, -56, -67.8, -78, -87.5, -99.7]
        y_list = [261, 257, 259.8, 259.5, 259, 258.5]  # y_list[1] changed to 257 from 260

        x_mm = x_list[tip_id]
        y_mm = y_list[tip_id]
        z_mm = 366.7
        roll = 177.3
        pitch = 1.6
        yaw = -90.8

        # 移动到最边缘tip的正上方
        rospy.loginfo("Go to rack corner tip")
        self.go_to_arm_pose([x_mm, y_mm, z_mm, roll, pitch, yaw])
        rospy.sleep(2.0)
        rate = rospy.Rate(0.5)

        id = tip_id

        while not rospy.is_shutdown():
            y_mm_new = y_list[id]
            x_mm_new = x_list[id]
            if self.latest_color is None:
                rospy.logwarn("Waiting for color image...")
                rate.sleep()
                continue

            center = self.get_tip_roi_center()
            if center is None:
                rospy.logwarn("Waiting for color image...")
                rate.sleep()
                continue
            roi = self.crop_roi(self.latest_color, center)
            pred_class, conf = self.predict_tip_or_hole(roi)

            rospy.loginfo(f"Tip Recognition:: {pred_class} (conf={conf:2f})")

            # ---- 绘制可视化框 ----
            debug_img = self.latest_color.copy()
            size = self.roi_size_px
            x, y = int(center[0]), int(center[1])
            half = size // 2
            x1, y1 = max(x - half, 0), max(y - half, 0)
            x2, y2 = x1 + size, y1 + size

            color = (0, 255, 0) if (pred_class == 'tip' and conf > self.tip_conf_threshold) else (0, 0, 255)
            cv2.rectangle(debug_img, (x1, y1), (x2, y2), color, 2)
            label = f"{pred_class} ({conf:.2f})"
            cv2.putText(debug_img, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # ---- 发布调试图像 ----
            self.debug_image_pub.publish(self.bridge.cv2_to_imgmsg(debug_img, encoding="bgr8"))

            # ---- 判决动作 ----
            """
            if pred_class == 'tip' and conf > self.tip_conf_threshold:

                self.go_to_arm_pose([x_mm_new, y_mm_new, z_mm, roll, pitch, yaw])

            else:
                id = id + 1
                y_mm_new = y_list[id]
                x_mm_new = x_list[id]
                self.go_to_arm_pose([x_mm_new, y_mm_new, z_mm, roll, pitch, yaw])
                rate.sleep()
                continue
            """

            self.go_to_arm_pose([x_mm_new, y_mm_new, z_mm, roll, pitch, yaw])

            rospy.sleep(0.1)

            z_mm2 = z_mm
            rospy.loginfo("Attaching tip...")
            for i in range(3):
                z_mm2 = z_mm2 - 4
                self.go_to_arm_pose([x_mm_new, y_mm_new, z_mm2, roll, pitch, yaw])
                rospy.sleep(0.1)
            z_mm2 = z_mm - 14
            self.go_to_arm_pose([x_mm_new, y_mm_new, z_mm2, roll, pitch, yaw])
            rospy.sleep(0.1)
            # 向上移动

            for i in range(3):
                z_mm2 = z_mm2 + 4
                self.go_to_arm_pose([x_mm_new, y_mm_new, z_mm2, roll, pitch, yaw])
                rospy.sleep(0.1)

            z_mm3 = 445
            self.go_to_arm_pose([x_mm_new, y_mm_new, z_mm3, roll, pitch, yaw])
            return

    # volume unit :uL
    def aspirate(self, volume=500):
        rospy.loginfo("Go to aspiration beaker...")

        x_mm = -139.7
        y_mm = 349
        z_mm = 416.5
        roll = 175
        pitch = 25.4
        yaw = -90.5

        # 到beaker上方
        self.go_to_arm_pose([x_mm, y_mm, z_mm, roll, pitch, yaw])
        rospy.sleep(0.2)

        y_mm2 = y_mm + 10.4
        z_mm2 = z_mm - 81.7

        # 下降到beaker内部
        self.go_to_arm_pose([x_mm, y_mm2, z_mm2, roll, pitch, yaw])

        rospy.sleep(0.5)

        # 吸引液体
        rospy.loginfo(f"Aspirating liquid {volume} uL...")

        if volume >= 1000:
            volume = 1000
        self.pipetty_aspirate_pub.publish(Float32(volume))
        rospy.sleep(6.0)

        # 上升到beaker上方
        self.go_to_arm_pose([x_mm, y_mm, z_mm, roll, pitch, yaw])

    def aspirate_water(self, volume=500):
        rospy.loginfo("Go to aspiration beaker...")

        x_mm = -198.6
        y_mm = 310.1
        z_mm = 430.3
        roll = -179.2
        pitch = 29.1
        yaw = -56

        # 到beaker上方
        self.go_to_arm_pose([x_mm, y_mm, z_mm, roll, pitch, yaw])
        rospy.sleep(0.2)

        x_mm2 = -215.9
        y_mm2 = 334.7
        z_mm2 = 337.5

        # 下降到beaker内部
        self.go_to_arm_pose([x_mm2, y_mm2, z_mm2, roll, pitch, yaw])

        rospy.sleep(0.5)

        # 吸引液体
        rospy.loginfo("Aspirating liquid...")

        if volume >= 1000:
            volume = 1000
        self.pipetty_aspirate_pub.publish(Float32(volume))
        rospy.sleep(6.0)

        # 上升到beaker上方
        self.go_to_arm_pose([x_mm, y_mm, z_mm, roll, pitch, yaw])

    def dispense(self, id):
        rospy.loginfo("Go to dispense beaker...")

        if id == 1:
            x_mm = -190.1 + 2.5 + 7.5 - 1.5
            y_mm = 285.2 - 5 - 5
            z_mm = 407.7
            roll = 169.7
            pitch = 10.3
            yaw = -87.1
        else:
            x_mm = -194.9 + 5
            y_mm = 285.2
            z_mm = 417
            roll = 167.7
            pitch = 9.7
            yaw = -86.4

        # 到beaker上方
        self.go_to_arm_pose([x_mm, y_mm, z_mm, roll, pitch, yaw])

        rospy.sleep(0.5)

        # 下降到beaker内部
        if id == 1:
            z_mm2 = 343.7
        else:
            z_mm2 = 342.4

        self.go_to_arm_pose([x_mm, y_mm, z_mm2, roll, pitch, yaw])
        rospy.sleep(0.5)

        rospy.loginfo("Dispensing liquid...")

        self.pipetty_home_pub.publish(Bool(True))

        rospy.sleep(10)

        self.go_to_arm_pose([x_mm, y_mm, z_mm, roll, pitch, yaw])
        rospy.sleep(0.5)

    def dispose_of_tip(self):
        rospy.loginfo("Go to tip bin...")

        x_mm = -151.7
        y_mm = 231.3
        z_mm = 440.9
        roll = 175
        pitch = 4.4
        yaw = -90.5

        self.go_to_arm_pose([x_mm, y_mm, z_mm, roll, pitch, yaw])
        rospy.sleep(0.2)
        x_mm2 = x_mm + 333.1
        y_mm2 = y_mm - 27.9

        self.go_to_arm_pose([x_mm2, y_mm2, z_mm, roll, pitch, yaw])
        rospy.sleep(1.0)

        rospy.loginfo("Disposing of tip...")

        self.pipette_motor_pub.publish(Bool(True))

        rospy.sleep(4.0)

        self.go_to_arm_pose([x_mm, y_mm, z_mm, roll, pitch, yaw])

    def place_pipette(self):
        rospy.loginfo("Go to pipette stand...")

        x_mm = -259.9
        y_mm = 99.7
        z_mm = 416
        roll = 175
        pitch = 0.6
        yaw = -90.8

        self.go_to_arm_pose([x_mm, y_mm, z_mm, roll, pitch, yaw])
        rospy.sleep(0.5)

        rospy.loginfo("Placing pipette...")
        y_mm2 = y_mm + 81.3
        z_mm2 = z_mm - 34.7

        self.go_to_arm_pose([x_mm, y_mm2, z_mm2, roll, pitch, yaw])
        rospy.sleep(0.1)

        z_mm3 = z_mm2 - 44.6
        self.go_to_arm_pose([x_mm, y_mm2, z_mm3, roll, pitch, yaw])
        rospy.sleep(0.1)

        self.open_gripper()
        rospy.sleep(0.5)

        self.go_to_arm_pose([x_mm, y_mm, z_mm, roll, pitch, yaw])

    # -------------------- 占位：Aruco 查找 ------------------------

    def get_nearest_marker_pose(self):
        # TODO: 你补充你的 ArUco 查找逻辑，这里假设返回 PoseStamped
        pass

    # ------------------------- Tip Detection Utilities -------------------------

    def load_model(self):
        script_dir = os.path.dirname(os.path.realpath(__file__))
        model_path = os.path.join(script_dir, "1734448311.8315692", "best.pth")
        model = SimpleCNN()
        state = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state)
        model.to(self.device)
        model.eval()
        return model

    def crop_roi(self, image, center, size=None):
        if size is None:
            size = self.roi_size_px
        x, y = int(center[0]), int(center[1])
        half = size // 2
        x1, y1 = max(x - half, 0), max(y - half, 0)
        x2, y2 = x1 + size, y1 + size
        roi = image[y1:y2, x1:x2]
        if roi.shape[0] != size or roi.shape[1] != size:
            roi = cv2.resize(roi, (size, size), interpolation=cv2.INTER_LINEAR)
        return roi

    def predict_tip_or_hole(self, roi_bgr):
        # 关键：BGR → RGB
        roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.transform(roi_rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.model(tensor)
            softmax = F.softmax(out, dim=1)
            conf, pred = torch.max(softmax, 1)
        return ['hole', 'tip'][pred.item()], conf.item()

    def get_tip_roi_center(self):
        if self.latest_color is None:
            return None
        h, w = self.latest_color.shape[:2]
        c_x = int(722 / 1280 * w)
        c_y = int(178 / 720 * h)
        return (c_x, c_y)


# -------------------- 主函数 ------------------------

if __name__ == '__main__':
    controller = PipetteController()

    rospy.loginfo("[pipette_controller] ready, waiting for /pipette/dotwice service calls ...")
    rospy.spin()

    # controller.pick_pipette()
    # controller.attach_tip(tip_id=0)
    # controller.aspirate_water(volume=100)
    # controller.dispense()
    # controller.dispose_of_tip()
    # controller.place_pipette()
