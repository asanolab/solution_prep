#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import tf.transformations as tf_trans

from std_msgs.msg import Float32
from geometry_msgs.msg import PoseStamped
from lite6_labauto.msg import MovePose
from lite6_labauto.msg import LabwareOBB  # ← 如果你的 LabwareOBB 在别的包，请改这里
from lite6_labauto.srv import pHMeasure, pHMeasureResponse


class PhSensorController:
    def __init__(self):
        rospy.init_node('ph_sensor_controller', anonymous=True)

        # ---------------- 参数 ----------------
        # OBB 话题（可在 launch 里覆盖）
        self.ph_sensor_obb_topic = rospy.get_param("~ph_sensor_obb", "/labware/pH_sensor3/obb")
        # self.ph_rack_obb_topic     = rospy.get_param("~ph_rack_obb",     "/labware/ph_rack/obb")
        self.target_beaker_obb_topic = rospy.get_param("~target_beaker_obb", "/labware/beaker0/obb")

        # pH 值话题（字符串/浮点都可，这里默认 Float32；若你是别的类型请改回调与缓存）
        self.ph_topic = rospy.get_param("~ph_topic", "/pH_sensor/pH_value")

        # 高度/偏置参数（单位：m）
        self.safe_z_up = float(rospy.get_param("~safe_z_up", 0.080))  # 起落的安全上方高度（相对 OBB 顶面/桌面）
        self.lower_depth = float(rospy.get_param("~lower_depth", 0.060))  # 插入深度（相对 OBB 顶面/桌面向下）
        self.tool_z_offset = float(rospy.get_param("~tool_z_offset", 0.055))  # 机械末端工具厚度补偿(与你 pipette 的 offset 一致)
        self.approach_y_offset = float(rospy.get_param("~approach_y_offset", 0.0))  # 需要时可加一点Y侧向避让
        self.wait_after_lower = float(rospy.get_param("~wait_after_lower", 1.0))  # 下沉后等待秒数
        self.ph_timeout = float(rospy.get_param("~ph_timeout", 2.0))  # 读取 pH 的超时时间

        # marker
        self.maker_pose_0 = None
        self.maker_pose_1 = None
        self.maker_pose_2 = None
        self.maker_pose_3 = None
        self.maker_pose_4 = None

        # 抓取姿态（和你 pipette 的约定一致，rad）
        self.roll_rad = 179.1 * np.pi / 180.0
        self.pitch_rad = 5.6 * np.pi / 180.0
        self.yaw_base = -np.pi / 2  # -90deg 基准

        # 夹爪开合（单位与你系统一致，这里沿用你最近代码里用的 target_distance）
        self.grip_open = float(rospy.get_param("~grip_open", 52.0))
        self.grip_close = float(rospy.get_param("~grip_close", 0.0))

        # ---------------- 订阅/发布 ----------------
        rospy.Subscriber(self.ph_sensor_obb_topic, LabwareOBB, self.cb_ph_sensor_obb)
        # rospy.Subscriber(self.ph_rack_obb_topic,     LabwareOBB, self.cb_ph_rack_obb)
        rospy.Subscriber(self.target_beaker_obb_topic, LabwareOBB, self.cb_target_beaker_obb)
        rospy.Subscriber(self.ph_topic, Float32, self.cb_ph_value)
        rospy.Subscriber("/marker_pose_base/0", PoseStamped, self.cb_marker_0)
        rospy.Subscriber("/marker_pose_base/1", PoseStamped, self.cb_marker_1)
        rospy.Subscriber("/marker_pose_base/2", PoseStamped, self.cb_marker_2)
        rospy.Subscriber("/marker_pose_base/3", PoseStamped, self.cb_marker_3)
        rospy.Subscriber("/marker_pose_base/4", PoseStamped, self.cb_marker_4)

        self.pose_pub = rospy.Publisher("/arm_control/move_pose", MovePose, queue_size=1)
        self.gripper_pub = rospy.Publisher("/gripper/target_distance", Float32, queue_size=1)
        self.ph_pub = rospy.Publisher("/solution/current_ph", Float32, queue_size=10, latch=True)

        # 服务
        self.srv_measure = rospy.Service("/ph/measure", pHMeasure, self.cb_measure)
        rospy.loginfo("[ph_sensor_controller] ready. Waiting for /ph/measure calls...")

        # ---------------- 数据缓存 ----------------
        self.ph_sensor_pose = None  # LabwareOBB
        self.ph_rack_pose = None  # LabwareOBB
        self.target_beaker_pose = None  # LabwareOBB

        self.last_ph_value = None
        self.last_ph_time = rospy.Time(0)

        rospy.loginfo("[ph_sensor_controller] ready.")

    # ---------------- 回调 ----------------
    def cb_ph_sensor_obb(self, msg):
        self.ph_sensor_pose = msg

    # def cb_ph_rack_obb(self, msg):          self.ph_rack_pose = msg
    def cb_target_beaker_obb(self, msg):
        self.target_beaker_pose = msg

    def cb_ph_value(self, msg):
        self.last_ph_value = float(msg.data)
        self.last_ph_time = rospy.Time.now()

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

    def cb_measure(self, req):
        try:
            # 参数：超时优先用请求，否则用节点默认
            timeout_s = float(req.timeout_s) if req.timeout_s > 0.0 else float(self.ph_timeout)

            # 资源就绪

            # == 整套流程 ==
            if not self.pick_ph_sensor():
                return pHMeasureResponse(False, float('nan'), "pick failed")

            # 为这一次测量设置临时超时
            old_timeout = self.ph_timeout
            self.ph_timeout = timeout_s
            try:
                ph = self.measure_in_target()  # 内部会 wait_and_get_ph 并发布 /solution/current_ph (latched)
            finally:
                self.ph_timeout = old_timeout

            self.place_back()

            if ph is None or np.isnan(ph):
                return pHMeasureResponse(False, float('nan'), "timeout/no data")
            else:
                return pHMeasureResponse(True, float(ph), f"Measure pH value once: {ph:.2f}")

        except Exception as e:
            rospy.logerr(f"[ph/measure] failed: {e}")
            return pHMeasureResponse(False, float('nan'), str(e))

    # ---------------- 基本动作 ----------------
    def open_gripper(self, distance=None):
        if distance is None: distance = self.grip_open
        self.gripper_pub.publish(Float32(distance))
        rospy.sleep(1.5)

    def close_gripper(self, distance=None):
        if distance is None: distance = self.grip_close
        self.gripper_pub.publish(Float32(distance))
        rospy.sleep(1.5)

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

    def send_arm_pose_mmdeg(self, x_mm, y_mm, z_mm, roll_deg, pitch_deg, yaw_deg, wait=1.5):
        """调试/固定姿态用：角度转弧度，直接发 MovePose"""
        pose = MovePose()
        pose.pose = [
            float(x_mm), float(y_mm), float(z_mm),
            float(roll_deg * np.pi / 180.0),
            float(pitch_deg * np.pi / 180.0),
            float(yaw_deg * np.pi / 180.0),
        ]
        self.pose_pub.publish(pose)
        rospy.sleep(wait)

    def go_to_pose_from_obb(self, obb_msg, x_off=0.0, y_off=0.0, z_off=0.0, yaw_from_obb=True, wait=1.5):
        """
        用 LabwareOBB 的中心/姿态 + 偏置生成末端位姿并发布。
        偏置单位：x/y/z (m)；最终发送前换成 mm。
        yaw_from_obb=True 时：yaw = -90° + yaw_obb
        """
        base = obb_msg.pose
        x_mm = (base.position.x + x_off) * 1000.0
        y_mm = (base.position.y + y_off) * 1000.0
        # 采用 z_height 作为器具顶面高度（如无则用 position.z），再 + z_off + 工具厚度
        top_z = getattr(obb_msg, "z_height", 0.0) or base.position.z
        z_mm = (top_z + z_off + self.tool_z_offset) * 1000.0

        # yaw 跟随 OBB
        yaw = 0.0
        if yaw_from_obb:
            q = [base.orientation.x, base.orientation.y, base.orientation.z, base.orientation.w]
            _, _, yaw_obb = tf_trans.euler_from_quaternion(q)
            if yaw_obb > np.pi:  # > 180°
                yaw_obb -= np.pi  # 你之前的修正规则
            yaw = self.yaw_base + yaw_obb
        else:
            yaw = self.yaw_base

        pose = MovePose()
        pose.pose = [x_mm, y_mm, z_mm, self.roll_rad, self.pitch_rad, yaw]
        self.pose_pub.publish(pose)
        rospy.sleep(wait)

    # ---------------- 高阶流程 ----------------
    def pick_ph_sensor(self):
        """1) 前往抓取 pH sensor：先到架上方，再下降抓取，再抬起"""

        rospy.loginfo("Picking pH sensor...")
        # a. 架上方安全高度
        self.open_gripper()

        x_mm = -47.2
        y_mm = 328.3
        z_mm = 347.8
        roll = 178
        pitch = 3.8
        yaw = -90.7

        self.go_to_arm_pose([x_mm, y_mm, z_mm, roll, pitch, yaw])
        rospy.sleep(2.0)

        y_mm2 = 388
        z_mm2 = 207
        # b. 下降到传感器中心附近
        self.go_to_arm_pose([x_mm, y_mm2, z_mm2, roll, pitch, yaw])
        rospy.sleep(2)
        self.close_gripper()

        # c. 抬回安全高度
        z_mm3 = 291.6
        self.go_to_arm_pose([x_mm, y_mm2, z_mm3, roll, pitch, yaw])
        return True

    def measure_in_target(self):
        """2) 前往目标位置并下沉；3) 等待1s并读取 pH；4) 上升"""

        rospy.loginfo("Go to measure ph...")

        rospy.loginfo("Go to beaker...")
        # a. 到目标烧杯上方安全高度
        x_mm = -249.1
        y_mm = 310.6
        z_mm = 319.3
        roll = 177.3
        pitch = 7.5
        yaw = -63.8

        self.go_to_arm_pose([x_mm, y_mm, z_mm, roll, pitch, yaw])
        rospy.sleep(1.5)

        # b. 下沉到液面/液内（这里用 lower_depth，按你的实际高度定义场景）
        rospy.loginfo("Measuring...")
        z_mm2 = 200
        self.go_to_arm_pose([x_mm, y_mm, z_mm2, roll, pitch, yaw])
        rospy.sleep(0.1)

        while True:
            z_mm2 = z_mm2 - 1
            self.go_to_arm_pose([x_mm, y_mm, z_mm2, roll, pitch, yaw])
            rospy.sleep(0.1)

            if z_mm2 <= 181.7:
                z_mm2 = 181.7
                self.go_to_arm_pose([x_mm, y_mm, z_mm2, roll, pitch, yaw])
                rospy.sleep(0.1)
                break

        # c. 等待 & 取 pH
        rospy.sleep(12.0)
        ph_value = self.wait_and_get_ph(timeout=self.ph_timeout)

        # d. 上升回安全高度
        z_mm3 = z_mm2
        for i in range(5):
            z_mm3 = z_mm3 + 2
            self.go_to_arm_pose([x_mm, y_mm, z_mm3, roll, pitch, yaw])
        self.go_to_arm_pose([x_mm, y_mm, z_mm, roll, pitch, yaw])

        return ph_value

    def place_back(self):
        """把 pH 计放回架子"""

        # a. 去架上方
        x_mm = -49.2
        y_mm = 385.6
        z_mm = 317
        roll = 177.8
        pitch = 8.1
        yaw = -90.7
        self.go_to_arm_pose([x_mm, y_mm, z_mm, roll, pitch, yaw])
        rospy.sleep(1.5)

        # b. 下降到放置高度
        z_mm2 = 247.8
        self.go_to_arm_pose([x_mm, y_mm, z_mm2, roll, pitch, yaw])
        rospy.sleep(2.0)

        # c. 松开并上抬
        for i in range(10):
            z_mm3 = z_mm2 - i - 1
            self.go_to_arm_pose([x_mm, y_mm, z_mm3, roll, pitch, yaw])
            rospy.sleep(0.1)

        z_mm3 = 210
        self.go_to_arm_pose([x_mm, y_mm, z_mm3, roll, pitch, yaw])
        rospy.sleep(2.5)

        self.open_gripper()

        z_mm4 = 334
        self.go_to_arm_pose([x_mm, y_mm, z_mm4, roll, pitch, yaw])
        return True

    # ---------------- pH 读取 ----------------
    def wait_and_get_ph(self, timeout=2.0):
        """
        在下降插入后，等待直到 timeout 秒内收到最近一次 pH 值；
        若超时返回 None。
        """
        start = rospy.Time.now()
        last_seen = self.last_ph_time
        rate = rospy.Rate(50)
        while not rospy.is_shutdown() and (rospy.Time.now() - start).to_sec() < timeout:
            # 收到新值
            if self.last_ph_time > last_seen:
                rospy.loginfo(f"[ph] value = {self.last_ph_value}")
                self.ph_pub.publish(Float32(self.last_ph_value))
                return self.last_ph_value
            rate.sleep()
        rospy.logwarn("[ph] timeout, no new value.")
        # 退而求其次：返回最近一次（可能旧）
        rospy.loginfo(f"[ph] value = {self.last_ph_value}")
        self.ph_pub.publish(Float32(self.last_ph_value))
        return self.last_ph_value


# ---------------- 主程序 ----------------
if __name__ == "__main__":
    ctl = PhSensorController()
    # rospy.sleep(2.0)  # 等 OBB/pH 话题起来
    rospy.spin()

    # 1. 抓取 pH 计
    # if ctl.pick_ph_sensor():
    #     # 2~4. 去目标 → 下沉 → 读 pH → 上升
    #     ph = ctl.measure_in_target()
    #     rospy.loginfo(f"[FLOW] measured pH: {ph}")

    #     # 放回
    #     ctl.place_back()
