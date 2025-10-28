#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import tf.transformations as tf_trans

from std_msgs.msg import Float32
from geometry_msgs.msg import PoseStamped
from my_robot_msgs.msg import MovePose, LabwareOBB  # ← LabwareOBB 如在别的包请改导入
from my_robot_msgs.srv import BeakerMani, BeakerManiResponse


class BeakerManipulator:
    def __init__(self):
        rospy.init_node('beaker_manipulator', anonymous=True)

        # ---------------- 参数（可在 launch 覆盖） ----------------
        self.beaker_obb_topic = rospy.get_param("~beaker_obb_topic", "/labware/beaker0/obb")

        # 高度/偏置（m）
        self.tool_z_offset = float(rospy.get_param("~tool_z_offset", 0.055))  # 末端工具厚度补偿
        self.safe_z_up = float(rospy.get_param("~safe_z_up", 0.080))  # 安全高度（相对烧杯顶面）
        self.grasp_z_down = float(rospy.get_param("~grasp_z_down", 0.010))  # 抓取时再向下的深度（靠近杯沿/把手）

        # 姿态（rad）：保持和你之前逻辑一致
        self.roll_rad = 179.1 * np.pi / 180.0
        self.pitch_rad = 5.6 * np.pi / 180.0
        self.yaw_base = -np.pi / 2  # -90deg 基准，加上 OBB yaw

        # 夹爪开合（你的系统似乎用“距离”单位，保持和 pipette 一致）
        self.grip_open = float(rospy.get_param("~grip_open", 52.0))
        self.grip_close = float(rospy.get_param("~grip_close", 0.0))

        # 摇晃参数（单位：mm / 次 / Hz）
        self.shake_mode = rospy.get_param("~shake_mode", "line")  # "line" / "circle"
        self.shake_axis = rospy.get_param("~shake_axis", "x")  # line 模式下的轴："x" 或 "y"
        self.shake_amplitude = float(rospy.get_param("~shake_amplitude_mm", 15.0))  # 摇晃振幅（±mm）
        self.shake_height_up = float(rospy.get_param("~shake_height_up_mm", 40.0))  # 抓取后提升高度（mm）
        self.shake_cycles = int(rospy.get_param("~shake_cycles", 6))  # 往返次数（line）/ 圈数（circle）
        self.shake_hz = float(rospy.get_param("~shake_hz", 2.0))  # 摇晃频率（约）

        # ---------------- 话题 ----------------
        rospy.Subscriber(self.beaker_obb_topic, LabwareOBB, self.cb_beaker_obb)
        rospy.Subscriber("/marker_pose_base/0", PoseStamped, self.cb_marker_0)
        rospy.Subscriber("/marker_pose_base/1", PoseStamped, self.cb_marker_1)
        rospy.Subscriber("/marker_pose_base/2", PoseStamped, self.cb_marker_2)
        rospy.Subscriber("/marker_pose_base/3", PoseStamped, self.cb_marker_3)
        rospy.Subscriber("/marker_pose_base/4", PoseStamped, self.cb_marker_4)
        self.pose_pub = rospy.Publisher("/arm_control/move_pose", MovePose, queue_size=1)
        self.gripper_pub = rospy.Publisher("/gripper/target_distance", Float32, queue_size=10)

        # 服务
        self.srv_shake = rospy.Service('/beaker/shake', BeakerMani, self.cb_shake)
        rospy.loginfo("[beaker_manipulator] ready. Waiting for /beaker/shake calls...")

        # ---------------- 数据缓存 ----------------
        self.beaker_obb = None
        # marker
        self.maker_pose_0 = None
        self.maker_pose_1 = None
        self.maker_pose_2 = None
        self.maker_pose_3 = None
        self.maker_pose_4 = None

        rospy.loginfo("[BeakerManipulator] ready.")

    # ---------------- 回调 ----------------
    def cb_beaker_obb(self, msg):
        self.beaker_obb = msg

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

    def cb_shake(self, req):
        try:
            if not bool(req.shake):
                rospy.loginfo("[beaker/shake] shake==false, no-op.")
                return BeakerManiResponse(success=True, message="No shake command")

            # 执行固定的 4 步
            if self.run_once():
                return BeakerManiResponse(success=True, message="beaker shake once")
            else:
                return BeakerManiResponse(success=False, message="beaker shake failed")
        except Exception as e:
            rospy.logerr(f"[beaker/shake] failed: {e}")
            return BeakerManiResponse(success=False, message=str(e))

    # ---------------- 基础工具 ----------------
    def _pose_from_obb(self, obb_msg, x_off_m=0.0, y_off_m=0.0, z_off_m=0.0, yaw_from_obb=True):
        base = obb_msg.pose
        # 位置（mm）
        x_mm = (base.position.x + x_off_m) * 1000.0
        y_mm = (base.position.y + y_off_m) * 1000.0
        # 高度优先用 z_height（顶面）+ 偏置 + 工具厚度
        top_z = getattr(obb_msg, "z_height", 0.0) or base.position.z
        z_mm = (top_z + z_off_m + self.tool_z_offset) * 1000.0

        # yaw：-90° + OBB_yaw（并按你的规则修正 >180°）
        yaw = self.yaw_base
        if yaw_from_obb:
            q = [base.orientation.x, base.orientation.y, base.orientation.z, base.orientation.w]
            _, _, yaw_obb = tf_trans.euler_from_quaternion(q)
            if yaw_obb > np.pi:
                yaw_obb -= np.pi
            yaw = self.yaw_base + yaw_obb

        return [x_mm, y_mm, z_mm, self.roll_rad, self.pitch_rad, yaw]

    def send_arm_pose(self, pose_mmrad, wait=0.6):
        """pose_mmrad: [x_mm,y_mm,z_mm,roll_rad,pitch_rad,yaw_rad]"""
        msg = MovePose()
        # 防止 numpy.float64 导致序列化问题，转 float
        msg.pose = [float(pose_mmrad[0]), float(pose_mmrad[1]), float(pose_mmrad[2]),
                    float(pose_mmrad[3]), float(pose_mmrad[4]), float(pose_mmrad[5])]
        self.pose_pub.publish(msg)
        rospy.sleep(wait)

    def open_gripper(self, distance=None):
        if distance is None: distance = self.grip_open
        self.gripper_pub.publish(Float32(distance))
        rospy.sleep(0.4)

    def close_gripper(self, distance=None):
        if distance is None: distance = self.grip_close
        self.gripper_pub.publish(Float32(distance))
        rospy.sleep(0.4)

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
        rospy.sleep(0.5)

    # ---------------- 主流程 ----------------
    def move_above_beaker(self):
        """1) 前往目标烧杯正上方（安全高度）"""
        rospy.loginfo("Go to target beaker...")
        self.open_gripper()
        rospy.sleep(2)
        x_mm = -118.1
        y_mm = 262.7
        z_mm = 265.3
        roll = 177.8
        pitch = 3.8
        yaw = -90.7

        self.go_to_arm_pose([x_mm, y_mm, z_mm, roll, pitch, yaw])
        rospy.sleep(2)

        x_mm2 = -140.1
        y_mm2 = 262.7
        z_mm2 = 230.3
        roll2 = 177.8
        pitch2 = 3.8
        yaw2 = -90.7

        self.go_to_arm_pose([x_mm2, y_mm2, z_mm2, roll2, pitch2, yaw2])
        rospy.sleep(2)
        return True

    def grasp_beaker(self):
        """2) 下降抓取烧杯（靠近杯沿/把手，视夹爪形态微调 x/y_off 或 z_off）"""
        rospy.loginfo("Grasping beaker...")

        x_mm = -218.7
        y_mm = 331.7
        z_mm = 105.4
        roll = -175.8
        pitch = 1.3
        yaw = -0.2

        self.go_to_arm_pose([x_mm, y_mm, z_mm, roll, pitch, yaw])
        rospy.sleep(3.0)
        # b. 闭合夹爪
        self.close_gripper()
        rospy.sleep(2)
        return True

    def lift_and_shake(self):
        """3) 上升一点距离后摇晃烧杯"""

        # a. 抬到摇晃高度
        x_mm = -218.7
        y_mm = 331.7
        z_mm = 171.3
        roll = -175.8
        pitch = 1.3
        yaw = -0.2
        roll2 = 174.4
        roll3 = -169.7

        pose = []
        pose0 = [x_mm, y_mm, z_mm, roll, pitch, yaw]
        pose1 = [x_mm, y_mm, z_mm, roll2, pitch, yaw]
        pose2 = [x_mm, y_mm, z_mm, roll3, pitch, yaw]

        pose.append(pose1)
        pose.append(pose2)

        self.go_to_arm_pose(pose0)
        rospy.sleep(0.5)

        # b. 生成摇晃轨迹并执行
        rospy.loginfo("Shaking beaker...")
        for i in range(6):
            id = i % 2
            self.go_to_arm_pose(pose[id])
            rospy.sleep(0.1)
        rospy.sleep(2.0)
        return True

    def place_beaker(self):
        """4) 下降放下烧杯并松爪，回到上方"""
        rospy.loginfo("Place beaker...")
        # a. 下降到放置高度（接近桌面/原位）
        x_mm = -218.7
        y_mm = 331.7
        z_mm = 105.4
        roll = -175.8
        pitch = 1.3
        yaw = -0.2

        self.go_to_arm_pose([x_mm, y_mm, z_mm, roll, pitch, yaw])
        rospy.sleep(3)
        # b. 松开夹爪
        self.open_gripper()
        rospy.sleep(2)

        # c. 回到上方安全高度

        x_mm2 = -119.4
        z_mm2 = 308.9

        self.go_to_arm_pose([x_mm2, y_mm, z_mm2, roll, pitch, yaw])
        rospy.sleep(2.0)
        return True

    # ---------------- 摇晃轨迹 ----------------
    def _shake_line(self, center_pose, axis="x"):
        """
        在基坐标系的 X 或 Y 方向做来回直线摇晃。
        center_pose: [x_mm, y_mm, z_mm, r, p, y]
        """
        amp = float(self.shake_amplitude)  # mm（±）
        hz = max(0.5, float(self.shake_hz))
        cycles = max(1, int(self.shake_cycles))
        dt = 1.0 / (hz * 6.0)  # 每个半周期 3~6 个点
        x0, y0, z0, r, p, yaw = center_pose

        for k in range(cycles * 2):  # 往返
            sgn = 1.0 if (k % 2 == 0) else -1.0
            steps = int(max(3, np.ceil((1.0 / hz) / dt)))
            for i in range(steps):
                alpha = float(i) / steps  # 0 → 1
                offset = sgn * amp * np.sin(alpha * np.pi)  # 平滑起停
                x = x0 + offset if axis == "x" else x0
                y = y0 + offset if axis == "y" else y0
                self.send_arm_pose([x, y, z0, r, p, yaw], wait=dt)

    def _shake_circle(self, center_pose):
        """
        小圆圈摇晃（半径 = amplitude）
        """
        rad = float(self.shake_amplitude)  # 半径（mm）
        hz = max(0.5, float(self.shake_hz))
        cycles = max(1, int(self.shake_cycles))
        steps_per_circle = int(max(12, np.ceil(2 * np.pi * rad / 4.0)))  # 每 4mm 一个点
        dt = 1.0 / (hz * (steps_per_circle / 2.0))  # 近似频率控制

        x0, y0, z0, r, p, yaw = center_pose
        for _ in range(cycles):
            for i in range(steps_per_circle):
                theta = 2.0 * np.pi * (float(i) / steps_per_circle)
                x = x0 + rad * np.cos(theta)
                y = y0 + rad * np.sin(theta)
                self.send_arm_pose([x, y, z0, r, p, yaw], wait=dt)

    # ---------------- 一键执行 ----------------
    def run_once(self):
        if not self.move_above_beaker(): return False
        if not self.grasp_beaker():      return False
        if not self.lift_and_shake():    return False
        self.place_beaker()
        return True


if __name__ == "__main__":
    node = BeakerManipulator()
    #rospy.sleep(2.0)  # 等 OBB
    #node.run_once()

    rospy.spin()
