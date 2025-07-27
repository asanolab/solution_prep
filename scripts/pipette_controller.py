#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from std_msgs.msg import Header, Float32
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from my_robot_msgs.msg import LabwareOBB
from moveit_commander import MoveGroupCommander, RobotCommander
from tf.transformations import quaternion_from_euler
from my_robot_msgs.msg import MovePose
from xarm_msgs.srv import Move
import numpy as np
from std_msgs.msg import Bool
import tf.transformations as tf_trans


class PipetteController:
    def __init__(self):
        rospy.init_node('pipette_controller', anonymous=True)

        # 订阅 OBB
        rospy.Subscriber("/labware/pipette_rack/obb", LabwareOBB, self.cb_pipette_rack)
        rospy.Subscriber("/labware/pipette0/obb", LabwareOBB, self.cb_pipette)
        rospy.Subscriber("/labware/hcl_beaker/obb", LabwareOBB, self.cb_hcl_beaker)
        rospy.Subscriber("/labware/target_beaker/obb", LabwareOBB, self.cb_target_beaker)
        rospy.Subscriber('/marker_pose_base/0', PoseStamped, self.cb_marker)

        self.offset = 0.055

        # OBB 数据缓存
        self.pipette_rack_pose = None
        self.pipette_pose = None
        self.hcl_beaker_pose = None
        self.target_beaker_pose = None
        self.marker_pose = None
        self.marker_last_time = rospy.Time.now()

        # 机械臂控制
        self.pose_pub = rospy.Publisher("/arm_control/move_pose", MovePose, queue_size=1)

        # 夹爪控制（占位，根据你的实际控制方法替换）
        self.gripper_pub = rospy.Publisher('/gripper/current_distance', Float32, queue_size=10)

    # OBB 回调
    def cb_pipette_rack(self, msg):
        self.pipette_rack_pose = msg

    def cb_pipette(self, msg):
        self.pipette_pose = msg

    def cb_hcl_beaker(self, msg):
        self.hcl_beaker_pose = msg

    def cb_target_beaker(self, msg):
        self.target_beaker_pose = msg

    def cb_marker(self, msg):
        self.marker_pose = msg.pose
        self.marker_last_time = rospy.Time.now()

    # -------------------- 各步骤 ------------------------

    # ----------------- 机械臂控制 -----------------------

    def go_to_pose_offset(self, base_pose, z_offset=0.0):  # z: 单位m
        """将 OBB pose + z 偏移 → 转换为机械臂 MovePose"""
        # OBB 的位置
        x_mm = base_pose.pose.position.x * 1000
        y_mm = base_pose.pose.position.y * 1000
        z_mm = (base_pose.z_height + z_offset + self.offset) * 1000

        # OBB 四元数 → 欧拉角，取 yaw
        quat = [
            base_pose.pose.orientation.x,
            base_pose.pose.orientation.y,
            base_pose.pose.orientation.z,
            base_pose.pose.orientation.w
        ]
        _, _, yaw_obb = tf_trans.euler_from_quaternion(quat)

        if yaw_obb > 3.1415926:  # > π
            yaw_obb = yaw_obb - 3.1415926

        # 固定 r, p，动态 yaw
        roll = 179.1 * 3.1415926 / 180
        pitch = 5.6 * 3.1415926 / 180
        # yaw = -1.5707963 + yaw_obb  # -90° + OBB yaw
        yaw = -1.57

        # 发布 MovePose
        pose = MovePose()
        pose.pose = [x_mm, y_mm, z_mm, roll, pitch, yaw]

        print("pose_msg =", pose)
        print("pose_msg.pose type =", type(pose.pose))
        print("pose_msg.pose contents =", pose.pose)

        self.pose_pub.publish(pose)
        rospy.sleep(2.0)

    # ----------------- 夹爪控制 -------------------------
    def open_gripper(self, distance=0.0):
        rospy.loginfo(f"Opening gripper to {distance}m")
        self.gripper_pub.publish(Float32(distance))
        rospy.sleep(1.0)

    def close_gripper(self, distance=52.0):
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
        if self.pipette_pose is None:
            rospy.logwarn("No pipette OBB received yet.")
            return

        # 1️⃣ 到 pipette OBB 上方 5cm
        self.go_to_pose_offset(self.pipette_pose, z_offset=0.05)

        # 2️⃣ 找到最近 ArUco marker，返回 marker pose
        marker_pose = self.get_nearest_marker_pose()

        rospy.loginfo("Searching for marker...")
        timeout = rospy.Duration(3.0)  # 超时时间，自行调整
        rate = rospy.Rate(10)
        # 3️⃣ 去 marker 上方 13.5cm
        while not rospy.is_shutdown():
            now = rospy.Time.now()
            if self.marker_pose and now - self.marker_last_time < timeout:
                # 每次都发布目标位置
                self.publish_marker_target(self.marker_pose, z_offset=0.135)
            else:
                rospy.loginfo("Marker lost. Stop moving.")
                break
            rate.sleep()

            # 闭合夹爪夹 pipette
        self.close_gripper(distance=0.01)

        # 回到 pipette OBB 上方 5cm
        self.go_to_pose_offset(self.pipette_pose, z_offset=0.05)

        # 4️⃣ 先开一遍再闭合夹爪夹 pipette
        self.open_gripper()
        self.close_gripper()

        self.go_to_pose_offset(marker_pose, z_offset=0)

        # 5️⃣ 回到 pipette OBB 上方 5cm
        self.go_to_pose_offset(self.pipette_pose, z_offset=0.05)

        # 1️⃣ 先去 pipette obb 上方 5cm
        self.go_to_pose_offset(self.pipette_pose, z_offset=0.05)

        # 2️⃣ 查找 ArUco 最近 marker
        # TODO: 这里你自己补充你的 marker 查找逻辑
        marker_pose = self.get_nearest_marker_pose()

        # 3️⃣ 去 marker 上方 13.5cm
        self.go_to_pose_offset(marker_pose, z_offset=0.135)

        # 4️⃣ 闭合夹爪
        self.close_gripper()

        # 5️⃣ 回到 pipette obb 上方 5cm
        self.go_to_pose_offset(self.pipette_pose, z_offset=0.05)

    def attach_tip(self):
        rospy.loginfo("Attaching pipette tip...")
        # TODO: 你的具体动作逻辑实现
        pass

    def aspirate_hcl(self):
        rospy.loginfo("Aspirating HCl...")
        # TODO: 你的具体动作逻辑实现
        pass

    def dispense_to_beaker(self):
        rospy.loginfo("Dispensing to target beaker...")
        # TODO: 你的具体动作逻辑实现
        pass

    def discard_tip(self):
        rospy.loginfo("Discarding tip...")
        # TODO: 机械臂去固定 pose 丢弃
        pass

    def place_pipette(self):
        rospy.loginfo("Placing pipette back to rack...")
        # TODO: 你的具体动作逻辑实现
        pass

    # -------------------- 占位：Aruco 查找 ------------------------

    def get_nearest_marker_pose(self):
        # TODO: 你补充你的 ArUco 查找逻辑，这里假设返回 PoseStamped
        pose = PoseStamped()
        pose.header.frame_id = "link_base"
        pose.pose.position.x = 0.3
        pose.pose.position.y = 0.2
        pose.pose.position.z = 0.0
        quat = quaternion_from_euler(0, 0, 0)
        pose.pose.orientation = Quaternion(*quat)
        return pose


# -------------------- 主函数 ------------------------

if __name__ == '__main__':
    controller = PipetteController()
    rospy.sleep(2.0)  # 等待 OBB 收到

    controller.pick_pipette()
    controller.attach_tip()
    controller.aspirate_hcl()
    controller.dispense_to_beaker()
    controller.discard_tip()
    controller.place_pipette()
