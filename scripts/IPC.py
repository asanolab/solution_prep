#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from std_msgs.msg import Float32
from solution_prep.srv import PipetteDoTwice, TweezersDraw
import numpy as np


class SolutionOrchestrator(object):
    def __init__(self):
        rospy.init_node("solution_orchestrator", anonymous=True)

        # ===== parameters =====
        self.volume = 500  # uL

        # ===== Service client =====
        rospy.loginfo("Waiting for services ...")
        rospy.wait_for_service("/pipette/dotwice")
        rospy.wait_for_service("/tweezers/draw")
        self.srv_pipette = rospy.ServiceProxy("/pipette/dotwice", PipetteDoTwice)
        self.srv_tweezers = rospy.ServiceProxy("/tweezers/draw", TweezersDraw)
        rospy.loginfo("Services ready.")


        # ===== internal state =====
        self.tip_id = 0  # added per loop：0,1,2,...


    # ===== main =====
    def run(self):

        # pipette
        rospy.loginfo("Pipette two droplets")
        resp_p = self.srv_pipette(volume_ul=float(self.volume),tip_id=int(self.tip_id))
        if not resp_p.success:
            raise RuntimeError("pipette failed: " + resp_p.message)

        resp_t = self.srv_tweezers(draw=True)
        if not resp_t.success:
            raise RuntimeError("tweezers draw failed: " + resp_t.message)

        rospy.loginfo("IPC finished.")


if __name__ == "__main__":
    try:
        SolutionOrchestrator().run()
    except Exception as e:
        rospy.logerr("IPC error: %s", e)
