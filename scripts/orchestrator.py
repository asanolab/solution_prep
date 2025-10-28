#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from std_msgs.msg import Float32
from my_robot_msgs.srv import PipetteDo, BeakerMani, pHMeasure
import numpy as np


class SolutionOrchestrator(object):
    def __init__(self):
        rospy.init_node("solution_orchestrator", anonymous=True)

        # ===== 前提输入参数（必填/建议在 launch 里给）=====
        self.V0_ml = float(rospy.get_param("~current_water_volume_ml", 20))  # 初始水体积 mL
        self.target_ph = float(rospy.get_param("~target_ph", 4.00))  # 目标 pH
        self.c_HCl = float(rospy.get_param("~c_HCl", 0.01585))  # HCl 原液浓度 mol/L
        self.ph_tol = float(rospy.get_param("~ph_tolerance", 0.02))  # pH 误差容忍
        self.max_loops = int(rospy.get_param("~max_loops", 6))  # 最大迭代次数

        # ===== 可调策略参数（暂时用于占位/调试）=====
        # 初次加注与后续加注的“固定体积”，你之后会用算法替换
        self.first_dose_ul = float(rospy.get_param("~first_dose_ul", 200.0))

        self.next_dose_ul = float(rospy.get_param("~next_dose_ul", 100.0))
        self.allow_water = bool(rospy.get_param("~allow_water", True))  # 若为 True，低于目标时用water微调（需要你实现water路径）

        # ===== Service 代理 =====
        rospy.loginfo("Waiting for services ...")
        rospy.wait_for_service("/pipette/do")
        rospy.wait_for_service("/beaker/shake")
        rospy.wait_for_service("/ph/measure")
        self.srv_pipette = rospy.ServiceProxy("/pipette/do", PipetteDo)
        self.srv_shake = rospy.ServiceProxy("/beaker/shake", BeakerMani)
        self.srv_measure = rospy.ServiceProxy("/ph/measure", pHMeasure)
        rospy.loginfo("Services ready.")

        # ===== 控制策略选择 =====
        # strategy: "closed_form"（每次都用闭式解） 或 "mixed"（第一次闭式解×alpha，其余用PI）
        self.control_strategy = rospy.get_param("~control_strategy", "mixed")  # "closed_form" | "mixed"


        # ===== PI 参数（在 [H+] 空间）=====
        # 建议让增益随总体积缩放：Kp = kp_base * V(L), Ki = ki_base * V(L/step)
        self.kp_base = float(rospy.get_param("~kp_base", 1.0))

        # 酸侧保守系数（防过冲）
        # ===== 酸侧保护/偏好 =====
        self.alpha_acid = float(rospy.get_param("~alpha_acid", 0.80))  # 酸侧统一保守系数
        self.acid_dph_cap = float(rospy.get_param("~acid_dph_cap", 0.12))  # 本轮酸最多允许的 |ΔpH|
        self.no_acid_near_eps = float(rospy.get_param("~no_acid_near_eps", 0.05))  # 距目标 < 此 pH，优先禁酸


        # ===== 体积约束 =====
        self.pipette_max_ul = float(rospy.get_param("~pipette_max_ul", 1000.0))
        self.min_step_ul = float(rospy.get_param("~min_step_ul", 10.0))
        self.safety_cap_ul = float(rospy.get_param("~safety_cap_ul", 2000.0))

        # ===== 发布当前总体积（latched）=====
        self.vol_pub = rospy.Publisher("/solution/current_volume_ml", Float32, queue_size=1, latch=True)

        # ===== 内部状态 =====
        self.total_ml = float(self.V0_ml)
        self.tip_id = 0  # 每轮递增：0,1,2,...
        self.I_mol = 0.0  # 积分项（单位：mol），用于 PI
        self.loop_idx = 0

    # ===== 首轮闭式解（酸侧 × alpha）=====
    def compute_first_dose(self, V0_ml, target_ph, c_HCl):
        if c_HCl <= 0.0:
            rospy.logwarn("[first_dose] c_HCl must be > 0.")
            return ("HCl", 0.0)
        H = 10.0 ** (-float(target_ph))     # mol/L
        denom = float(c_HCl) - H
        if denom <= 1e-12:
            rospy.logwarn("[first_dose] denominator <= 0.")
            return ("HCl", 0.0)
        V_hcl_ml = (H * float(V0_ml)) / denom
        dose_ul  = max(0.0, V_hcl_ml * 1000.0)
        # 酸侧保守
        dose_ul *= self.alpha_acid
        # 夹紧
        dose_ul  = min(self.pipette_max_ul, max(self.min_step_ul, dose_ul))
        rospy.loginfo(f"[first_dose] V0={V0_ml} mL, pH*={target_ph}, C={c_HCl} → {dose_ul:.1f} μL HCl (alpha={self.alpha_acid})")
        return ("HCl", dose_ul)

    # ===== 闭式解（后续轮；含酸侧 alpha 与 ΔpH 封顶 + 近目标禁酸）=====
    def compute_next_dose_closed_form(self, last_ph, target_ph, total_ml, c_HCl):
        V_L   = float(total_ml) / 1000.0
        H_cur = 10.0 ** (-float(last_ph))
        H_tgt = 10.0 ** (-float(target_ph))
        C     = float(c_HCl)
        n     = H_cur * V_L

        # 接近目标：优先禁酸（交给水/下一轮）
        ph_err = target_ph - last_ph  # >0：需要更酸；<0：需要稀释
        if ph_err > 0 and ph_err < self.no_acid_near_eps:
            return ("water", 0.0)

        if H_cur < H_tgt:
            # 酸
            denom = C - H_tgt
            if denom <= 0:
                rospy.logwarn("[CF] target too acidic or C too low.")
                return ("HCl", 0.0)
            v_L = (H_tgt * V_L - n) / denom
            liquid = "HCl"
        else:
            # 水
            v_L = (n / H_tgt) - V_L
            liquid = "water"

        dose_ul = max(0.0, v_L * 1e6)

        # 酸侧：alpha + ΔpH 封顶
        if liquid.lower() == "hcl":
            # alpha
            dose_ul *= self.alpha_acid
            # ΔpH 封顶：dph_dv_acid = (C - H)/ (H V ln10)
            dph_dv_acid = (C - H_cur) / (max(H_cur,1e-12) * max(V_L,1e-12) * np.log(10.0))  # pH per L
            if dph_dv_acid > 1e-12:
                v_cap_L = self.acid_dph_cap / dph_dv_acid
                dose_ul_cap = max(0.0, v_cap_L * 1e6)
                dose_ul = min(dose_ul, dose_ul_cap)

        # 统一夹紧
        dose_ul = min(dose_ul, self.pipette_max_ul, self.safety_cap_ul)
        if dose_ul < self.min_step_ul:
            return (liquid, 0.0)
        return (liquid, float(dose_ul))

    # ===== P 控制（后续轮；酸侧 alpha + ΔpH 封顶 + 近目标禁酸）=====
    def compute_next_dose_P(self, last_ph, target_ph, total_ml, c_HCl):
        V_L   = float(total_ml) / 1000.0
        H_cur = 10.0 ** (-float(last_ph))
        H_tgt = 10.0 ** (-float(target_ph))
        C     = float(c_HCl)

        # 误差与比例摩尔量
        e   = H_tgt - H_cur                  # mol/L
        Kp  = self.kp_base * V_L             # mol
        u   = Kp * e                         # 需要补/去的H+摩尔量（mol）

        ph_err = target_ph - last_ph         # >0 需要更酸；<0 需要稀释
        if ph_err > 0 and ph_err < self.no_acid_near_eps:
            # 目标邻域内，优先禁酸
            return ("water", 0.0)

        if u >= 0.0:
            # 加酸：体积 = u / C
            v_L = u / max(C, 1e-12)
            v_L *= self.alpha_acid
            liquid = "HCl"

            # ΔpH 封顶
            dph_dv_acid = (C - H_cur) / (max(H_cur,1e-12) * max(V_L,1e-12) * np.log(10.0))  # pH per L
            if dph_dv_acid > 1e-12:
                v_cap_L = self.acid_dph_cap / dph_dv_acid
                v_L = min(v_L, v_cap_L)

        else:
            # 加水：把 u（负）转成目标浓度 y_des，再求水体积
            yk    = H_cur
            n     = yk * V_L
            y_des = max(1e-9, yk + u / max(V_L, 1e-12))  # 防负
            v_L   = max(0.0, n / y_des - V_L)
            liquid = "water"

        dose_ul = max(0.0, v_L * 1e6)
        dose_ul = min(dose_ul, self.pipette_max_ul, self.safety_cap_ul)
        if dose_ul < self.min_step_ul:
            return (liquid, 0.0)
        return (liquid, float(dose_ul))

    # ===== 剂量调度 =====
    def plan_dose(self, last_ph, is_first_loop):
        if is_first_loop:
            # 首轮：闭式解（酸侧已乘 alpha）
            return self.compute_first_dose(self.total_ml, self.target_ph, self.c_HCl)

        if self.control_strategy == "closed_form":
            return self.compute_next_dose_closed_form(last_ph, self.target_ph, self.total_ml, self.c_HCl)
        else:
            return self.compute_next_dose_P(last_ph, self.target_ph, self.total_ml, self.c_HCl)


    # ===== 单轮执行 =====
    def run_one_cycle(self, liquid, dose_ul, measure_timeout_s=2.0):
        rospy.loginfo(f"[Cycle {self.loop_idx}] pipette: {liquid}, {dose_ul:.1f} μL, tip_id={self.tip_id}")
        resp_p = self.srv_pipette(liquid=liquid, volume_ul=float(dose_ul), tip_id=int(self.tip_id))
        if not resp_p.success:
            raise RuntimeError("pipette failed: " + resp_p.message)

        rospy.loginfo(f"[Cycle {self.loop_idx}] beaker shake")
        resp_b = self.srv_shake(shake=True)
        if not resp_b.success:
            raise RuntimeError("beaker shake failed: " + resp_b.message)

        rospy.loginfo(f"[Cycle {self.loop_idx}] pH measure")
        resp_m = self.srv_measure(timeout_s=float(measure_timeout_s))
        if not resp_m.success:
            raise RuntimeError("pH measure failed: " + resp_m.message)

        # 更新总体积（uL→mL）
        self.total_ml += float(dose_ul) / 1000.0
        self.vol_pub.publish(Float32(self.total_ml))

        # tip & loop++
        self.tip_id   += 1
        self.loop_idx += 1

        return float(resp_m.ph)


    # ===== 主流程 =====
    def run(self):
        rospy.loginfo(f"Start orchestration: V0={self.V0_ml} mL, pH*={self.target_ph}, C={self.c_HCl} M, strategy={self.control_strategy}")
        self.vol_pub.publish(Float32(self.total_ml))

        # 首轮
        liq, ul = self.plan_dose(last_ph=None, is_first_loop=True)
        last_ph = self.run_one_cycle(liq, ul)

        loops = 1
        while not rospy.is_shutdown() and loops < self.max_loops:
            rospy.loginfo(f"[Loop {loops}] measured pH={last_ph:.3f}, total={self.total_ml:.3f} mL")

            if abs(last_ph - self.target_ph) <= self.ph_tol:
                rospy.loginfo(f"DONE: pH={last_ph:.3f} within ±{self.ph_tol}, total={self.total_ml:.3f} mL")
                break

            liq, ul = self.plan_dose(last_ph=last_ph, is_first_loop=False)
            last_ph = self.run_one_cycle(liq, ul)
            loops += 1

        rospy.loginfo("Orchestration finished.")

        rospy.loginfo(f"Target pH {self.target_ph:.2f} Feedback Control Finished. Final pH value: {last_ph:.2f}")


if __name__ == "__main__":
    try:
        SolutionOrchestrator().run()
    except Exception as e:
        rospy.logerr("Orchestrator error: %s", e)
