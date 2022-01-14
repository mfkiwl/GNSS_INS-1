import numpy as np
import pandas as pd
import INS_MECH_FUNCTION as ins
import os
import struct
import matplotlib.pyplot as plt

import INS_MECH_CLASS


# meas_prev = previous   [t gyro_x gyro_y gyro_z accel_x accel_y accel_z]
# meas_cur = current measurements, units [s, rad, m/s]
# EM = Earth model
# nav.r = position [lat; lon; h]
# nav.v = velocity [Vn; Ve; Vd]
# nav.dv_n = velocity increment in navigation frame
# nav.q_bn = quaternion from the b-frame to the n-frame;
# nav.q_ne = quaternion from the n-frame to the e-frame;
# par previous k-2 par1 k-1
# par.Rm = radius of curvature in meridian;
# par.Rn = radius of curvature in prime vertical;
# par.w_ie
# par.w_en
# par.g
# par.f_n = specific force
#  - OUTPUT -
# nav1 = updated navigation solution
# dv_n1 = current velocity increment

# Cbn = ins.euler2dcm(np.pi * 0.854 / 180, np.pi * -2.0345 / 180, np.pi * 185.696 / 180)


# print(Cbn)


def INS_MECH_CS(meas_pre, meas_cur, nav, par):
    nav1 = INS_MECH_CLASS.Nav()
    EM = INS_MECH_CLASS.Earth_Model()
    nav.t = meas_pre[0, 0]
    nav1.t = meas_cur[0, 0]
    par.Rm = ins.GetRM(EM.a, EM.e2, nav.r[0, 0])
    par.Rn = ins.GetRN(EM.a, EM.e2, nav.r[0, 0])

    dt = meas_cur[0, 0] - meas_pre[0, 0]

    beta = ins.CrossProduct(meas_pre[1:4, 0], meas_cur[1:4, 0]) / 12.0
    scul = ins.CrossProduct(meas_cur[1:4, 0], meas_cur[4:7, 0]) / 2.0 + \
           ins.CrossProduct(meas_pre[1:4, 0], meas_cur[4:7, 0]) / 12 + \
           ins.CrossProduct(meas_pre[4:7, 0], meas_cur[1:4, 0]) / 12

    # 1、速度更新
    # （1）高度和经纬度的预测

    # mid_r = np.zeros((3, 1))
    # mid_r[2, 0] = nav.r[2, 0] - 0.5 * mid_v[2, 0] * dt
    # par.Rm = ins.GetRM(EM.a, EM.e2, nav.r[0, 0])
    # mid_r[0, 0] = nav.r[0, 0] + 0.5 * mid_v[0, 0] * dt / (par.Rm + mid_r[2, 0])
    # par.Rn = ins.GetRN(EM.a, EM.e2, mid_r[0, 0])
    # mid_r[1, 0] = nav.r[1, 0] + 0.5 * mid_v[1, 0] * dt / ((par.Rn + mid_r[2, 0]) * np.cos(mid_r[0, 0]))

    mid_r = nav.r.copy()
    mid_r[2, 0] = nav.r[2, 0] - 0.5 * nav.v[2, 0] * dt
    d_lat = 0.5 * nav.v[0, 0] * dt / (par.Rm + mid_r[2, 0])
    d_lon = 0.5 * nav.v[1, 0] * dt / (par.Rn + mid_r[2, 0]) / np.cos(mid_r[0, 0])
    d_theta = ins.dpos2rvec(nav.r[0, 0], d_lat, d_lon)
    mid_q = ins.qmul(nav.q_ne, ins.rvec2quat(d_theta))
    mid_r[0, 0], mid_r[1, 0] = ins.quat2pos(mid_q)

    par.g = np.mat([0, 0, ins.NormalGravity(mid_r[0, 0], mid_r[2, 0])]).T
    mid_v = nav.v + nav.dv_n * 0.5  # 用上一时刻的速度增量和速度，求出当前k和k-1的中间时刻的速度
    # (2)
    par.w_ie = ins.GetW_ie(EM.w_e, mid_r)
    par.w_en = ins.GetW_en(mid_r, mid_v, par.Rn, par.Rm)

    # （3）

    zeta = (par.w_ie + par.w_en) * dt

    Cn = np.eye(3) - ins.cp_form(zeta) / 2
    dv_f_n = (Cn.dot(nav.C_bn)).dot(meas_cur[4:7, 0] + scul)
    par.f_n = dv_f_n / dt

    dv_g_cor = (par.g - ins.CrossProduct(2 * par.w_ie + par.w_en, mid_v)) * dt
    nav1.dv_n = dv_f_n + dv_g_cor
    nav1.v = nav.v + nav1.dv_n
    # 2、位置更新
    mid_v = 0.5 * (nav1.v + nav.v)

    par.w_en = ins.GetW_en(mid_r, mid_v, par.Rn, par.Rm)
    zeta = (par.w_en + par.w_ie) * dt
    qn = ins.rvec2quat(zeta)  # q_nk_nk-1
    xi = np.mat([0, 0, -EM.w_e * dt]).T
    qe = ins.rvec2quat(xi)
    nav1.q_ne = ins.qmul(qe, ins.qmul(nav.q_ne, qn))
    nav1.q_ne = ins.norm_quat(nav1.q_ne)
    [nav1.r[0, 0], nav1.r[1, 0]] = ins.quat2pos(nav1.q_ne)
    nav1.r[2, 0] = nav.r[2, 0] - mid_v[2, 0] * dt

    # 3、姿态更新
    # (1)
    q = ins.rvec2quat(meas_cur[1:4, 0] + beta)  # coning
    nav1.q_bn = ins.qmul(nav.q_bn, q)

    mid_r = np.mat([nav1.r[0, 0] + 0.5 * ins.dist_ang(nav1.r[0, 0], nav.r[0, 0]),
                    nav1.r[1, 0] + 0.5 * ins.dist_ang(nav1.r[1, 0], nav.r[1, 0]),
                    0.5 * (nav1.r[2, 0] + nav.r[2, 0])]).T

    par.w_ie = ins.GetW_ie(EM.w_e, mid_r)
    par.w_en = ins.GetW_en(mid_r, mid_v, par.Rn, par.Rm)
    zeta = (par.w_en + par.w_ie) * dt
    # （2）
    q = ins.rvec2quat(-zeta)  # q_nk-1_nk
    nav1.q_bn = ins.qmul(q, nav1.q_bn)
    nav1.q_bn = ins.norm_quat(nav1.q_bn)
    nav1.C_bn = ins.quat2dcm(nav1.q_bn)
    return nav1
