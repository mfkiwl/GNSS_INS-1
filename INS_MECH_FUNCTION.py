import numpy as np
import pandas as pd


def CrossProduct(a, b):
    c = np.zeros(3)
    c[0] = a[1, 0] * b[2, 0] - a[2, 0] * b[1, 0]
    c[1] = a[2, 0] * b[0, 0] - a[0, 0] * b[2, 0]
    c[2] = a[0, 0] * b[1, 0] - a[1, 0] * b[0, 0]
    return np.mat(c).T


def GetRM(a, e2, lat):
    return a * (1 - e2) / np.power(1 - e2 * np.sin(lat) * np.sin(lat), 1.5)


def GetRN(a, e2, lat):
    return a / np.power(1 - e2 * np.sin(lat) * np.sin(lat), 0.5)


def GetW_en(r, v, n, m):
    return np.mat([v[1, 0] / (n + r[2, 0]), -v[0, 0] / (m + r[2, 0]), -v[1, 0] * np.tan(r[0, 0]) / (n + r[2, 0])]).T


def GetW_ie(w_e, r):
    return w_e * np.mat([np.cos(r[0, 0]), 0, -np.sin(r[0, 0])]).T


def NormalGravity(latitude, he):
    a1 = 9.7803267715
    a2 = 0.0052790414
    a3 = 0.0000232718
    a4 = -0.000003087691089
    a5 = 0.000000004397731
    a6 = 0.000000000000721
    s2 = np.sin(latitude) * np.sin(latitude)
    s4 = s2 * s2
    return a1 * (1 + a2 * s2 + a3 * s4) + (a4 + a5 * s2) * he + a6 * he * he


def cp_form(v):
    V = np.zeros((3, 3))
    V[0, 1] = -v[2, 0]
    V[0, 2] = v[1, 0]
    V[1, 0] = v[2, 0]
    V[1, 2] = -v[0, 0]
    V[2, 0] = -v[1, 0]
    V[2, 1] = v[0, 0]
    return V


def pos2dcm(lat, lon):  # refrence to 4-32
    s_lat = np.sin(lat)
    c_lat = np.cos(lat)
    s_lon = np.sin(lon)
    c_lon = np.cos(lon)
    C_ne = np.mat(
        [-s_lat * c_lon, -s_lon, -c_lat * c_lon, -s_lat * s_lon, c_lon, -c_lat * s_lon, c_lat, 0.0, -s_lat]).reshape(
        3, 3)
    return C_ne


def dcm2pos(C_ne):
    lat = -np.arctan2(C_ne[2, 2], C_ne[2, 0])
    lon = -np.arctan2(C_ne[0, 1], C_ne[1, 1])
    return [lat, lon]


def rvec2quat(rot_vec):  ##refrence to 3-48
    mag2 = rot_vec[0, 0] * rot_vec[0, 0] + rot_vec[1, 0] * rot_vec[1, 0] + rot_vec[2, 0] * rot_vec[2, 0]
    if mag2 < np.pi * np.pi:
        mag2 = 0.25 * mag2
        c = 1.0 - mag2 / 2.0 * (1.0 - mag2 / 12.0 * (1.0 - mag2 / 30.0))
        s = 1.0 - mag2 / 6.0 * (1.0 - mag2 / 20.0 * (1.0 - mag2 / 42.0))
        q = np.mat([c, s * 0.5 * rot_vec[0, 0], s * 0.5 * rot_vec[1, 0], s * 0.5 * rot_vec[2, 0]]).reshape(4, 1)
    else:
        mag = np.sqrt(mag2)
        s_mag = np.sin(mag / 2)

        q = np.mat(
            [np.cos(mag / 2.0), rot_vec[0, 0] * s_mag / mag, rot_vec[1, 0] * s_mag / mag,
             rot_vec[2, 0] * s_mag / mag]).reshape(4, 1)

        if q[0, 0] < 0:
            q = -q
    return q


def quat2rvec(q):  # refrence to 3-48
    if q[0, 0] == 0:
        rot_vec = np.mat(np.pi * [q[1, 0], q[2, 0], q[3, 0]]).T
        return rot_vec

    if q[0, 0] < 0:
        q = -q
    mag2 = np.arctan(np.sqrt(q[1, 0] * q[1, 0] + q[2, 0] * q[2, 0] + q[3, 0] * q[3, 0]) / q[0, 0])
    f = np.sin(mag2) / mag2 / 2
    rot_vec = q[1:4, 0] / f
    # mag2 = (q[1, 0] * q[1, 0] + q[2, 0] * q[2, 0] + q[3, 0] * q[3, 0]) / (q[0, 0]*q[0, 0])
    # f = 1 - mag2 / 6.0 * (1 - mag2 / 20.0 * (1 - mag2 / 42))
    # f = 0.5 * f
    # rot_vec = q[1:4, 0] / f
    return np.mat(rot_vec).T


def dcm2quat(C):  # refrence to 3-47
    Tr = np.trace(C)
    q = np.zeros((4, 1))
    p = np.zeros(4)
    p[0] = 1 + Tr
    p[1] = 1 + 2 * C[0, 0] - Tr
    p[2] = 1 + 2 * C[1, 1] - Tr
    p[3] = 1 + 2 * C[2, 2] - Tr
    index = np.argmax(p)
    if index == 0:
        q[0, 0] = 0.5 * np.sqrt(p[0])
        q[1, 0] = (C[2, 1] - C[1, 2]) / q[0, 0] / 4
        q[2, 0] = (C[0, 2] - C[2, 0]) / q[0, 0] / 4
        q[3, 0] = (C[1, 0] - C[0, 1]) / q[0, 0] / 4
    elif index == 1:
        q[1, 0] = 0.5 * np.sqrt(p[1])
        q[0, 0] = (C[2, 1] - C[1, 2]) / q[1, 0] / 4
        q[2, 0] = (C[1, 0] + C[0, 1]) / q[1, 0] / 4
        q[3, 0] = (C[0, 2] + C[2, 0]) / q[1, 0] / 4
    elif index == 2:
        q[2, 0] = 0.5 * np.sqrt(p[2])
        q[0, 0] = (C[0, 2] - C[2, 0]) / q[2, 0] / 4
        q[1, 0] = (C[1, 0] + C[0, 1]) / q[2, 0] / 4
        q[3, 0] = (C[2, 1] + C[1, 2]) / q[2, 0] / 4
    else:
        q[3, 0] = 0.5 * np.sqrt(p[3])
        q[0, 0] = (C[1, 0] - C[0, 1]) / q[3, 0] / 4
        q[1, 0] = (C[0, 2] + C[2, 0]) / q[3, 0] / 4
        q[2, 0] = (C[2, 1] + C[1, 2]) / q[3, 0] / 4
    if q[0, 0] < 0:
        q = -q
    return q


def quat2dcm(q):  # refrence to 3-46
    C_ne = np.zeros((3, 3))
    C_ne[0, 0] = q[0, 0] * q[0, 0] + q[1, 0] * q[1, 0] - q[2, 0] * q[2, 0] - q[3, 0] * q[3, 0]
    C_ne[0, 1] = 2 * (q[1, 0] * q[2, 0] - q[0, 0] * q[3, 0])
    C_ne[0, 2] = 2 * (q[1, 0] * q[3, 0] + q[0, 0] * q[2, 0])
    C_ne[1, 0] = 2 * (q[1, 0] * q[2, 0] + q[0, 0] * q[3, 0])
    C_ne[1, 1] = q[0, 0] * q[0, 0] - q[1, 0] * q[1, 0] + q[2, 0] * q[2, 0] - q[3, 0] * q[3, 0]
    C_ne[1, 2] = 2 * (q[2, 0] * q[3, 0] - q[0, 0] * q[1, 0])
    C_ne[2, 0] = 2 * (q[1, 0] * q[3, 0] - q[0, 0] * q[2, 0])
    C_ne[2, 1] = 2 * (q[2, 0] * q[3, 0] + q[0, 0] * q[1, 0])
    C_ne[2, 2] = q[0, 0] * q[0, 0] - q[1, 0] * q[1, 0] - q[2, 0] * q[2, 0] + q[3, 0] * q[3, 0]
    return C_ne


def quat2pos(q_ne):
    lat = -2 * np.arctan(q_ne[2, 0] / q_ne[0, 0]) - np.pi / 2
    lon = 2 * np.arctan2(q_ne[3, 0], q_ne[0, 0])
    return lat, lon


def pos2quat(lat, lon):
    s1 = np.sin(lon / 2)
    c1 = np.cos(lon / 2)
    s2 = np.sin(-np.pi / 4 - lat / 2)
    c2 = np.cos(-np.pi / 4 - lat / 2)
    q_ne = np.mat([c1 * c2, -s1 * s2, c1 * s2, c2 * s1]).T
    return q_ne


def qmul(q1, q2):
    q = np.mat([q1[0, 0] * q2[0, 0] - q1[1, 0] * q2[1, 0] - q1[2, 0] * q2[2, 0] - q1[3, 0] * q2[3, 0],
                q1[0, 0] * q2[1, 0] + q1[1, 0] * q2[0, 0] + q1[2, 0] * q2[3, 0] - q1[3, 0] * q2[2, 0],
                q1[0, 0] * q2[2, 0] + q1[2, 0] * q2[0, 0] + q1[3, 0] * q2[1, 0] - q1[1, 0] * q2[3, 0],
                q1[0, 0] * q2[3, 0] + q1[3, 0] * q2[0, 0] + q1[1, 0] * q2[2, 0] - q1[2, 0] * q2[1, 0]]).T
    if q[0, 0] < 0:
        q = -q
    return q


def norm_quat(q):  # refrence to 7-28
    e = (q.T * q - 1) / 2
    e = e[0, 0]
    q_n = (1 - e) * q
    return q_n


def dist_ang(ang1, ang2):
    ang = ang2 - ang1
    if ang > np.pi:
        ang = ang - 2 * np.pi
    elif ang < -np.pi:
        ang = ang + 2 * np.pi
    return ang


def dpos2rvec(lat, delta_lat, delta_lon):
    return np.mat([delta_lon * np.cos(lat), -delta_lat, -delta_lon * np.sin(lat)]).T


def euler2dcm(roll, pitch, heading):
    C_bn = np.zeros((3, 3))
    cr = np.cos(roll)
    cp = np.cos(pitch)
    ch = np.cos(heading)
    sr = np.sin(roll)
    sp = np.sin(pitch)
    sh = np.sin(heading)

    C_bn[0, 0] = cp * ch
    C_bn[0, 1] = -cr * sh + sr * sp * ch
    C_bn[0, 2] = sr * sh + cr * sp * ch

    C_bn[1, 0] = cp * sh
    C_bn[1, 1] = cr * ch + sr * sp * sh
    C_bn[1, 2] = -sr * ch + cr * sp * sh

    C_bn[2, 0] = - sp
    C_bn[2, 1] = sr * cp
    C_bn[2, 2] = cr * cp
    return C_bn


def dcm2euler(C_bn):  # refrence to 3-34
    roll = 0
    pitch = np.arctan(-C_bn[2, 0] / (np.sqrt(C_bn[2, 1] * C_bn[2, 1] + C_bn[2, 2] * C_bn[2, 2])))
    heading = 0

    if C_bn[2, 0] <= -0.999:
        roll = np.NaN
        heading = np.arctan2((C_bn[1, 2] - C_bn[0, 1]), (C_bn[0, 2] + C_bn[1, 1]))
    elif C_bn[2, 0] >= 0.999:
        roll = np.NaN
        heading = np.pi + np.arctan2((C_bn[1, 2] + C_bn[0, 1]), (C_bn[0, 2] - C_bn[1, 1]))
    else:
        roll = np.arctan2(C_bn[2, 1], C_bn[2, 2])
        heading = np.arctan2(C_bn[1, 0], C_bn[0, 0])
    # roll = np.arctan(C_bn[2, 1] / C_bn[2, 2])
    # pitch = np.arctan(-C_bn[2, 0] / (np.sqrt(C_bn[2, 1] * C_bn[2, 1] + C_bn[2, 2] * C_bn[2, 2])))
    # heading = np.arctan(C_bn[1, 0] / C_bn[0, 0])

    return np.mat([roll, pitch, heading]).T


def qconj(qi):
    q = np.copy(qi)
    q[1:4, 0] = -q[1:4, 0]
    return q


def deg2rad(a):
    return a / 180 * np.pi


def rad2deg(r):
    return r / np.pi * 180




