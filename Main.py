import numpy as np
import INS_MECH_CLASS
import INS_MECH_FUNCTION as ins
import INS_MECH_CS as mech
import pandas as pd
import matplotlib.pyplot as plt
import struct
import os


def read_file(filename, flag=0):  # flag 1 为，0的话为空
    # filename = 'GNSS_RTK.txt'  # txt文件和当前脚本在同一目录下，所以不用写具体路径
    Efield = []
    with open(filename, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline()  # 整行读取数据
            if not lines:
                break
                pass
            if flag:
                le = len(lines)
                # print(le)
                lines = lines[1:le - 2]
                # print(lines)
                E_tmp = [float(i) for i in lines.split(',')]  # 将整行数据分割处理，如果分割符是空格，括号里就不用传入参数，如果是逗号， 则传入‘，'字符。
            else:
                # print(lines.split())
                E_tmp = [float(i) for i in lines.split()]
            Efield.append(E_tmp)
            pass
    Efield = np.array(Efield)
    pass
    return Efield


def main():
    meas_cur = np.mat([0.0, 0, 0, 0, 0, 0, 0]).T
    meas_prev = np.mat([0.0, 0, 0, 0, 0, 0, 0]).T
    Cbn = ins.euler2dcm(np.pi * 0.854 / 180, np.pi * -2.0345 / 180, np.pi * 185.696 / 180)
    nav = INS_MECH_CLASS.Nav(
        r=np.mat([np.pi * 30.444787369 / 180, np.pi * 114.471863247 / 180, 20.910]).T, C_bn=Cbn,
        v=np.mat([0.0, -0.0, 0.0]).T)

    nav.q_bn = ins.dcm2quat(Cbn)
    nav.q_ne = ins.pos2quat(nav.r[0, 0], nav.r[1, 0])
    par = INS_MECH_CLASS.Par()
    r = []
    v = []

    times = 640000
    RTK = read_file('GNSS_RTK.txt')
    print(RTK.shape)
    Truth = read_file('truth.nav')
    i = 0
    while (True):
        if Truth[i, 1] >= 456300.000:
            break
        i += 1
    Truth = Truth[i:i + times, 1:]
    # print(Truth[0, :])
    for t in range(Truth[:, 9].shape[0]):
        x = Truth[t, 9]
        Truth[t, 9] = x - 360 if x > 180 else x

    print(Truth.shape)

    # binfile = open('A15_imu.bin', 'rb')  # 打开二进制文件
    # size = os.path.getsize('A15_imu.bin')  # 获得文件大小
    # index = 1
    # print(size)
    # # f = open("INS_Ref.txt", "a")  # 利用追加模式,参数从w替换为a即可
    # # f.truncate(0)
    # f2 = open("IMU_Ref.txt", "a")  # 利用追加模式,参数从w替换为a即可
    # f2.truncate(0)
    # # res_imu = np.mat(np.zeros((times, 10)))
    # temp = [456300,
    #         nav.r[0, 0] * 180 / np.pi, nav.r[1, 0] * 180 / np.pi, nav.r[2, 0],
    #         nav.v[0, 0], nav.v[1, 0], nav.v[2, 0],
    #         ins.dcm2euler(nav.C_bn)[0, 0] * 180 / np.pi, ins.dcm2euler(nav.C_bn)[1, 0] * 180 / np.pi,
    #         ins.dcm2euler(nav.C_bn)[2, 0] * 180 / np.pi]
    # # f.write("{}\n".format(temp))
    # # f.close()
    # # res_imu[0, :] = temp
    # for i in range(size):
    #     data = binfile.read(8)  # 每次输出8个字节
    #     num = struct.unpack('d', data)
    #     meas_cur[i % 7, 0] = num[0]
    #     if meas_cur[0, 0] < 456300.0 and meas_cur[0, 0] > 456300.0 - 2:
    #         meas_prev[0, 0] = meas_cur[0, 0]
    #
    #     if (i + 1) % 7 == 0 and meas_cur[0, 0] >= 456300.0:
    #         f2.write("{}\n".format(meas_cur))
    #         nav1 = mech.INS_MECH_CS(meas_prev, meas_cur, nav, par)
    #         r.append([nav1.r[0, 0], nav1.r[1, 0], nav1.r[2, 0]])
    #         v.append([nav1.v[0, 0], nav1.v[1, 0], nav1.v[2, 0]])
    #         # nav = feed.INS_MECH_BACK(meas_prev, meas_cur, nav1, par)
    #         # nav1 = INS_MECH_CS(meas_prev, meas_cur, nav, par)
    #
    #         meas_prev[:, 0] = meas_cur[:, 0]
    #         nav = nav1.copy()
    #         temp = [meas_cur[0, 0],
    #                 nav.r[0, 0] * 180 / np.pi, nav.r[1, 0] * 180 / np.pi, nav.r[2, 0],
    #                 nav.v[0, 0], nav.v[1, 0], nav.v[2, 0],
    #                 ins.dcm2euler(nav.C_bn)[0, 0] * 180 / np.pi, ins.dcm2euler(nav.C_bn)[1, 0] * 180 / np.pi,
    #                 ins.dcm2euler(nav.C_bn)[2, 0] * 180 / np.pi]
    #         # res_imu[index, :] = temp
    #         # f.write("{}\n".format(temp))
    #         if times / (index + 1) == 5:
    #             print('20%')
    #         elif times / (index + 1) == 2:
    #             print('50%')
    #         elif times / (index + 1.0) == 1.25:
    #             print('80%')
    #         index += 1
    #         if index == times:
    #             break
    # print(index)
    # # f.close()
    # f2.close()


    ylabel = ['lat/deg', 'lot/deg', 'altitude/m', 'Vx/m/s', 'Vy/m/s', 'Vz/m/s', 'roll/deg', 'pitch/deg', 'heading/deg']
    title = ['lat_compare', 'lot_compare', 'altitude_compare', 'Vx_compare', 'Vy_compare', 'Vz_compare', 'roll_compare',
             'pitch_compare', 'heading_compare']
    # df = pd.read_csv('imu.csv')
    res_imu = read_file('INS_Ref.txt', flag=1)
    print(res_imu[0, :])
    print(Truth[0, :])
    # df = pd.read_csv('RTK.csv')
    # RTK = np.array(df)
    # print(res_ref.shape)
    print(res_imu.shape)
    for i in range(9):
        fig_i, ax = plt.subplots(figsize=(12, 8))

        ax.plot(Truth[:, 0], Truth[:, i + 1], 'g', label='truth')
        ax.plot(res_imu[:, 0], res_imu[:, i + 1], 'r', label='imu')
        # ax.plot(Truth[0:times-1, 0], res_imu[:, i + 1] - Truth[0:times-1, i + 1], 'y', label='Compare')
        # if i < 3:
        #     ax.plot(RTK[:, 0], RTK[:, i + 1], 'b', label='ref')
        # ax.plot(res_ref[:, 0], res_ref[:, i + 1], 'b', label='ref')
        # ax.plot(res_imu[0:times-2, 0], res_imu[0:times-2, i + 1] - res_ref[1:times, i + 1], 'y', label='Compare')

        ax.legend(loc=2)
        ax.set_xlabel('time/s')
        ax.set_ylabel(ylabel[i])
        ax.set_title(title[i])
    plt.show()


if __name__ == '__main__':
    main()
