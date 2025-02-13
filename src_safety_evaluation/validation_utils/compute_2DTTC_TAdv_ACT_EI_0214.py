import numpy as np


# 计算Guo2023版2DTTC的辅助函数1
def compute_TTC_lon_Guo2023(x_A, y_A, v_A, h_A, l_A, w_A, x_B, y_B, v_B, h_B, l_B, w_B):
    v_x_A = v_A * np.cos(h_A)
    v_y_A = v_A * np.sin(h_A)
    v_x_B = v_B * np.cos(h_B)
    v_y_B = v_B * np.sin(h_B)
    delta_x = x_B - x_A
    delta_y = y_B - y_A
    delta_v_x = v_x_B - v_x_A
    delta_v_y = v_y_B - v_y_A
    if delta_x * delta_v_x < 0:
        TTC_lon = (abs(delta_x) - (l_A + l_B) / 2) / abs(delta_v_x)
        S_lat_TTC_lon = delta_y + delta_v_y * TTC_lon
        if abs(S_lat_TTC_lon) < 1.0 * (w_A + w_B) / 2 and TTC_lon >= 0:
            return TTC_lon
    return np.nan

# 计算Guo2023版2DTTC的辅助函数2
def compute_TTC_lat_Guo2023(x_A, y_A, v_A, h_A, l_A, w_A, x_B, y_B, v_B, h_B, l_B, w_B):
    v_x_A = v_A * np.cos(h_A)
    v_y_A = v_A * np.sin(h_A)
    v_x_B = v_B * np.cos(h_B)
    v_y_B = v_B * np.sin(h_B)
    delta_x = x_B - x_A
    delta_y = y_B - y_A
    delta_v_x = v_x_B - v_x_A
    delta_v_y = v_y_B - v_y_A
    if delta_y * delta_v_y < 0:
        TTC_lat = (abs(delta_y) - (w_A + w_B) / 2) / abs(delta_v_y)
        S_lon_TTC_lat = delta_x + delta_v_x * TTC_lat
        if abs(S_lon_TTC_lat) < 1.0 * (l_A + l_B) / 2 and TTC_lat >= 0:
            return TTC_lat
    return np.nan


# 计算TAdv的辅助函数
def compute_TAdv(x_A, y_A, v_A, h_A, l_A, x_B, y_B, v_B, h_B, l_B):
    # 计算角度差
    # angle_difference = abs(h_A - h_B)
    # if angle_difference > np.pi:
    #     angle_difference = 2 * np.pi - angle_difference

    angle_difference = abs(((h_A - h_B) + np.pi) % (2 * np.pi) - np.pi)

    delta_x = x_B - x_A
    delta_y = y_B - y_A
    norm_delta = np.sqrt(delta_x ** 2 + delta_y ** 2)

    def compute_time_to_collision_point(v_A, v_B, delta_x, delta_y, h_A, h_B):
        denominator_ac = v_A * np.sin(h_B - h_A)
        denominator_bc = v_B * np.sin(h_A - h_B)

        # 处理分母为零的情况
        t_ac = np.nan if abs(denominator_ac) < EPSILON \
            else (delta_x * np.sin(h_B) - delta_y * np.cos(h_B)) / denominator_ac
        t_bc = np.nan if abs(denominator_bc) < EPSILON \
            else ((-delta_x) * np.sin(h_A) - (-delta_y) * np.cos(h_A)) / denominator_bc

        return t_ac, t_bc

    # 如果角度差小于 GAMMA，则计算 TAdv
    if 0 <= angle_difference <= GAMMA:
        if np.dot([delta_x, delta_y], [np.cos(h_A), np.sin(h_A)]) > 0:
            TAdv = (norm_delta - l_B / 2 - l_A / 2) / v_A
        else:
            TAdv = (norm_delta - l_B / 2 - l_A / 2) / v_B
    else:
        # 计算 t_ac 和 t_bc
        t_ac, t_bc = compute_time_to_collision_point(v_A, v_B, delta_x, delta_y, h_A, h_B)
        TAdv = abs(t_ac - t_bc)

    return TAdv if TAdv >= 0 else np.nan



# 计算ACT和EI的辅助函数1
def compute_v_Br(x_A, y_A, v_A, h_A, x_B, y_B, v_B, h_B):
    delta_x = x_B - x_A
    delta_y = y_B - y_A
    norm_delta = np.sqrt(delta_x ** 2 + delta_y ** 2)
    if norm_delta != 0:
        unit_vector = np.array([delta_x / norm_delta, delta_y / norm_delta])
        velocity_diff = np.array([v_B * np.cos(h_B) - v_A * np.cos(h_A), v_B * np.sin(h_B) - v_A * np.sin(h_A)])
        v_Br = -np.dot(unit_vector, velocity_diff)
    else:
        v_Br = 0
    return v_Br

# 计算ACT和EI的辅助函数2
def compute_TDM_InDepth(x_A, y_A, v_A, h_A, l_A, w_A, x_B, y_B, v_B, h_B, l_B, w_B):

    v_diff = np.array([v_B * np.cos(h_B) - v_A * np.cos(h_A), v_B * np.sin(h_B) - v_A * np.sin(h_A)])
    theta_B_prime = v_diff / np.linalg.norm(v_diff)
    delta = np.array([x_B - x_A, y_B - y_A])
    D_t1 = np.linalg.norm(delta - np.dot(delta, theta_B_prime) * theta_B_prime)
    AB = np.array([x_B - x_A, y_B - y_A])

    AA1 = np.array([l_A / 2 * np.cos(h_A) - w_A / 2 * -np.sin(h_A), l_A / 2 * np.sin(h_A) - w_A / 2 * np.cos(h_A)])
    AA2 = np.array([l_A / 2 * np.cos(h_A) + w_A / 2 * -np.sin(h_A), l_A / 2 * np.sin(h_A) + w_A / 2 * np.cos(h_A)])
    AA3 = np.array([-l_A / 2 * np.cos(h_A) - w_A / 2 * -np.sin(h_A), -l_A / 2 * np.sin(h_A) - w_A / 2 * np.cos(h_A)])
    AA4 = np.array([-l_A / 2 * np.cos(h_A) + w_A / 2 * -np.sin(h_A), -l_A / 2 * np.sin(h_A) + w_A / 2 * np.cos(h_A)])
    d_A1 = np.linalg.norm(AA1 - np.dot(AA1, theta_B_prime) * theta_B_prime)
    d_A2 = np.linalg.norm(AA2 - np.dot(AA2, theta_B_prime) * theta_B_prime)
    d_A3 = np.linalg.norm(AA3 - np.dot(AA3, theta_B_prime) * theta_B_prime)
    d_A4 = np.linalg.norm(AA4 - np.dot(AA4, theta_B_prime) * theta_B_prime)
    d_As = np.array([d_A1, d_A2, d_A3, d_A4])
    d_A_max = np.max(d_As)

    BB1 = np.array([l_B / 2 * np.cos(h_B) - w_B / 2 * -np.sin(h_B), l_B / 2 * np.sin(h_B) - w_B / 2 * np.cos(h_B)])
    BB2 = np.array([l_B / 2 * np.cos(h_B) + w_B / 2 * -np.sin(h_B), l_B / 2 * np.sin(h_B) + w_B / 2 * np.cos(h_B)])
    BB3 = np.array([-l_B / 2 * np.cos(h_B) - w_B / 2 * -np.sin(h_B), -l_B / 2 * np.sin(h_B) - w_B / 2 * np.cos(h_B)])
    BB4 = np.array([-l_B / 2 * np.cos(h_B) + w_B / 2 * -np.sin(h_B), -l_B / 2 * np.sin(h_B) + w_B / 2 * np.cos(h_B)])
    d_B1 = np.linalg.norm(BB1 - np.dot(BB1, theta_B_prime) * theta_B_prime)
    d_B2 = np.linalg.norm(BB2 - np.dot(BB2, theta_B_prime) * theta_B_prime)
    d_B3 = np.linalg.norm(BB3 - np.dot(BB3, theta_B_prime) * theta_B_prime)
    d_B4 = np.linalg.norm(BB4 - np.dot(BB4, theta_B_prime) * theta_B_prime)
    d_Bs = np.array([d_B1, d_B2, d_B3, d_B4])
    d_B_max = np.max(d_Bs)

    MFD = D_t1 - (d_A_max + d_B_max)
    D_B_prime = -np.dot(delta, theta_B_prime)
    v_Br_norm = np.linalg.norm(v_diff)
    TDM = D_B_prime / v_Br_norm if v_Br_norm != 0 else None
    InDepth = D_SAFE - MFD

    # 计算ACT所需的角点间最短距离
    # 计算16个向量并找出模长最短的向量
    vectors = [
        BB1 + AB - AA1, BB2 + AB - AA1, BB3 + AB - AA1, BB4 + AB - AA1,
        BB1 + AB - AA2, BB2 + AB - AA2, BB3 + AB - AA2, BB4 + AB - AA2,
        BB1 + AB - AA3, BB2 + AB - AA3, BB3 + AB - AA3, BB4 + AB - AA3,
        BB1 + AB - AA4, BB2 + AB - AA4, BB3 + AB - AA4, BB4 + AB - AA4
    ]
    # 计算每个向量的模长
    norms = [np.linalg.norm(vec) for vec in vectors]
    # 找到最小的模长
    dis_shortest = min(norms)

    return TDM, InDepth, dis_shortest


# 计算2DTTC、TAdv、ACT、EI数值
def compute_real_time_metrics(x_A, y_A, v_A, h_A, l_A, w_A, x_B, y_B, v_B, h_B, l_B, w_B):

    # 计算Guo2023版2DTTC
    ttc_lon_Guo2023 = compute_TTC_lon_Guo2023(x_A, y_A, v_A, h_A, l_A, w_A, x_B, y_B, v_B, h_B, l_B, w_B)
    ttc_lat_Guo2023 = compute_TTC_lat_Guo2023(x_A, y_A, v_A, h_A, l_A, w_A, x_B, y_B, v_B, h_B, l_B, w_B)
    ttc_lon_Guo2023 = ttc_lon_Guo2023
    ttc_lat_Guo2023 = ttc_lat_Guo2023
    if np.isnan(ttc_lon_Guo2023) and np.isnan(ttc_lat_Guo2023):
        TTC2D_Guo2023 = np.nan
    else:
        TTC2D_Guo2023 = round(np.nanmin([ttc_lon_Guo2023, ttc_lat_Guo2023]), 4)

    # 计算 TAdv
    TAdv = compute_TAdv(x_A, y_A, v_A, h_A, l_A, x_B, y_B, v_B, h_B, l_B)
    TAdv = round(TAdv, 4) if not np.isnan(TAdv) else np.nan

    # 计算 ACT 和 EI
    v_Br = compute_v_Br(x_A, y_A, v_A, h_A, x_B, y_B, v_B, h_B)
    if v_Br > 0:
        TDM, InDepth, dis_shortest = compute_TDM_InDepth(x_A, y_A, v_A, h_A, l_A, w_A, x_B, y_B, v_B, h_B, l_B, w_B)
        TDM = np.nan if TDM is None or TDM < 0 else TDM
        InDepth = np.nan if InDepth is None else InDepth
        dis_shortest = np.nan if dis_shortest is None else dis_shortest
        if InDepth >= 0:
            ACT = np.nan if (dis_shortest / v_Br) < 0 else round(dis_shortest / v_Br, 4)
            EI = round(InDepth / TDM, 4) if not np.isnan(TDM) and TDM != 0 else np.nan
        else:
            EI = ACT = np.nan
    else:
        InDepth = EI = ACT = np.nan

    return TTC2D_Guo2023, TAdv, ACT, InDepth, EI


# 定义常数
K_2DTTC = 1.0  # 2DTTC判定条件使用
D_SAFE = 0  # EI的安全区域参数，暂时默认为0（不考虑安全冗余）
GAMMA = np.pi / 10  # 如果两车航向角之差小于这个值，则不计算TAdv，因为交点太远了
EPSILON = 1e-6  # 一个小量，避免除零错误，TAdv使用

# 主函数
def main():

    # 自车A参数实时参数传入（示例）
    x_A = 0  # 绝对坐标，单位是m
    y_A = 0
    v_A = 5  # 单位是m/s
    h_A = 0  # 航向角，弧度制，范围是[-pi, pi]，例如1.57是90°
    l_A = 4.8  # 车长
    w_A = 1.8  # 车宽

    # 周车B参数实时参数传入（示例）
    x_B = 0
    y_B = 5
    v_B = 7
    h_B = -0.5
    l_B = 4.8
    w_B = 1.8


    # 计算2DTTC、TAdv、ACT、EI数值
    TTC2D_Guo2023, TAdv, ACT, InDepth, EI = compute_real_time_metrics(x_A, y_A, v_A, h_A, l_A, w_A, x_B, y_B, v_B, h_B, l_B, w_B)

    # 输出结果
    print(f"2DTTC_Guo2023: {TTC2D_Guo2023} s")
    print(f"TAdv: {TAdv} s")
    print(f"ACT: {ACT} s")
    print(f"InDepth: {round(InDepth, 4)} m")
    print(f"EI: {EI} m/s")

if __name__ == "__main__":
    main()
