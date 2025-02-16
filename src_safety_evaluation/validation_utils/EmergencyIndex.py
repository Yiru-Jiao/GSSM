'''
This script reuses and adapts EmergencyIndex (https://github.com/AutoChengh/EmergencyIndex)
The commented code is from the original script but not used in this script.
Adaption is right before or after the commented code.
'''

import pandas as pd
import numpy as np
# import argparse
from tqdm import tqdm # for progress bar


def compute_Pc(x_A, y_A, x_B, y_B, h_A, h_B):
    delta_x = x_B - x_A
    delta_y = y_B - y_A
    sin_diff = np.sin(h_B - h_A)
    Pc = np.array([x_A, y_A]) + (delta_x * np.sin(h_B) - delta_y * np.cos(h_B)) / sin_diff * np.array([np.cos(h_A), np.sin(h_A)])
    return Pc

def compute_t_ac(x_A, y_A, x_B, y_B, h_A, h_B, v_A):
    delta_x = x_B - x_A
    delta_y = y_B - y_A
    sin_diff = np.sin(h_B - h_A)
    t_ac = (delta_x * np.sin(h_B) - delta_y * np.cos(h_B)) /(v_A * sin_diff)
    return t_ac

def compute_t_bc(x_A, y_A, x_B, y_B, h_A, h_B, v_B):
    delta_x = x_A - x_B
    delta_y = y_A - y_B
    sin_diff = np.sin(h_A - h_B)
    t_bc = (delta_x * np.sin(h_A) - delta_y * np.cos(h_A)) / (v_B * sin_diff)
    return t_bc

def compute_dac(w_A, h_A, h_B):
    numerator = w_A
    denominator = np.sqrt(1 - np.dot([np.cos(h_A), np.sin(h_A)], [np.cos(h_B), np.sin(h_B)])**2)
    return numerator / denominator

def compute_dbc(w_B, h_A, h_B):
    numerator = w_B
    denominator = np.sqrt(1 - np.dot([np.cos(h_A), np.sin(h_A)], [np.cos(h_B), np.sin(h_B)])**2)
    return numerator / denominator

def compute_A_bcC_i(Pc, dac, dbc, h_A, h_B, x_A, y_A, l_A, i):
    C = np.array([np.cos(h_A), np.sin(h_A)])
    if i == 1:
        return Pc - dbc / 2 * C + dac / 2 * np.array([np.cos(h_B), np.sin(h_B)]) - [x_A - l_A/2 * np.cos(h_A), y_A - l_A/2 * np.sin(h_A)]
    elif i == 2:
        return Pc + dbc / 2 * C + dac / 2 * np.array([np.cos(h_B), np.sin(h_B)]) - [x_A - l_A/2 * np.cos(h_A), y_A - l_A/2 * np.sin(h_A)]
    elif i == 3:
        return Pc - dbc / 2 * C - dac / 2 * np.array([np.cos(h_B), np.sin(h_B)]) - [x_A - l_A/2 * np.cos(h_A), y_A - l_A/2 * np.sin(h_A)]
    elif i == 4:
        return Pc + dbc / 2 * C - dac / 2 * np.array([np.cos(h_B), np.sin(h_B)]) - [x_A - l_A/2 * np.cos(h_A), y_A - l_A/2 * np.sin(h_A)]

def compute_B_bcC_j(Pc, dac, dbc, h_A, h_B, x_B, y_B, l_B, j):
    C = np.array([np.cos(h_B), np.sin(h_B)])
    if j == 1:
        return Pc - dbc / 2 * np.array([np.cos(h_A), np.sin(h_A)]) + dac / 2 * C - [x_B - l_B/2 * np.cos(h_B), y_B - l_B/2 * np.sin(h_B)]
    elif j == 2:
        return Pc + dbc / 2 * np.array([np.cos(h_A), np.sin(h_A)]) + dac / 2 * C - [x_B - l_B/2 * np.cos(h_B), y_B - l_B/2 * np.sin(h_B)]
    elif j == 3:
        return Pc - dbc / 2 * np.array([np.cos(h_A), np.sin(h_A)]) - dac / 2 * C - [x_B - l_B/2 * np.cos(h_B), y_B - l_B/2 * np.sin(h_B)]
    elif j == 4:
        return Pc + dbc / 2 * np.array([np.cos(h_A), np.sin(h_A)]) - dac / 2 * C - [x_B - l_B/2 * np.cos(h_B), y_B - l_B/2 * np.sin(h_B)]

def compute_theta_B_prime(v_A, h_A, v_B, h_B):
    velocity_diff = np.array([v_B * np.cos(h_B) - v_A * np.cos(h_A), v_B * np.sin(h_B) - v_A * np.sin(h_A)])
    theta_B_prime = velocity_diff / np.linalg.norm(velocity_diff)
    return theta_B_prime

def compute_D_t1(x_A, y_A, x_B, y_B, theta_B_prime):
    delta = np.array([x_B - x_A, y_B - y_A])
    D_t1 = np.linalg.norm(delta - np.dot(delta, theta_B_prime) * theta_B_prime)
    return D_t1

def compute_D_B_prime(x_A, y_A, x_B, y_B, theta_B_prime):
    delta = np.array([x_B - x_A, y_B - y_A])
    D_B_prime = -np.dot(delta, theta_B_prime)
    return D_B_prime

def compute_v_B_prime_norm(v_A, h_A, v_B, h_B):
    v_B_prime = np.array([v_B * np.cos(h_B) - v_A * np.cos(h_A), v_B * np.sin(h_B) - v_A * np.sin(h_A)])
    return np.linalg.norm(v_B_prime)

def compute_TDM(x_A, y_A, x_B, y_B, theta_B_prime, v_A, h_A, v_B, h_B):
    D_B_prime = compute_D_B_prime(x_A, y_A, x_B, y_B, theta_B_prime)
    v_B_prime_norm = compute_v_B_prime_norm(v_A, h_A, v_B, h_B)
    TDM = D_B_prime / v_B_prime_norm
    return TDM, D_B_prime, v_B_prime_norm

def compute_AA_l_A_w_A(x_A, y_A, h_A, l_A, w_A):
    cos_h_A, sin_h_A = np.cos(h_A), np.sin(h_A)
    AA1 = np.array([l_A / 2 * cos_h_A - w_A / 2 * -sin_h_A, l_A / 2 * sin_h_A - w_A / 2 * cos_h_A])
    AA2 = np.array([l_A / 2 * cos_h_A + w_A / 2 * -sin_h_A, l_A / 2 * sin_h_A + w_A / 2 * cos_h_A])
    AA3 = np.array([-l_A / 2 * cos_h_A - w_A / 2 * -sin_h_A, -l_A / 2 * sin_h_A - w_A / 2 * cos_h_A])
    AA4 = np.array([-l_A / 2 * cos_h_A + w_A / 2 * -sin_h_A, -l_A / 2 * sin_h_A + w_A / 2 * cos_h_A])
    return AA1, AA2, AA3, AA4

def compute_BB_l_B_w_B(x_B, y_B, h_B, l_B, w_B):
    cos_h_B, sin_h_B = np.cos(h_B), np.sin(h_B)
    BB1 = np.array([l_B / 2 * cos_h_B - w_B / 2 * -sin_h_B, l_B / 2 * sin_h_B - w_B / 2 * cos_h_B])
    BB2 = np.array([l_B / 2 * cos_h_B + w_B / 2 * -sin_h_B, l_B / 2 * sin_h_B + w_B / 2 * cos_h_B])
    BB3 = np.array([-l_B / 2 * cos_h_B - w_B / 2 * -sin_h_B, -l_B / 2 * sin_h_B - w_B / 2 * cos_h_B])
    BB4 = np.array([-l_B / 2 * cos_h_B + w_B / 2 * -sin_h_B, -l_B / 2 * sin_h_B + w_B / 2 * cos_h_B])
    return BB1, BB2, BB3, BB4

def compute_d_A(x_A, y_A, h_A, l_A, w_A, theta_B_prime):
    AA1, AA2, AA3, AA4 = compute_AA_l_A_w_A(x_A, y_A, h_A, l_A, w_A)
    d_A1 = np.linalg.norm(AA1 - np.dot(AA1, theta_B_prime) * theta_B_prime)
    d_A2 = np.linalg.norm(AA2 - np.dot(AA2, theta_B_prime) * theta_B_prime)
    d_A3 = np.linalg.norm(AA3 - np.dot(AA3, theta_B_prime) * theta_B_prime)
    d_A4 = np.linalg.norm(AA4 - np.dot(AA4, theta_B_prime) * theta_B_prime)
    return max(d_A1, d_A2, d_A3, d_A4)

def compute_d_B(x_B, y_B, h_B, l_B, w_B, theta_B_prime):
    BB1, BB2, BB3, BB4 = compute_BB_l_B_w_B(x_B, y_B, h_B, l_B, w_B)
    d_B1 = np.linalg.norm(BB1 - np.dot(BB1, theta_B_prime) * theta_B_prime)
    d_B2 = np.linalg.norm(BB2 - np.dot(BB2, theta_B_prime) * theta_B_prime)
    d_B3 = np.linalg.norm(BB3 - np.dot(BB3, theta_B_prime) * theta_B_prime)
    d_B4 = np.linalg.norm(BB4 - np.dot(BB4, theta_B_prime) * theta_B_prime)
    return max(d_B1, d_B2, d_B3, d_B4)

def compute_MFD(D_t1, d_A, d_B):
    return D_t1 - (d_A + d_B)


def get_EI(samples, toreturn='dataframe', D_safe=0.):
# def main(file_path, output_file, D_0, π, gamma, D_safe):
#     df = pd.read_csv(file_path)

#     df['Q_Veh_ID'] = ''
#     df['TDM (s)'] = ''
#     df['InDepth (m)'] = ''
#     df['EI (m/s)'] = ''
    gamma = 0.01396
    '''
    D_0 is designed to select relevant (i.e., close enough) vehicles, and is not used in this function
    gamma: the default value recommended by the original authors is 0.01396 rad (0.8 degree)
    D_safe: the default value is set to 0, which means considering the bounding boxes of vehicles with no buffer
            the original authors implicitly set D_safe to 5 by "InDepth > -5" in the original script
    '''
    original_indices = samples.index.values
    samples = samples.reset_index(drop=True)

    # Iterate over each row (moment in a case) for calculation
    progress_bar = tqdm(total=len(samples), desc='Calculating EI', ascii=True)
    for index in samples.index.values:
#    # Iterate over each row for calculation
#     for i, row in df.iterrows():
#         x_A, y_A, v_A, h_A, l_A, w_A = row.iloc[1:7]
#         same_time_rows = df[(df[df.columns[0]] == row.iloc[0]) & (df.index != i)]

#         P11_potential_indices = []
#         P11_indices = []
#         P12_potential_indices = []
#         P12_indices = []
#         P2_indices = []

#         Q_vehicle_ids = []

#         for j, same_time_row in same_time_rows.iterrows():
#             x_B, y_B, v_B, h_B, l_B, w_B = same_time_row.iloc[1:7]
        x_A, y_A, v_A, h_A, l_A, w_A = samples.loc[index, ['x_i', 'y_i', 'v_i', 'psi_i', 'length_i', 'width_i']].values
        x_B, y_B, v_B, h_B, l_B, w_B = samples.loc[index, ['x_j', 'y_j', 'v_j', 'psi_j', 'length_j', 'width_j']].values

        delta_x = x_B - x_A
        delta_y = y_B - y_A
        norm_delta = np.sqrt(delta_x ** 2 + delta_y ** 2)
        if norm_delta != 0:
            unit_vector = np.array([delta_x / norm_delta, delta_y / norm_delta])
            velocity_diff = np.array([v_B * np.cos(h_B) - v_A * np.cos(h_A), v_B * np.sin(h_B) - v_A * np.sin(h_A)])
            v_B_r = -np.dot(unit_vector, velocity_diff)
        else:
            v_B_r = 0

        v_B_r = round(v_B_r, 2)
#            if v_B_r > 0:
#                P2_indices.append(j)
        if v_B_r > 0:
            condition_P2 = True
        else:
            condition_P2 = False

#            D = np.sqrt(delta_x ** 2 + delta_y ** 2)

#            condition1 = D < D_0
#            condition2 = (abs(h_B - h_A) <= gamma) or (abs(h_B - h_A + π) <= gamma) or (abs(h_B - h_A - π) <= gamma) or (abs(abs(h_B - h_A) - π) <= gamma)

#            if condition1 and condition2:
#                P12_potential_indices.append(j)
#            elif condition1:
#                P11_potential_indices.append(j)
        condition_P11 = True
        if (abs(h_B - h_A) <= gamma) or (abs(h_B - h_A + np.pi) <= gamma) or (abs(h_B - h_A - np.pi) <= gamma) or (abs(abs(h_B - h_A) - np.pi) <= gamma):
            condition_P12 = True
        else:
            condition_P12 = False

        if condition_P11:
#        for j in P11_potential_indices:
#            same_time_row = df.iloc[j]
#            x_B, y_B, v_B, h_B, l_B, w_B = same_time_row.iloc[1:7]

            Pc = compute_Pc(x_A, y_A, x_B, y_B, h_A, h_B)
            dac = compute_dac(w_A, h_A, h_B)
            dbc = compute_dbc(w_B, h_A, h_B)

            condition_i = any(np.dot([np.cos(h_A), np.sin(h_A)], compute_A_bcC_i(Pc, dac, dbc, h_A, h_B, x_A, y_A, l_A, i)) > 0 for i in range(1, 5))
            condition_j = any(np.dot([np.cos(h_B), np.sin(h_B)], compute_B_bcC_j(Pc, dac, dbc, h_A, h_B, x_B, y_B, l_B, j)) > 0 for j in range(1, 5))

            if condition_i and condition_j:
#                P11_indices.append(j)
                condition_P11 = True
            else:
                condition_P11 = False

        if condition_P12:
#        for j in P12_potential_indices:
#            same_time_row = df.iloc[j]

#            x_B, y_B, v_B, h_B, l_B, w_B = same_time_row.iloc[1:7]
            delta_x = x_B - x_A
            delta_y = y_B - y_A

            P_1221 = np.dot([delta_x, delta_y], [np.cos(h_A), np.sin(h_A)])
            P_1222 = np.dot([-delta_x, -delta_y], [np.cos(h_B), np.sin(h_B)])
            P_1223 = np.abs(delta_x * np.cos(h_A) + delta_y * (-np.sin(h_A))) - (l_A + l_B) / 2
            P_123 = np.abs(delta_x * np.sin(h_A) - delta_y * np.cos(h_A)) - (w_A + w_B) / 2
            if (P_1221 >= 0 or P_1222 >= 0 or P_1223 <= 0) and P_123 <= 0:
#                P12_indices.append(j)
                condition_P12 = True
            else:
                condition_P12 = False

#         P1_indices = list(set(P11_indices) | set(P12_indices))
        condition_P1 = condition_P11 or condition_P12

#         Q_indices = list(set(P2_indices) & set(P1_indices))
        condition_Q = condition_P2 and condition_P1

        if condition_Q:
#         for j in Q_indices:
#             vehicle_id_B = df.iloc[j]['Vehicle ID']
#             Q_vehicle_ids.append(vehicle_id_B)
#         df.at[i, 'Q_Veh_ID'] = ';'.join(map(str, Q_vehicle_ids))

#         TDM_values = []
#         InDepth_values = []
#         EI_values = []

#         for j in Q_indices:
#             same_time_row = df.iloc[j]
#             x_B, y_B, v_B, h_B, l_B, w_B = same_time_row.iloc[1:7]

            theta_B_prime = compute_theta_B_prime(v_A, h_A, v_B, h_B)
            D_t1 = compute_D_t1(x_A, y_A, x_B, y_B, theta_B_prime)
            TDM, D_B_prime, v_B_prime_norm = compute_TDM(x_A, y_A, x_B, y_B, theta_B_prime, v_A, h_A, v_B, h_B)
            d_A = compute_d_A(x_A, y_A, h_A, l_A, w_A, theta_B_prime)
            d_B = compute_d_B(x_B, y_B, h_B, l_B, w_B, theta_B_prime)

            MFD = compute_MFD(D_t1, d_A, d_B)
            InDepth = D_safe - MFD
            EI = InDepth / TDM

#             if 0 < TDM < 5 and InDepth > -5 and EI > -10:
#                 TDM_values.append(round(TDM, 2))
#                 InDepth_values.append(round(InDepth, 2))
#                 EI_values.append(round(EI, 2))
            samples.loc[index, 'EI'] = EI

#         df.at[i, 'TDM (s)'] = ','.join(map(str, TDM_values))
#         df.at[i, 'InDepth (m)'] = ','.join(map(str, InDepth_values))
#         df.at[i, 'EI (m/s)'] = ','.join(map(str, EI_values))
        else:
            samples.loc[index, 'EI'] = -np.inf

        if index%10000 == 9999:
            progress_bar.update(10000)

#     df.to_csv(output_file, index=False)
#     print(f"finish: {output_file}")

    progress_bar.update(len(samples) % 10000)
    progress_bar.close()

    samples = samples.set_index(original_indices)
    if toreturn=='dataframe':
        return samples
    elif toreturn=='values':
        return samples['EI'].values


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--file_path', default='data/Case1_angle_conflict_raw_input.csv')
#     parser.add_argument('--output_file', default='data/Case1_angle_conflict_EI_output.csv')
#     parser.add_argument('--D_0', default=100, help='The distance range of interest')
#     parser.add_argument('--π', default=3.14159)
#     parser.add_argument('--gamma', default=0.01396, help='If the angle difference between two vehicles is less than gamma, it is considered a parallel situation (used in Condition P1)')
#     parser.add_argument('--D_safe', default=0, help='D_safe is temporarily set to 0')
#     args = parser.parse_args()

#     main(args.file_path, args.output_file, args.D_0, args.π, args.gamma, args.D_safe)
