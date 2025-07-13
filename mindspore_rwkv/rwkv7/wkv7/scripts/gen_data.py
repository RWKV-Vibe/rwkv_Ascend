import numpy as np
import torch
import torch_npu
import os


def rwkv_time_mix(B, T, C, H, data_type):
    N = C // H  # 头的维度
    param_shape = (B, H, T, N)
    u_shape = (H, N)
    state_shape = (B, H, N, N)
    k = np.random.uniform(-1, 1, param_shape).astype(data_type)
    v = np.random.uniform(-1, 1, param_shape).astype(data_type)
    w = np.random.uniform(-1, 1, param_shape).astype(data_type)
    q = np.random.uniform(-1, 1, param_shape).astype(data_type)
    u = np.random.uniform(-1, 1, u_shape).astype(data_type)
    # h = np.random.uniform(-1, 1, param_shape).astype(data_type)
    h = np.zeros(state_shape).astype(data_type)
    o = np.zeros(param_shape).astype(data_type)
    input_dir = "./input/"
    # save k, v, w, r, u, o original values
    k.tofile(os.path.join(input_dir,  "input_k.bin"))
    v.tofile(os.path.join(input_dir,  "input_v.bin"))
    w.tofile(os.path.join(input_dir,  "input_w.bin"))
    q.tofile(os.path.join(input_dir,  "input_q.bin"))
    u.tofile(os.path.join(input_dir,  "input_u.bin"))
    h.tofile(os.path.join(input_dir,  "input_h0.bin"))
    # print("h0:",h)
    np.save(os.path.join(input_dir, "input_k.npy"), k)
    np.save(os.path.join(input_dir, "input_v.npy"), v)
    np.save(os.path.join(input_dir, "input_w.npy"), w)
    np.save(os.path.join(input_dir, "input_q.npy"), q)
    np.save(os.path.join(input_dir, "input_u.npy"), u)
    np.save(os.path.join(input_dir, "input_h0.npy"), h)

    for b_i in range(B):
        for h_i in range(H):
            for i in range(N):
                for t in range(T):
                    for j in range(N):
                        x = (k[b_i, h_i, t, j] * 0.5) * (v[b_i, h_i, t, i] * 0.5)
                        s = h[b_i, h_i, j, i]
                        # print(h)
                        o[b_i, h_i, t, i] += q[b_i, h_i, t, j] * (u[h_i, j] * x + s)
                        # we are actually using exp(-exo(w)) as decay coefficient, which is always within (0,1)
                        h[b_i, h_i, j, i] = s * np.exp(-np.exp(w[b_i, h_i, t, j])) + x

    # output o_golden bin
    o.tofile(os.path.join( "./output/output_o_golden.bin"))
    np.save(os.path.join( "./output/output_o_golden.bin.npy"), o)
    print("o:", o)
    # output h_golden bin
    h.tofile(os.path.join( "./output/output_ht_golden.bin"))
    np.save(os.path.join( "./output/output_ht_golden.bin.npy"), o)
    print("ht:", h)
    return


if __name__ == "__main__":
    B = 1
    T = 64
    C = 64
    H = 1
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    np.set_printoptions(threshold=np.inf)
    data_type = np.float16
    rwkv_time_mix(B, T, C, H, data_type)
