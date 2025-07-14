import numpy as np
import torch
import torch_npu
import os


def rwkv_time_mix(B, T, N, H, data_type):
    # N = C // H  # 头的维度
    param_shape = (B, H, T, N)
    u_shape = (H, N)
    state_shape = (B, H, N, N)
    k = np.random.uniform(-1, 1, param_shape).astype(data_type)
    v = np.random.uniform(-1, 1, param_shape).astype(data_type)
    w = np.random.uniform(-1, 1, param_shape).astype(data_type)
    q = np.random.uniform(-1, 1, param_shape).astype(data_type)
    a = np.random.uniform(-1, 1, param_shape).astype(data_type)
    b = np.random.uniform(-1, 1, param_shape).astype(data_type)
    # u = np.random.uniform(-1, 1, u_shape).astype(data_type)
    # h = np.random.uniform(-1, 1, param_shape).astype(data_type)
    h = np.zeros(state_shape).astype(data_type)
    o = np.zeros(param_shape).astype(data_type)
    input_dir = "./input/"
    # save k, v, w, r, u, o original values
    k.tofile(os.path.join(input_dir,  "input_k.bin"))
    v.tofile(os.path.join(input_dir,  "input_v.bin"))
    w.tofile(os.path.join(input_dir,  "input_w.bin"))
    q.tofile(os.path.join(input_dir,  "input_q.bin"))
    a.tofile(os.path.join(input_dir,  "input_a.bin"))
    b.tofile(os.path.join(input_dir,  "input_b.bin"))
    # u.tofile(os.path.join(input_dir,  "input_u.bin"))
    h.tofile(os.path.join(input_dir,  "input_h0.bin"))
    # print("h0:",h)
    np.save(os.path.join(input_dir, "input_k.npy"), k)
    np.save(os.path.join(input_dir, "input_v.npy"), v)
    np.save(os.path.join(input_dir, "input_w.npy"), w)
    np.save(os.path.join(input_dir, "input_q.npy"), q)
    np.save(os.path.join(input_dir, "input_a.npy"), a)
    np.save(os.path.join(input_dir, "input_b.npy"), b)
    # np.save(os.path.join(input_dir, "input_u.npy"), u)
    np.save(os.path.join(input_dir, "input_h0.npy"), h)

    for t in range(T):
        for bi in range(B):
            for hi in range(H):
                q_t = q[bi, hi, t]
                k_t = k[bi, hi, t]
                v_t = v[bi, hi, t]
                a_t = a[bi, hi, t]
                b_t = b[bi, hi, t]
                w_t = np.exp(w[bi, hi, t])

                sa = np.sum((a_t[None, :] * h[bi, hi]), axis=1)

                h[bi, hi] = (h[bi, hi] * w_t[None, :] +  # [N,V] * [N,1]
                                 k_t[None, :] * v_t[:, None] +     # [N,1] * [1,V]
                                 sa[:, None] * b_t[None, :])                # [V] * [N,1]

                y = np.sum((h[bi, hi] * q_t[None, :]), axis=1)

                o[bi, hi, t] = y
    # output o_golden bin
    o.tofile(os.path.join( "./output/output_o_golden.bin"))
    np.save(os.path.join( "./output/output_o_golden.bin.npy"), o)
    # print("o:", o)
    # output h_golden bin
    h.tofile(os.path.join( "./output/output_ht_golden.bin"))
    np.save(os.path.join( "./output/output_ht_golden.bin.npy"), o)
    # print("ht:", h)
    return


if __name__ == "__main__":
    B = 1
    T = 64
    N = 64
    H = 1
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    np.set_printoptions(threshold=np.inf)
    data_type = np.float32
    rwkv_time_mix(B, T, N, H, data_type)


