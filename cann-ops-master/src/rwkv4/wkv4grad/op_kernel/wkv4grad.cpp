
#include "kernel_operator.h"

using namespace AscendC;
#define BUFFER_NUM 1

template <typename T>
class KernelWkvGrad
{
public:
    __aicore__ inline KernelWkvGrad()
    {
    }
    
    /*
    Init  ,根据tiling 设置 算子所需要的参数
    设置每个核 需要处理的数据（对数据偏移）
    初始化 管道、gm、localtensor<T> 地址
    */
    __aicore__ inline void Init(GM_ADDR w, GM_ADDR u, GM_ADDR k, GM_ADDR v, GM_ADDR gy,
                            GM_ADDR gw, GM_ADDR gu, GM_ADDR gk, GM_ADDR gv, GM_ADDR workspace, GM_ADDR tiling)
    {

        GET_TILING_DATA(tiling_data, tiling);
        batch = tiling_data.B;
        Tile = tiling_data.T;
        C = tiling_data.C;
        int ub_max = tiling_data.ub_size;

        // inque outque 的数量*BUFFER_NUM  + LCMbuff 的数量
        int use_block_nums = 10 * BUFFER_NUM + 14;
        int once_cal_sizes = use_block_nums * sizeof(T);
        int min_nums = (32 / sizeof(T));

        // 每个epoch 最大处理的数据量
        max_n_values = int(ub_max / once_cal_sizes);
        max_n_values = max_n_values >= C + min_nums ? C + min_nums : max_n_values;
        max_n_values -= max_n_values % (32 / sizeof(T));

        // 尾块据量
        tail_n_values = C % max_n_values;
        n_epoch = int((C - 1) / max_n_values) + 1;

        // 尾块对齐32b，进行填充
        int padd_nums = 0;
        if (tail_n_values % min_nums != 0)
        {
            padd_nums = min_nums - tail_n_values % min_nums;
        }

        // 真实尾块长度，和填充后的尾块长度
        tail_n_values_real = tail_n_values;
        tail_n_values += padd_nums;

        // batch_now 当前核 所需要计算的 batch 数
        int batch_min = int(batch / block_num);
        if ((batch % block_num) >= block_idx + 1)
        {
            batch_now = batch_min + 1;
        }
        else
        {
            batch_now = batch_min;
        }

        // 当前核 根据需要处理的batch 对数据偏移
        int batch_offset = block_idx >= batch % block_num ? batch % block_num : block_idx;
        int start_batch = (block_idx * batch_min + batch_offset);
        int singleBatchAddr = Tile * C * sizeof(T);
        int start_offset = start_batch * singleBatchAddr;

        // 中间变量分为14块，gm到ub共需要6块空间，ub到gm共需要4块空间
        pipe.InitBuffer(all_buf, 14 * max_n_values * sizeof(T));
        pipe.InitBuffer(inQueue_all, BUFFER_NUM, 6 * max_n_values * sizeof(T));
        pipe.InitBuffer(outQueue_all, BUFFER_NUM, 4 * max_n_values * sizeof(T));

        // 全局gm地址 设置
        gm_w.SetGlobalBuffer((__gm__ T *)(w));
        gm_u.SetGlobalBuffer((__gm__ T *)(u));
        gm_k.SetGlobalBuffer((__gm__ T *)(k + start_offset));
        gm_v.SetGlobalBuffer((__gm__ T *)(v + start_offset));
        gm_gy.SetGlobalBuffer((__gm__ T *)(gy + start_offset));
        dst_gw.SetGlobalBuffer((__gm__ T *)(gw + start_batch * C * sizeof(T)));
        dst_gu.SetGlobalBuffer((__gm__ T *)(gu + start_batch * C * sizeof(T)));
        dst_gk.SetGlobalBuffer((__gm__ T *)(gk + start_offset));
        dst_gv.SetGlobalBuffer((__gm__ T *)(gv + start_offset));

        size_t single_enum_sizes = batch * Tile * C * sizeof(T);
        gm_ze.SetGlobalBuffer((__gm__ T *)(workspace + single_enum_sizes * 0 + start_offset));
        gm_z.SetGlobalBuffer((__gm__ T *)(workspace + single_enum_sizes * 1 + start_offset));
        gm_y.SetGlobalBuffer((__gm__ T *)(workspace + single_enum_sizes * 2 + start_offset));

        // 全局localtensor地址设置
        // all_buf分为 14 个变量空间，共14个中间变量使用这块空间, 但是因为芯片不支持定义太多全局变量，部分变量以临时变量的形式在使用的时候才定义。
        vec_m = all_buf.Get<T>(14 * max_n_values);
        vec_p = vec_m[max_n_values];
        vec_q = vec_m[2 * max_n_values];
        vec_w = vec_m[5 * max_n_values];
        vec_u = vec_m[6 * max_n_values];
        vec_a = vec_m[7 * max_n_values];
        vec_b = vec_m[8 * max_n_values];
        vec_v1 = vec_m[9 * max_n_values];
        vec_x1 = vec_m[10 * max_n_values];
    }

    __aicore__ inline void process()
    {

        for (int batch_id = 0; batch_id < batch_now; batch_id++)
        {
            for (int i = 0; i < n_epoch; i++)
            {
                // 计算偏移值
                size_t offset_c = i * max_n_values;
                size_t offset_st = batch_id * Tile * C + offset_c;
                size_t n_values = i + 1 == n_epoch ? tail_n_values : max_n_values;
                real_n_value = i + 1 == n_epoch ? tail_n_values_real : max_n_values;

                // 前处理，初始化数据
                preprocess(offset_c, n_values);

                // 核心处理步骤
                stage1(offset_st, n_values);
                set_flag(PIPE_MTE3, PIPE_MTE2, 0);
                wait_flag(PIPE_MTE3, PIPE_MTE2, 0);
                stage2(offset_st, n_values);

                // 后处理，输出结果
                postprocess(offset_c, n_values, batch_id);
            }
        }
    }

    /*
    算子主要计算逻辑1，将计算部分分为 copyin compute copyout。
    所有gm→ub 的统一放在copyin，
    ub计算统一放在 compute，
    gm→ub 的统一放在copyout
    */
    __aicore__ inline void stage1(int offset_st, int n_values)
    {

        for (size_t t_id = 0; t_id < Tile; ++t_id)
        {
            size_t offset = offset_st + t_id * C;
            copyin1(offset, n_values);
            compute1(offset, n_values);
            copyout1(offset, n_values);
        }
    }

    /*
    算子主要计算逻辑2，将计算部分分为 copyin compute copyout。
    所有gm→ub 的统一放在copyin，
    ub计算统一放在 compute，
    gm→ub 的统一放在copyout
    */
    __aicore__ inline void stage2(int offset_st, int n_values)
    {

        Duplicate(vec_m, T(-1e+38), n_values);
        Duplicate(vec_p, T(0), n_values);
        Duplicate(vec_q, T(0), n_values);

        pipe_barrier(PIPE_ALL);
        for (size_t t_id = Tile; t_id > 0; --t_id)
        {
            size_t offset = offset_st + (t_id - 1) * C;
            copyin2(offset, n_values);
            compute2(offset, n_values);
            copyout2(offset, n_values);
        }
    }

    __aicore__ inline void copyin1(int offset, int n_values)
    {
        // vec_k 其实存放了多个中间变量，每个中间变量的大小是n_values
        LocalTensor<T> vec_k;
        vec_k = inQueue_all.AllocTensor<T>();
        DataCopy(vec_k, gm_k[offset], n_values);
        DataCopy(vec_k[n_values], gm_v[offset], n_values);
        DataCopy(vec_k[2 * n_values], gm_gy[offset], n_values);
        inQueue_all.EnQue<T>(vec_k);
    }

    __aicore__ inline void compute1(int offset, int n_values)
    {
        LocalTensor<T> vec_k, vec_v, vec_x, vec_z1, vec_y;

        // inQueue_all 里实际存了 3个 变量的数据，分别放到3个变量中
        vec_k = inQueue_all.DeQue<T>();
        vec_v = vec_k[n_values];
        vec_x = vec_k[2 * n_values];

        // outQueue_all 里实际存了 3个 需要输出的变量
        vec_z1 = outQueue_all.AllocTensor<T>();
        vec_y = vec_z1[2 * n_values];

        // 定义的临时的LocalTensor，其实都应该放在全局变量中的，但芯片的栈空间不够。
        LocalTensor<T> vec_z, vec_gw, vec_gu, vec_dpdw, vec_dqdw;
        vec_gw = vec_m[12 * max_n_values];
        vec_gu = vec_m[13 * max_n_values];
        vec_dpdw = vec_m[3 * max_n_values];
        vec_dqdw = vec_m[4 * max_n_values];
        vec_z = vec_m[11 * max_n_values];

        Add(vec_v1, vec_u, vec_k, n_values);
        Max(vec_x1, vec_v1, vec_m, n_values);
        Sub(vec_a, vec_m, vec_x1, n_values);
        Exp(vec_a, vec_a, n_values);
        Sub(vec_z, vec_v1, vec_x1, n_values);
        pipe_barrier(PIPE_V);
        DataCopy(vec_z1, vec_z, n_values);
        pipe_barrier(PIPE_V);
        Exp(vec_b, vec_z, n_values);
        Mul(vec_y, vec_a, vec_p, n_values);
        Mul(vec_z, vec_b, vec_v, n_values);
        Add(vec_y, vec_y, vec_z, n_values);
        Mul(vec_z, vec_a, vec_q, n_values);
        Add(vec_z, vec_z, vec_b, n_values);
        pipe_barrier(PIPE_V);
        DataCopy(vec_z1[n_values], vec_z, n_values);
        pipe_barrier(PIPE_V);
        Div(vec_y, vec_y, vec_z, n_values);
        Div(vec_x, vec_x, vec_z, n_values);
        Sub(vec_z, vec_v, vec_y, n_values);
        Mul(vec_z, vec_z, vec_x, n_values);
        Mul(vec_z, vec_z, vec_b, n_values);
        Add(vec_gu, vec_gu, vec_z, n_values);
        Mul(vec_z, vec_dqdw, vec_y, n_values);
        Sub(vec_z, vec_dpdw, vec_z, n_values);
        Mul(vec_z, vec_z, vec_x, n_values);
        Mul(vec_z, vec_z, vec_a, n_values);
        Add(vec_gw, vec_gw, vec_z, n_values);
        Add(vec_x, vec_w, vec_m, n_values);
        Max(vec_m, vec_x, vec_k, n_values);
        Sub(vec_a, vec_x, vec_m, n_values);
        Exp(vec_a, vec_a, n_values);
        Sub(vec_b, vec_k, vec_m, n_values);
        Exp(vec_b, vec_b, n_values);
        Add(vec_dpdw, vec_dpdw, vec_p, n_values);
        Mul(vec_dpdw, vec_dpdw, vec_a, n_values);
        Add(vec_dqdw, vec_dqdw, vec_q, n_values);
        Mul(vec_dqdw, vec_dqdw, vec_a, n_values);
        Mul(vec_q, vec_q, vec_a, n_values);
        Add(vec_q, vec_q, vec_b, n_values);
        Mul(vec_b, vec_b, vec_v, n_values);
        Mul(vec_p, vec_p, vec_a, n_values);
        Add(vec_p, vec_p, vec_b, n_values);
        outQueue_all.EnQue<T>(vec_z1);
        inQueue_all.FreeTensor(vec_k);
    }

    // 将 outQueue_all 里的3个变量分别输出到 gm
    __aicore__ inline void copyout1(int offset, int n_values)
    {

        LocalTensor<T> vec_z_out = outQueue_all.DeQue<T>();
// 910B用非对齐的搬运方式， 910A采用datacopy
#if defined(DAV_C220_VEC)

        if (sizeof(T) == 4)
        {

            copy_ubuf_to_gm_align_b32((__gm__ T *)gm_ze[offset].GetPhyAddr(), (__ubuf__ T *)vec_z_out.GetPhyAddr(), 0,
                                      1, real_n_value * sizeof(T), 0, 0, 0, 0);
            copy_ubuf_to_gm_align_b32((__gm__ T *)gm_z[offset].GetPhyAddr(), (__ubuf__ T *)vec_z_out[n_values].GetPhyAddr(), 0,
                                      1, real_n_value * sizeof(T), 0, 0, 0, 0);
            copy_ubuf_to_gm_align_b32((__gm__ T *)gm_y[offset].GetPhyAddr(), (__ubuf__ T *)vec_z_out[2 * n_values].GetPhyAddr(), 0,
                                      1, real_n_value * sizeof(T), 0, 0, 0, 0);
        }
        else if (sizeof(T) == 2)
        {
            copy_ubuf_to_gm_align_b16((__gm__ T *)gm_ze[offset].GetPhyAddr(), (__ubuf__ T *)vec_z_out.GetPhyAddr(), 0,
                                      1, real_n_value * sizeof(T), 0, 0, 0, 0);
            copy_ubuf_to_gm_align_b16((__gm__ T *)gm_z[offset].GetPhyAddr(), (__ubuf__ T *)vec_z_out[n_values].GetPhyAddr(), 0,
                                      1, real_n_value * sizeof(T), 0, 0, 0, 0);
            copy_ubuf_to_gm_align_b16((__gm__ T *)gm_y[offset].GetPhyAddr(), (__ubuf__ T *)vec_z_out[2 * n_values].GetPhyAddr(), 0,
                                      1, real_n_value * sizeof(T), 0, 0, 0, 0);
        }
#else
        DataCopy(gm_ze[offset], vec_z_out, n_values);
        DataCopy(gm_z[offset], vec_z_out[n_values], n_values);
        DataCopy(gm_y[offset], vec_z_out[2 * n_values], n_values);
#endif

        outQueue_all.FreeTensor(vec_z_out);
    }

    __aicore__ inline void copyin2(int offset, int n_values)
    {
        LocalTensor<T> vec_k_v;
        vec_k_v = inQueue_all.AllocTensor<T>();
        DataCopy(vec_k_v, gm_k[offset], n_values);
        DataCopy(vec_k_v[n_values], gm_v[offset], n_values);
        DataCopy(vec_k_v[2 * n_values], gm_gy[offset], n_values);
        DataCopy(vec_k_v[3 * n_values], gm_z[offset], n_values);
        DataCopy(vec_k_v[4 * n_values], gm_ze[offset], n_values);
        DataCopy(vec_k_v[5 * n_values], gm_y[offset], n_values);

        inQueue_all.EnQue<T>(vec_k_v);
    }

    __aicore__ inline void compute2(int offset, int n_values)
    {

        LocalTensor<T> vec_k, vec_v, vec_x, vec_z, vec_y, vec_dpdw2;
        vec_k = inQueue_all.DeQue<T>();
        vec_v = vec_k[n_values];
        vec_x = vec_k[2 * n_values];
        vec_z = vec_k[3 * n_values];
        vec_dpdw2 = vec_z[1 * n_values];
        vec_y = vec_z[2 * n_values];

        vec_x1 = outQueue_all.AllocTensor<T>();
        Div(vec_z, vec_x, vec_z, n_values);
        Exp(vec_x, vec_dpdw2, n_values);
        Mul(vec_a, vec_z, vec_x, n_values);
        Add(vec_b, vec_k, vec_m, n_values);
        Exp(vec_b, vec_b, n_values);
        Mul(vec_x, vec_b, vec_p, n_values);
        Add(vec_x, vec_x, vec_a, n_values);
        pipe_barrier(PIPE_V);
        DataCopy(vec_x1, vec_x, n_values);
        pipe_barrier(PIPE_V);
        Sub(vec_x, vec_v, vec_y, n_values);
        Mul(vec_v, vec_v, vec_p, n_values);
        Add(vec_v, vec_v, vec_q, n_values);
        Mul(vec_v, vec_v, vec_b, n_values);
        Mul(vec_x, vec_x, vec_a, n_values);
        Add(vec_x, vec_x, vec_v, n_values);
        pipe_barrier(PIPE_V);
        DataCopy(vec_x1[n_values], vec_x, n_values);
        pipe_barrier(PIPE_V);
        Add(vec_x, vec_w, vec_m, n_values);
        Sub(vec_dpdw2, vec_dpdw2, vec_k, n_values);
        Sub(vec_dpdw2, vec_dpdw2, vec_u, n_values);
        Max(vec_m, vec_x, vec_dpdw2, n_values);
        Sub(vec_a, vec_x, vec_m, n_values);
        Exp(vec_a, vec_a, n_values);
        Sub(vec_b, vec_dpdw2, vec_m, n_values);
        Exp(vec_b, vec_b, n_values);
        Mul(vec_b, vec_b, vec_z, n_values);
        Mul(vec_p, vec_p, vec_a, n_values);
        Add(vec_p, vec_p, vec_b, n_values);
        Mul(vec_x, vec_b, vec_y, n_values);
        Mul(vec_q, vec_q, vec_a, n_values);
        Sub(vec_q, vec_q, vec_x, n_values);

        outQueue_all.EnQue<T>(vec_x1);
        inQueue_all.FreeTensor(vec_k);
    }

    __aicore__ inline void copyout2(int offset, int n_values)
    {
        LocalTensor<T> vec_k_v = outQueue_all.DeQue<T>();
#if defined(DAV_C220_VEC)

        if (sizeof(T) == 4)
        {
            copy_ubuf_to_gm_align_b32((__gm__ T *)dst_gv[offset].GetPhyAddr(), (__ubuf__ T *)vec_k_v.GetPhyAddr(), 0,
                                      1, real_n_value * sizeof(T), 0, 0, 0, 0);
            copy_ubuf_to_gm_align_b32((__gm__ T *)dst_gk[offset].GetPhyAddr(), (__ubuf__ T *)vec_k_v[n_values].GetPhyAddr(), 0,
                                      1, real_n_value * sizeof(T), 0, 0, 0, 0);
        }
        else if (sizeof(T) == 2)
        {
            copy_ubuf_to_gm_align_b16((__gm__ T *)dst_gv[offset].GetPhyAddr(), (__ubuf__ T *)vec_k_v.GetPhyAddr(), 0,
                                      1, real_n_value * sizeof(T), 0, 0, 0, 0);
            copy_ubuf_to_gm_align_b16((__gm__ T *)dst_gk[offset].GetPhyAddr(), (__ubuf__ T *)vec_k_v[n_values].GetPhyAddr(), 0,
                                      1, real_n_value * sizeof(T), 0, 0, 0, 0);
        }
#else
        DataCopy(dst_gv[offset], vec_k_v, n_values);
        DataCopy(dst_gk[offset], vec_k_v[n_values], n_values);
#endif
        outQueue_all.FreeTensor(vec_k_v);
    }

    __aicore__ inline void preprocess(int offset_c, int n_values)
    {
        LocalTensor<T> vec_gw, vec_gu, vec_dpdw, vec_dqdw;
        vec_gw = vec_m[12 * max_n_values];
        vec_gu = vec_m[13 * max_n_values];
        vec_dpdw = vec_m[3 * max_n_values];
        vec_dqdw = vec_m[4 * max_n_values];

        Duplicate(vec_m, T(-1e+38), max_n_values);
        Duplicate(vec_gw, T(0), max_n_values);
        Duplicate(vec_gu, T(0), max_n_values);
        Duplicate(vec_p, T(0), max_n_values);
        Duplicate(vec_q, T(0), max_n_values);
        Duplicate(vec_dpdw, T(0), max_n_values);
        Duplicate(vec_dqdw, T(0), max_n_values);
        LocalTensor<T> vec_w_u;
        vec_w_u = inQueue_all.AllocTensor<T>();
        DataCopy(vec_w_u, gm_w[offset_c], n_values);
        DataCopy(vec_w_u[n_values], gm_u[offset_c], n_values);
        inQueue_all.EnQue<T>(vec_w_u);
        vec_w_u = inQueue_all.DeQue<T>();

        DataCopy(vec_w, vec_w_u, n_values);
        DataCopy(vec_u, vec_w_u[n_values], n_values);
        inQueue_all.FreeTensor(vec_w_u);
    }

    __aicore__ inline void postprocess(int offset_c, int n_values, int batch_id)
    {
        LocalTensor<T> vec_out_gw_u;
        LocalTensor<T> vec_gw, vec_gu;
        vec_gw = vec_m[12 * max_n_values];
        vec_gu = vec_m[13 * max_n_values];

        vec_out_gw_u = outQueue_all.AllocTensor<T>();
        DataCopy(vec_out_gw_u, vec_gw, n_values);
        DataCopy(vec_out_gw_u[n_values], vec_gu, n_values);
        outQueue_all.EnQue<T>(vec_out_gw_u);
        vec_out_gw_u = outQueue_all.DeQue<T>();
#if defined(DAV_C220_VEC)
        if (sizeof(T) == 4)
        {
            copy_ubuf_to_gm_align_b32((gm T *)dst_gw[offset_c + batch_id * C].GetPhyAddr(), (ubuf T *)vec_out_gw_u.GetPhyAddr(), 0,
                                      1, real_n_value * sizeof(T), 0, 0, 0, 0);
            copy_ubuf_to_gm_align_b32((gm T *)dst_gu[offset_c + batch_id * C].GetPhyAddr(), (ubuf T *)vec_out_gw_u[n_values].GetPhyAddr(), 0,
                                      1, real_n_value * sizeof(T), 0, 0, 0, 0);
        }
        else if (sizeof(T) == 2)
        {
            copy_ubuf_to_gm_align_b16((gm T *)dst_gw[offset_c + batch_id * C].GetPhyAddr(), (ubuf T *)vec_out_gw_u.GetPhyAddr(), 0,
                                      1, real_n_value * sizeof(T), 0, 0, 0, 0);
            copy_ubuf_to_gm_align_b16((gm T *)dst_gu[offset_c + batch_id * C].GetPhyAddr(), (ubuf T *)vec_out_gw_u[n_values].GetPhyAddr(), 0,
                                      1, real_n_value * sizeof(T), 0, 0, 0, 0);
        }
#else
        DataCopy(dst_gw[offset_c + batch_id * C], vec_out_gw_u, n_values);
        DataCopy(dst_gu[offset_c + batch_id * C], vec_out_gw_u[n_values], n_values);
#endif
        outQueue_all.FreeTensor(vec_out_gw_u);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueue_all;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue_all;
    // 用来存放 ub 的中间变量
    TBuf<QuePosition::LCM> all_buf;

    // 这些变量会在多个stage中多个循环中使用，定义为全局变量，可以减少算子的scalar耗时。（每次重新定义和get的过程会增加耗时）
    LocalTensor<T> vec_a, vec_m, vec_b, vec_p, vec_q, vec_x1, vec_v1, vec_w, vec_u;

    GlobalTensor<T> gm_w, gm_u, gm_k, gm_v, gm_gy, dst_gw, dst_gu, dst_gk, dst_gv;
    GlobalTensor<T> gm_ze, gm_z, gm_y;

public:
    int32_t max_n_values, tail_n_values, tail_n_values_real, n_epoch, batch_now, batch, Tile, C, real_n_value;
};

extern "C" __global__ __aicore__ void wkv4grad(GM_ADDR w, GM_ADDR u, GM_ADDR k, GM_ADDR v, GM_ADDR gy,
                                       GM_ADDR gw, GM_ADDR gu, GM_ADDR gk, GM_ADDR gv, GM_ADDR workspace, GM_ADDR tiling)

{
    KernelWkvGrad<float> op;
    GM_ADDR usrWorkspace = GetUserWorkspace(workspace);
    op.Init(w, u, k, v, gy, gw, gu, gk, gv, usrWorkspace, tiling);
    op.process();
}
