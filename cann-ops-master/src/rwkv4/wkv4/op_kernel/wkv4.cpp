#include "kernel_operator.h"
using namespace AscendC;

template <typename T>
class KernelWkv {
    public:
        __aicore__ inline KernelWkv() {
        }
        __aicore__ inline void Init(GM_ADDR w, GM_ADDR u, GM_ADDR k, GM_ADDR v,
                                    GM_ADDR m, GM_ADDR p, GM_ADDR q, GM_ADDR o,
                                    GM_ADDR mo, GM_ADDR po, GM_ADDR qo, GM_ADDR tiling) {
            GET_TILING_DATA(tiling_data, tiling);
            int32_t batch_size = tiling_data.batchSize;
            int32_t ctx_len = tiling_data.ctxLen;
            int32_t n_embd = tiling_data.nEmbd;
            int32_t ub_limit = tiling_data.ubLimit;

            int32_t num_mpq = 3;

            size_t n_items = n_embd;
            constexpr size_t n_bytes_each_item = sizeof(T) * 11;
            constexpr size_t n_items_each_block = 32 / sizeof(T);
            size_t n_items_each_tile = ub_limit / n_bytes_each_item / n_items_each_block * n_items_each_block;

            size_t n_items_in_tail = n_items % n_items_each_tile == 0 ? n_items_each_tile : n_items % n_items_each_tile;
            size_t n_items_up_tail_num = n_items_in_tail - (n_items_in_tail % n_items_each_block) + n_items_each_block;
            size_t n_items_in_up_tail = (n_items_in_tail % n_items_each_block == 0 ?
                                         n_items_in_tail : n_items_up_tail_num);
            size_t n_tiles_total = (n_items + n_items_each_tile - 1) / n_items_each_tile * n_items_each_tile;
            size_t n_tiles = n_tiles_total / n_items_each_tile;

            SingleCoreAddrLen = ctx_len * n_embd * sizeof(T);
            SingleCoreAddrLen_m_p_q = n_embd * sizeof(T);
            out_round = n_tiles;
            inner_round = ctx_len;
            Veclen = n_items_each_tile * sizeof(T);
            items_each_tile = n_items_each_tile;
            items_in_tail = n_items_in_up_tail;
            items_each_block = n_items_each_block;
            items_in_real_tail = n_items_in_tail;
            num_item = n_embd;

            pipe.InitBuffer(bbuf, Veclen);
            pipe.InitBuffer(cbuf, Veclen);
            pipe.InitBuffer(dbuf, Veclen);

            pipe.InitBuffer(winQue, 1, Veclen);
            pipe.InitBuffer(uinQue, 1, Veclen);
            pipe.InitBuffer(kinQue, 1, Veclen);
            pipe.InitBuffer(vinQue, 1, Veclen);
            pipe.InitBuffer(m_p_q_inQue, 1, num_mpq * Veclen);
            pipe.InitBuffer(outQue, 1, Veclen);

            wAddrGlobal.SetGlobalBuffer((__gm__ T *)(w));
            uAddrGlobal.SetGlobalBuffer((__gm__ T *)(u));
            kAddrGlobal.SetGlobalBuffer((__gm__ T *)(k + block_idx * SingleCoreAddrLen));
            vAddrGlobal.SetGlobalBuffer((__gm__ T *)(v + block_idx * SingleCoreAddrLen));
            oAddrGlobal.SetGlobalBuffer((__gm__ T *)(o + block_idx * SingleCoreAddrLen));
            mAddrGlobal.SetGlobalBuffer((__gm__ T *)(m + block_idx * SingleCoreAddrLen_m_p_q));
            pAddrGlobal.SetGlobalBuffer((__gm__ T *)(p + block_idx * SingleCoreAddrLen_m_p_q));
            qAddrGlobal.SetGlobalBuffer((__gm__ T *)(q + block_idx * SingleCoreAddrLen_m_p_q));
            moAddrGlobal.SetGlobalBuffer((__gm__ T *)(mo + block_idx * SingleCoreAddrLen_m_p_q));
            poAddrGlobal.SetGlobalBuffer((__gm__ T *)(po + block_idx * SingleCoreAddrLen_m_p_q));
            qoAddrGlobal.SetGlobalBuffer((__gm__ T *)(qo + block_idx * SingleCoreAddrLen_m_p_q));

            vec_c = cbuf.Get<T>(num_item);
            vec_d = dbuf.Get<T>(num_item);
            vec_b = bbuf.Get<T>(num_item);
        }

        __aicore__ inline void Process() {
            for (size_t tile_id = 0; tile_id < out_round; ++tile_id) {
                size_t n_items_in_tile = tile_id + 1 == out_round ? items_in_tail : items_each_tile;
                size_t offset_tile = tile_id * items_each_tile;

                LocalTensor<T> tensor_w = winQue.AllocTensor<T>();
                LocalTensor<T> tensor_u = uinQue.AllocTensor<T>();
                LocalTensor<T> tensor_m_p_q = m_p_q_inQue.AllocTensor<T>();

                DataCopy(tensor_w, wAddrGlobal[tile_id * items_each_tile], n_items_in_tile);
                DataCopy(tensor_u, uAddrGlobal[tile_id * items_each_tile], n_items_in_tile);

                DataCopy(tensor_m_p_q, mAddrGlobal[tile_id * items_each_tile], n_items_in_tile);
                DataCopy(tensor_m_p_q[n_items_in_tile], pAddrGlobal[tile_id * items_each_tile], n_items_in_tile);
                DataCopy(tensor_m_p_q[offset_vecnum * n_items_in_tile],
                         qAddrGlobal[tile_id * items_each_tile], n_items_in_tile);

                winQue.EnQue(tensor_w);
                uinQue.EnQue(tensor_u);
                m_p_q_inQue.EnQue(tensor_m_p_q);

                Forward_Stage(inner_round, n_items_in_tile, offset_tile);
            }
        }

        __aicore__ inline void Forward_Stage(size_t cycle_times, size_t n_items, size_t offset_in) {
            vec_w = winQue.DeQue<T>();
            vec_u = uinQue.DeQue<T>();
            vec_m = m_p_q_inQue.DeQue<T>();
            for (size_t t_id = 0; t_id < cycle_times; ++t_id) {
                size_t offset = offset_in + t_id * num_item;
                CopyIn(n_items, offset);
                Compute(n_items);
                CopyOut(n_items, offset);
            }
            DataCopy(moAddrGlobal[offset_in], vec_m, n_items);
            DataCopy(poAddrGlobal[offset_in], vec_m[n_items], n_items);
            DataCopy(qoAddrGlobal[offset_in], vec_m[offset_vecnum * n_items], n_items);
            winQue.FreeTensor(vec_w);
            uinQue.FreeTensor(vec_u);
            m_p_q_inQue.FreeTensor(vec_m);
        }

        __aicore__ inline void CopyIn(size_t n_items, size_t offset_num) {
            LocalTensor<T> tensor_a = kinQue.AllocTensor<T>();
            LocalTensor<T> tensor_e = vinQue.AllocTensor<T>();
            DataCopy(tensor_a, kAddrGlobal[offset_num], n_items);
            DataCopy(tensor_e, vAddrGlobal[offset_num], n_items);
            kinQue.EnQue(tensor_a);
            vinQue.EnQue(tensor_e);
        }

        __aicore__ inline void Compute(size_t n_items) {
            // vec_a vec_u vec_e vec_w
            LocalTensor<T> vec_a = kinQue.DeQue<T>();
            LocalTensor<T> vec_e = vinQue.DeQue<T>();
            LocalTensor<T> vec_b1 = outQue.AllocTensor<T>();

            Add(vec_d, vec_u, vec_a, n_items);
            Max(vec_b, vec_d, vec_m, n_items);

            /// A = exp(m - no)
            Sub(vec_c, vec_m, vec_b, n_items);
            Exp(vec_c, vec_c, n_items);

            /// B = exp(u + k[ii] - no)
            Sub(vec_b, vec_d, vec_b, n_items);
            Exp(vec_b, vec_b, n_items);

            /// (A * q + B)
            Mul(vec_d, vec_c, vec_m[n_items], n_items);

            Mul(vec_c, vec_c, vec_m[offset_vecnum * n_items], n_items);
            Add(vec_c, vec_c, vec_b, n_items);

            /// (A * p + B * v[ii])
            Mul(vec_b, vec_b, vec_e, n_items);

            Add(vec_b, vec_b, vec_d, n_items);
            /// y[ii] = (A * p + B * v[ii]) / (A * q + B)
            Div(vec_b, vec_b, vec_c, n_items);
            // When LCM temp variable need to copy to InQue/OutQue, we should add pipe_barrier to ensure correct copy
            pipe_barrier(PIPE_V);
            DataCopy(vec_b1, vec_b, n_items);
            /// no = max(w + m, k[ii])
            Add(vec_c, vec_w, vec_m, n_items);
            Max(vec_m, vec_c, vec_a, n_items);

            /// A = exp(w + m - no)
            Sub(vec_c, vec_c, vec_m, n_items);
            Exp(vec_c, vec_c, n_items);
            /// B = exp(k[ii] - no)
            Sub(vec_b, vec_a, vec_m, n_items);
            Exp(vec_b, vec_b, n_items);

            /// q = A * q + B
            Mul(vec_m[offset_vecnum * n_items], vec_m[offset_vecnum * n_items], vec_c, n_items);
            Add(vec_m[offset_vecnum * n_items], vec_m[offset_vecnum * n_items], vec_b, n_items);

            /// p = A * p + B * v[ii]
            Mul(vec_b, vec_b, vec_e, n_items);

            Mul(vec_m[n_items], vec_m[n_items], vec_c, n_items);
            Add(vec_m[n_items], vec_m[n_items], vec_b, n_items);

            outQue.EnQue<T>(vec_b1);

            kinQue.FreeTensor(vec_a);
            vinQue.FreeTensor(vec_e);
        }

        __aicore__ inline void CopyOut(size_t n_items, size_t offset_num) {
            LocalTensor<T> tensor_b1 = outQue.DeQue<T>();
            DataCopy(oAddrGlobal[offset_num], tensor_b1, n_items);
            outQue.FreeTensor(tensor_b1);
        }

    public:
        int32_t out_round, inner_round, SingleCoreAddrLen, SingleCoreAddrLen_m_p_q, Veclen, num_item;
        int32_t items_in_tail, items_each_tile, items_each_block, items_in_real_tail;
        size_t offset_vecnum = 2;

    private:
        TPipe pipe;
        TBuf<QuePosition::LCM> cbuf, dbuf, bbuf;
        TQue<QuePosition::VECIN, 1> winQue, uinQue, kinQue, vinQue, m_p_q_inQue;
        TQue<QuePosition::VECOUT, 1> outQue;
        GlobalTensor<T> wAddrGlobal, uAddrGlobal, kAddrGlobal, vAddrGlobal, oAddrGlobal;
        GlobalTensor<T> mAddrGlobal, pAddrGlobal, qAddrGlobal;
        GlobalTensor<T> moAddrGlobal, poAddrGlobal, qoAddrGlobal;
        LocalTensor<T> vec_m, vec_c, vec_d, vec_b, vec_w, vec_u;
};

extern "C" __global__ __aicore__ void wkv4(GM_ADDR w, GM_ADDR u, GM_ADDR k, GM_ADDR v,
                                          GM_ADDR m, GM_ADDR p, GM_ADDR q, GM_ADDR o,
                                          GM_ADDR mo, GM_ADDR po, GM_ADDR qo, GM_ADDR workspace, GM_ADDR tiling) {
    KernelWkv<float> op;
    GM_ADDR usrWorkspace = GetUserWorkspace(workspace);
    op.Init(w, u, k, v, m, p ,q, o, mo, po, qo, tiling);
    op.Process();
}