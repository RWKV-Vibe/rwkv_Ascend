#include "kernel_operator.h"

using namespace AscendC;

constexpr uint32_t BUFFER_NUM = 1;

class KernelWKV7
{
public:
    __aicore__ inline KernelWKV7() {}
    __aicore__ inline void Init(uint32_t tileNum, uint32_t tileNumRemainLength, uint32_t totalHeads, uint32_t T,
                                uint32_t tileLength, uint32_t HEAD_SIZE, uint32_t HEAD_NUMS, bool hasRemainer,
                                GM_ADDR k, GM_ADDR v, GM_ADDR w, GM_ADDR r, GM_ADDR a, GM_ADDR b, 
                                GM_ADDR h0, GM_ADDR o,  GM_ADDR ht)
    {
        // k:[B, H, T, N]
        // v:[B, H, T, N]
        // w:[B, H, T, N]
        // r:[B, H, T, N]
        // u:[H, N]
        // o:[B, H, T, N]
        // h0:[B, H, N, N]
        // ht:[B, H, N, N]
        this->tileNum = tileNum;
        this->tileNumRemainLength = tileNumRemainLength;
        this->totalHeads = totalHeads;
        this->T = T;
        this->tileLength = tileLength;
        this->HEAD_SIZE = HEAD_SIZE;
        this->HEAD_NUMS = HEAD_NUMS;
        this->HEAD_ELEMENTS = this->HEAD_SIZE * this->HEAD_SIZE;
        this->scale = 0.5;
        this->hasRemainer = hasRemainer;
        uint32_t blockNum = GetBlockNum();
        uint32_t baseHeadsPerCore = totalHeads / blockNum; // 基础分配
        uint32_t remainerHeads = totalHeads % blockNum;   // 余数
        uint32_t currentBlock = GetBlockIdx();
        // 计算当前核心实际处理的head数量
        this->headPerCore = baseHeadsPerCore;
        if (currentBlock < remainerHeads)
        {
            this->headPerCore += 1; // 前面几个核多处理一个head
        }
        // 计算当前核心的数据偏移
        uint32_t headOffset = baseHeadsPerCore * currentBlock;
        if (currentBlock < remainerHeads)
        {
            headOffset += currentBlock;
        }
        else
        {
            headOffset += remainerHeads;
        }
        uint32_t uh_offset = headOffset % this->HEAD_NUMS;
        this->sizePerCore = this->headPerCore * T * this->HEAD_SIZE;
        kGm.SetGlobalBuffer((__gm__ float *)k + headOffset * T * this->HEAD_SIZE, this->sizePerCore);
        vGm.SetGlobalBuffer((__gm__ float *)v + headOffset * T * this->HEAD_SIZE, this->sizePerCore);
        wGm.SetGlobalBuffer((__gm__ float *)w + headOffset * T * this->HEAD_SIZE, this->sizePerCore);
        rGm.SetGlobalBuffer((__gm__ float *)r + headOffset * T * this->HEAD_SIZE, this->sizePerCore);
        oGm.SetGlobalBuffer((__gm__ float *)o + headOffset * T * this->HEAD_SIZE, this->sizePerCore);
        aGm.SetGlobalBuffer((__gm__ float *)a + headOffset * T * this->HEAD_SIZE, this->sizePerCore);
        bGm.SetGlobalBuffer((__gm__ float *)b + headOffset * T * this->HEAD_SIZE, this->sizePerCore);
        // uGm.SetGlobalBuffer((__gm__ float *)u + uh_offset * this->HEAD_SIZE, this->headPerCore * this->HEAD_SIZE);
        h0Gm.SetGlobalBuffer((__gm__ float *)h0 + headOffset * this->HEAD_ELEMENTS, this->headPerCore * this->HEAD_ELEMENTS);
        htGm.SetGlobalBuffer((__gm__ float *)ht + headOffset * this->HEAD_ELEMENTS, this->headPerCore * this->HEAD_ELEMENTS);
        // k,v,w,r,a,b,o每次搬运[tileLength, N]大小的tensor
        pipe.InitBuffer(inQueueK, BUFFER_NUM, this->tileLength * this->HEAD_SIZE * sizeof(float));
        pipe.InitBuffer(inQueueV, BUFFER_NUM, this->tileLength * this->HEAD_SIZE * sizeof(float));
        pipe.InitBuffer(inQueueW, BUFFER_NUM, this->tileLength * this->HEAD_SIZE * sizeof(float));
        pipe.InitBuffer(inQueueR, BUFFER_NUM, this->tileLength * this->HEAD_SIZE * sizeof(float));
        pipe.InitBuffer(inQueueAB, BUFFER_NUM, 2 * this->tileLength * this->HEAD_SIZE * sizeof(float));
        // pipe.InitBuffer(inQueueB, BUFFER_NUM, this->tileLength * this->HEAD_SIZE * sizeof(float));
        // pipe.InitBuffer(inQueueU, BUFFER_NUM, this->HEAD_SIZE * sizeof(float));
        // h0, ht每次搬运[HEAD_SIZE, HEAD_SIZE]大小的tensor
        pipe.InitBuffer(inQueueH, BUFFER_NUM, this->HEAD_ELEMENTS * sizeof(float));
        pipe.InitBuffer(outQueueH, BUFFER_NUM, this->HEAD_ELEMENTS * sizeof(float));
        // 其中 o 既是输入也是输出，所以既需要vecin的buffer也需要vecout的buffer
        pipe.InitBuffer(inQueueO, BUFFER_NUM, this->tileLength * this->HEAD_SIZE * sizeof(float));
        pipe.InitBuffer(outQueueO, BUFFER_NUM, this->tileLength * this->HEAD_SIZE * sizeof(float));
        // state及中间变量，每个中间变量大小为[N, N]
        pipe.InitBuffer(stateBuf, 3 * this->HEAD_ELEMENTS * sizeof(float));
        // 用于储存broadcast结果
        pipe.InitBuffer(broadBuf0, this->HEAD_ELEMENTS * sizeof(float));
        // pipe.InitBuffer(broadBuf1, this->HEAD_ELEMENTS * sizeof(float));
        // pipe.InitBuffer(broadBuf2, this->HEAD_ELEMENTS * sizeof(float));
        // 设置broadcast shape参数
        SetBroadShapes();
    }

    __aicore__ inline void Process()
    {
        LocalTensor<float> stateLocal = stateBuf.Get<float>();
        LocalTensor<float> broadLocal0 = broadBuf0.Get<float>();
        // LocalTensor<float> broadLocal1 = broadBuf1.Get<float>();
        // LocalTensor<float> broadLocal2 = broadBuf2.Get<float>();

        for (uint32_t h = 0; h < this->headPerCore; h++)
        {
            // copy tensor u[h,:]
            // CopyInU(h);
            // LocalTensor<float> uLocal = inQueueU.DeQue<float>();
            // broadcast u and store in broadLocal0:[1, N] to [N, N]
            // BroadCast<float, 2, 0>(broadLocal0, uLocal, broadDstShape, broadSrcShape);

            // 加载当前头的初始 h0 到 stateLocal[0]
            uint32_t h_offset = h * this->HEAD_ELEMENTS;
            CopyInH0(h, h_offset);
            LocalTensor<float> hLocal = inQueueH.DeQue<float>();
            // DataCopy(stateLocal[0], hLocal, this->HEAD_ELEMENTS);
            Adds(stateLocal[0], hLocal, 0.0f, this->HEAD_ELEMENTS);
            inQueueH.FreeTensor(hLocal);
            
            for (uint32_t tile = 0; tile < this->tileNum; tile++)
            {
                // copy tensor k,v,w,r,a,b,o[b, h, tile * tileLength:(tile+1)*tileLength, :]
                CopyInKVWRABO(h, tile, this->tileLength);
                LocalTensor<float> kLocal = inQueueK.DeQue<float>();
                LocalTensor<float> vLocal = inQueueV.DeQue<float>();
                LocalTensor<float> wLocal = inQueueW.DeQue<float>();
                LocalTensor<float> rLocal = inQueueR.DeQue<float>();
                LocalTensor<float> abLocal = inQueueAB.DeQue<float>();
                LocalTensor<float> oLocal = inQueueO.DeQue<float>();
                Compute(kLocal, vLocal, wLocal, rLocal, abLocal, oLocal, stateLocal, broadLocal0, h, 
                        tile, this->tileLength);
                CopyOutO(h, tile, this->tileLength);
            }

            // 处理余数
            if (this->hasRemainer)
            {
                CopyInKVWRABO(h, this->tileNum, this->tileNumRemainLength);
                LocalTensor<float> kLocal = inQueueK.DeQue<float>();
                LocalTensor<float> vLocal = inQueueV.DeQue<float>();
                LocalTensor<float> wLocal = inQueueW.DeQue<float>();
                LocalTensor<float> rLocal = inQueueR.DeQue<float>();
                LocalTensor<float> abLocal = inQueueAB.DeQue<float>();
                LocalTensor<float> oLocal = inQueueO.DeQue<float>();
                Compute(kLocal, vLocal, wLocal, rLocal, abLocal, oLocal, stateLocal, broadLocal0, h, 
                        this->tileNum, this->tileNumRemainLength);
                CopyOutO(h, this->tileNum, this->tileNumRemainLength);
            }

            // copy out stateLocal[0] to ht[b, h, :, :]
            LocalTensor<float> htOutLocal = outQueueH.AllocTensor<float>();
            DataCopy(htOutLocal, stateLocal[0], this->HEAD_ELEMENTS);
            // DumpTensor(htOutLocal, 11, this->HEAD_ELEMENTS);
            outQueueH.EnQue<float>(htOutLocal);      
            CopyOutHt(h, h_offset);
        }
    }

private:
    __aicore__ inline void SetBroadShapes()
    {
        // 除了v以外所有的broadcast shape: [1, N] -> [N, N]
        broadDstShape[0] = this->HEAD_SIZE;
        broadDstShape[1] = this->HEAD_SIZE;
        broadSrcShape[0] = 1;
        broadSrcShape[1] = this->HEAD_SIZE;
        // v的broadcast shape: [N, 1] -> [N, N]
        vDstShape[0] = this->HEAD_SIZE;
        vDstShape[1] = this->HEAD_SIZE;
        vSrcShape[0] = this->HEAD_SIZE;
        vSrcShape[1] = 1;
    }


    __aicore__ inline void CopyInKVWRABO(uint32_t progress_h, uint32_t progress_tile, uint32_t currentTileLength)
    {
        // copy k,v,w,r,a,b,o[b, h, tile*tileLength:(tile+1)*tileLength, :]
        
        uint32_t offset = progress_h * this->T * this->HEAD_SIZE + progress_tile * this->tileLength * this->HEAD_SIZE;
        LocalTensor<float> kLocal = inQueueK.AllocTensor<float>();
        LocalTensor<float> vLocal = inQueueV.AllocTensor<float>();
        LocalTensor<float> wLocal = inQueueW.AllocTensor<float>();
        LocalTensor<float> rLocal = inQueueR.AllocTensor<float>();
        LocalTensor<float> abLocal = inQueueAB.AllocTensor<float>();
        LocalTensor<float> oLocal = inQueueO.AllocTensor<float>();
        
        DataCopy(wLocal, wGm[offset], currentTileLength * this->HEAD_SIZE);
        DataCopy(abLocal[this->tileLength * this->HEAD_SIZE], bGm[offset], currentTileLength * this->HEAD_SIZE);
        // DataCopy(bLocal, bGm[offset], currentTileLength * this->HEAD_SIZE);
        DataCopy(oLocal, oGm[offset], currentTileLength * this->HEAD_SIZE);

        DataCopy(kLocal, kGm[offset], currentTileLength * this->HEAD_SIZE);
        DataCopy(vLocal, vGm[offset], currentTileLength * this->HEAD_SIZE);
        DataCopy(rLocal, rGm[offset], currentTileLength * this->HEAD_SIZE);
        DataCopy(abLocal, aGm[offset], currentTileLength * this->HEAD_SIZE);
        
        inQueueK.EnQue<float>(kLocal);
        inQueueV.EnQue<float>(vLocal);
        inQueueW.EnQue<float>(wLocal);
        inQueueR.EnQue<float>(rLocal);
        inQueueAB.EnQue<float>(abLocal);
        // inQueueB.EnQue<float>(bLocal);
        inQueueO.EnQue<float>(oLocal);
    }

    __aicore__ inline void CopyInH0(uint32_t progress_h, uint32_t offset)
    {
        // copy in h0[b, h, HEAD_SIZE, HEAD_SIZE]
        LocalTensor<float> hLocal = inQueueH.AllocTensor<float>();
        DataCopy(hLocal, h0Gm[offset], this->HEAD_ELEMENTS);
        inQueueH.EnQue<float>(hLocal);
    }

    __aicore__ inline void CopyOutO(uint32_t progress_h, uint32_t progress_tile, uint32_t currentTileLength)
    {
        // copy out o[b, h, tile*tileLength:(tile+1)*tileLength,:]
        uint32_t offset = progress_h * this->T * this->HEAD_SIZE + progress_tile * this->tileLength * this->HEAD_SIZE;
        LocalTensor<float> oOutLocal = outQueueO.DeQue<float>();
        DataCopy(oGm[offset], oOutLocal, currentTileLength * this->HEAD_SIZE);
        outQueueO.FreeTensor(oOutLocal);
    }

    __aicore__ inline void CopyOutHt(uint32_t progress_h, uint32_t offset)
    {
        LocalTensor<float> htOutLocal = outQueueH.DeQue<float>();
        DataCopy(htGm[offset], htOutLocal, this->HEAD_ELEMENTS); 
        outQueueH.FreeTensor(htOutLocal);       
    }

    __aicore__ inline void Compute(LocalTensor<float> kLocal, LocalTensor<float> vLocal, LocalTensor<float> wLocal, LocalTensor<float> rLocal, 
                                   LocalTensor<float> abLocal, LocalTensor<float> oLocal, LocalTensor<float> stateLocal, LocalTensor<float> broadLocal0,
                                   uint32_t progress_h, uint32_t progress_tile,
                                   uint32_t currentTileLength)
    {
        uint32_t offset0 = 0; // reserved for state vectors
        uint32_t offset1 = this->HEAD_ELEMENTS;
        uint32_t offset2 = this->HEAD_ELEMENTS * 2;

        Exp(wLocal, wLocal, this->HEAD_SIZE *currentTileLength);
        // PipeBarrier<PIPE_V>();
        LocalTensor<float> oOutLocal = outQueueO.AllocTensor<float>();
        for (uint32_t t = 0; t < currentTileLength; t++)
        {
            
            // compute astate = a * state, offset2
            Mul(stateLocal[offset2], abLocal[t * this->HEAD_SIZE], stateLocal[offset0], 64, 64, { 1, 1, 1, 8, 0, 8 });
            PipeBarrier<PIPE_V>();

            // compute sa = reduceSum(astate), shape: N, offset1
            // mask=N, repeatTimes=N, dstRepStride=1, srcBlkStride=1, srcRepStride=N*sizeof(float)/32=4
            WholeReduceSum(stateLocal[offset1], stateLocal[offset2], this->HEAD_SIZE, this->HEAD_SIZE, 1, 1, 8);
            
            // compute sab = sa * b, shape:N * N, offset2
            // broadcast sa from [N, 1] to [N, N]
            // BroadCast<float, 2, 1>(broadLocal0, stateLocal[offset1], broadDstShape, broadSrcShape);
            Brcb(broadLocal0, stateLocal[offset1], 8, {1,8});
            PipeBarrier<PIPE_V>();
            Brcb(stateLocal[offset1], broadLocal0, 64, {1,8});
            PipeBarrier<PIPE_V>();
            Mul(stateLocal[offset2], abLocal[this->tileLength * this->HEAD_SIZE + t * this->HEAD_SIZE], stateLocal[offset1], 64, 64, { 1, 1, 1, 8, 0, 8 });//this->HEAD_ELEMENTS);
            PipeBarrier<PIPE_V>();

            // compute kv = k.mT@v, offset1
            // broadcast v from [V, 1] to [V, V]
            // BroadCast<float, 2, 1>(broadLocal0, vLocal[t * this->HEAD_SIZE], vDstShape, vSrcShape);
            Brcb(stateLocal[offset1], vLocal[t * this->HEAD_SIZE], 8, {1,8});
            PipeBarrier<PIPE_V>();
            Brcb(broadLocal0, stateLocal[offset1], 64, {1,8});
            PipeBarrier<PIPE_V>();
            Mul(stateLocal[offset1], kLocal[t * this->HEAD_SIZE], broadLocal0, 64, 64, { 1, 1, 1, 8, 0, 8 });//this->HEAD_ELEMENTS);
            PipeBarrier<PIPE_V>();

            
            // compute state =  state * exp(-exp(w)), shape:N * N, broadLocal0
            // Exp(wLocal[t * this->HEAD_SIZE], wLocal[t * this->HEAD_SIZE], this->HEAD_SIZE);
            // PipeBarrier<PIPE_V>();
            Mul(broadLocal0, stateLocal[offset0], wLocal[t * this->HEAD_SIZE],  64, 64, { 1, 1, 1, 8, 8, 0});
            PipeBarrier<PIPE_V>();


            // compute state = state * w + kv + sab, shape:N * N, offset0
            Add(stateLocal[offset1], broadLocal0, stateLocal[offset1], this->HEAD_ELEMENTS);
            PipeBarrier<PIPE_V>();
            Add(stateLocal[offset0], stateLocal[offset1], stateLocal[offset2], this->HEAD_ELEMENTS);
            PipeBarrier<PIPE_V>();

            // compute out = state * r, shape:N * N, offset2
            // broadcast r from [1, N] to [N, N]
            Mul(stateLocal[offset2], rLocal[t * this->HEAD_SIZE], stateLocal[offset0], 64, 64, { 1, 1, 1, 8, 0, 8});
            PipeBarrier<PIPE_V>();
            
            // compute reduceSum(out), shape: N, oOutLocal
            // mask=N, repeatTimes=N, dstRepStride=1, srcBlkStride=1, srcRepStride=N*sizeof(float)/32=4
            WholeReduceSum(oOutLocal[t * this->HEAD_SIZE], stateLocal[offset2], this->HEAD_SIZE, this->HEAD_SIZE, 1, 1, 8);
        }
        
        // move o from vecin to vecout then free vecin o
        
        // DataCopy(oOutLocal, oLocal, this->tileLength * this->HEAD_SIZE);
        
        outQueueO.EnQue<float>(oOutLocal);

        inQueueO.FreeTensor(oLocal);

        // free k,v,w,r vecin for reuse
        inQueueK.FreeTensor(kLocal);
        inQueueV.FreeTensor(vLocal);
        inQueueW.FreeTensor(wLocal);
        inQueueR.FreeTensor(rLocal);
        inQueueAB.FreeTensor(abLocal);
        // inQueueB.FreeTensor(bLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueK, inQueueV, inQueueW, 
                inQueueR, inQueueAB, inQueueO, inQueueH;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueO, outQueueH;
    GlobalTensor<float> kGm, vGm, wGm, rGm, aGm, bGm, oGm, h0Gm, htGm;
    TBuf<QuePosition::VECCALC> stateBuf, broadBuf0; // broadBuf1, broadBuf2;
    uint32_t  tileLength, HEAD_NUMS, HEAD_SIZE, HEAD_ELEMENTS;
    uint32_t tileNum, tileNumRemainLength, baseHeadsPerCore, remainerHeads, totalHeads;
    uint32_t batchPerCore, sizePerCore, headPerCore, uSizePerCore;
    uint32_t T;
    float scale;
    bool hasRemainer;
    uint32_t broadDstShape[2], broadSrcShape[2];
    uint32_t vDstShape[2], vSrcShape[2];
};

// implementation of kernel function
extern "C" __global__ __aicore__ void wkv7(GM_ADDR k, GM_ADDR v, GM_ADDR w, GM_ADDR r, 
                                            GM_ADDR a, GM_ADDR b, GM_ADDR h0, GM_ADDR o, GM_ADDR ht, 
                                            GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);

    KernelWKV7 op;
    op.Init(tiling_data.tileNum, tiling_data.tileNumRemainLength, tiling_data.totalHeads, tiling_data.T,
            tiling_data.tileLength, tiling_data.HEAD_SIZE, tiling_data.HEAD_NUMS, tiling_data.hasRemainer,
            k, v, w, r, a, b, h0, o, ht);
    op.Process();
}