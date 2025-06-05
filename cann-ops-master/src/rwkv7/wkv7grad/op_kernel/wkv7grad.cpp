#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace matmul;
using namespace AscendC;

constexpr uint32_t BUFFER_NUM = 1;

template<typename aType, typename bType, typename cType> class KernelMatMulWKV7Grad {
    public:
        __aicore__ inline KernelMatMulWKV7Grad(){};
         __aicore__ inline void Init(GM_ADDR k, GM_ADDR v, GM_ADDR w, GM_ADDR r, GM_ADDR a, GM_ADDR b, GM_ADDR h, GM_ADDR o,
                                    GM_ADDR dk, GM_ADDR dv, GM_ADDR dw, GM_ADDR dr, GM_ADDR da, GM_ADDR db, GM_ADDR dh, 
                                    GM_ADDR workspace, const TCubeTiling &tiling, 
                                    uint32_t tileNum, uint32_t tileNumRemainLength, uint32_t totalHeads, uint32_t T,
                                    uint32_t tileLength, uint32_t HEAD_SIZE, uint32_t HEAD_NUMS, bool hasRemainer,
                                    TPipe *pipe);

        __aicore__ inline void Process();
        // __aicore__ inline void MatmulCompute();
        __aicore__ inline void SetBroadShapes();
        __aicore__ inline void CopyInKVWRABO(uint32_t progress_h, uint32_t progress_tile, uint32_t currentTileLength);
        __aicore__ inline void CopyInh(uint32_t progress_h, uint32_t progress_tile);
        __aicore__ inline void CopyOutDKVWRAB(uint32_t progress_h, uint32_t progress_tile, uint32_t currentTileLength);
        __aicore__ inline void CopyOutdh(uint32_t progress_h, uint32_t offset);
        __aicore__ inline void Compute(LocalTensor<float> kLocal, LocalTensor<float> vLocal, LocalTensor<float> wLocal, LocalTensor<float> rLocal, 
                                        LocalTensor<float> abLocal, LocalTensor<float> oLocal, LocalTensor<float> stateLocal, LocalTensor<float> dstateLocal,
                                        LocalTensor<float> broadLocal0, LocalTensor<float> broadLocal1, 
                                        uint32_t progress_h, uint32_t progress_tile,
                                        uint32_t currentTileLength);

        // WKV7Grad 
        TQue<QuePosition::VECIN, BUFFER_NUM> inQueueK, inQueueV, inQueueW, 
                    inQueueR, inQueueAB, inQueueO, inQueueH;
        TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueDK, outQueueDV, outQueueDW, 
                    outQueueDQ, outQueueDAB, outQueueDH;
        GlobalTensor<float> kGm, vGm, wGm, rGm, aGm, bGm, hGm, oGm;  //input
        GlobalTensor<float> dkGm, dvGm, dwGm, drGm, daGm, dbGm, dhGm;  //output
        TBuf<QuePosition::VECCALC> stateBuf, dstateBuf, broadBuf0, broadBuf1;  // broadBuf1, broadBuf2;
        TBuf<QuePosition::VECCALC>  vectorBuf0,  vectorBuf1, vectorBufexp; // vectorBufq, vectorBufk, vectorBufv, vectorBufw, vectorBufa, vectorBufb;
        uint32_t  tileLength, HEAD_NUMS, HEAD_SIZE, HEAD_ELEMENTS;
        uint32_t tileNum, tileNumRemainLength, baseHeadsPerCore, remainerHeads, totalHeads;
        uint32_t batchPerCore, sizePerCore, headPerCore, uSizePerCore;
        uint32_t T;
        float scale;
        bool hasRemainer;
        uint32_t broadDstShape[2], broadSrcShape[2];
        uint32_t vDstShape[2], vSrcShape[2];
        
        // matmul 
        Matmul<MatmulType<TPosition::VECOUT, CubeFormat::ND, float>, 
                MatmulType<TPosition::VECOUT, CubeFormat::ND, float>,
                MatmulType<TPosition::VECIN, CubeFormat::ND, float>>
            matmulObj;
    
        TCubeTiling tiling;
    
        // int tailM;
        // int tailN;
}; 

template <typename aType, typename bType, typename cType>
__aicore__ inline void
KernelMatMulWKV7Grad<aType, bType, cType>::Init(GM_ADDR k, GM_ADDR v, GM_ADDR w, GM_ADDR r, GM_ADDR a, GM_ADDR b, GM_ADDR h, GM_ADDR o,
                                                GM_ADDR dk, GM_ADDR dv, GM_ADDR dw, GM_ADDR dr, GM_ADDR da, GM_ADDR db, GM_ADDR dh, 
                                                GM_ADDR workspace, const TCubeTiling &tiling, 
                                                uint32_t tileNum, uint32_t tileNumRemainLength, uint32_t totalHeads, uint32_t T,
                                                uint32_t tileLength, uint32_t HEAD_SIZE, uint32_t HEAD_NUMS, bool hasRemainer,
                                                TPipe *pipe)
{
    // k:[B, H, T, N]
    // v:[B, H, T, N]
    // w:[B, H, T, N]
    // r:[B, H, T, N]
    // u:[H, N]
    // o:[B, H, T, N]
    // h:[B, H, N, N]
    // dh:[B, H, N, N]
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
    // input k, v, w, r, a, b, o : shape[B, H, T, N]
    kGm.SetGlobalBuffer((__gm__ float *)k + headOffset * T * this->HEAD_SIZE, this->sizePerCore);
    vGm.SetGlobalBuffer((__gm__ float *)v + headOffset * T * this->HEAD_SIZE, this->sizePerCore);
    wGm.SetGlobalBuffer((__gm__ float *)w + headOffset * T * this->HEAD_SIZE, this->sizePerCore);
    rGm.SetGlobalBuffer((__gm__ float *)r + headOffset * T * this->HEAD_SIZE, this->sizePerCore);
    aGm.SetGlobalBuffer((__gm__ float *)a + headOffset * T * this->HEAD_SIZE, this->sizePerCore);
    bGm.SetGlobalBuffer((__gm__ float *)b + headOffset * T * this->HEAD_SIZE, this->sizePerCore);
    oGm.SetGlobalBuffer((__gm__ float *)o + headOffset * T * this->HEAD_SIZE, this->sizePerCore);
    // input states : shape [B, H, 5, N, N]
    hGm.SetGlobalBuffer((__gm__ float *)h + (this->tileLength+1) * headOffset * this->HEAD_ELEMENTS, (this->tileLength+1) * this->headPerCore * this->HEAD_ELEMENTS);

    // output dk, dv, dw, dr, da, db : shape[B, H, T, N] ;   dh : shape [B, H, N, N]
    dkGm.SetGlobalBuffer((__gm__ float *)dk + headOffset * T * this->HEAD_SIZE, this->sizePerCore);
    dvGm.SetGlobalBuffer((__gm__ float *)dv + headOffset * T * this->HEAD_SIZE, this->sizePerCore);
    dwGm.SetGlobalBuffer((__gm__ float *)dw + headOffset * T * this->HEAD_SIZE, this->sizePerCore);
    drGm.SetGlobalBuffer((__gm__ float *)dr + headOffset * T * this->HEAD_SIZE, this->sizePerCore);
    daGm.SetGlobalBuffer((__gm__ float *)da + headOffset * T * this->HEAD_SIZE, this->sizePerCore);
    dbGm.SetGlobalBuffer((__gm__ float *)db + headOffset * T * this->HEAD_SIZE, this->sizePerCore);
    dhGm.SetGlobalBuffer((__gm__ float *)dh + headOffset * this->HEAD_ELEMENTS, this->headPerCore * this->HEAD_ELEMENTS);
    // k,v,w,r,a&b,o每次搬入[tileLength, N]大小的tensor
    pipe->InitBuffer(inQueueK, BUFFER_NUM, this->tileLength * this->HEAD_SIZE * sizeof(float)); 
    pipe->InitBuffer(inQueueV, BUFFER_NUM, this->tileLength * this->HEAD_SIZE * sizeof(float)); 
    pipe->InitBuffer(inQueueW, BUFFER_NUM, this->tileLength * this->HEAD_SIZE * sizeof(float));  
    pipe->InitBuffer(inQueueR, BUFFER_NUM, this->tileLength * this->HEAD_SIZE * sizeof(float));
    pipe->InitBuffer(inQueueAB, BUFFER_NUM, 2 * this->tileLength * this->HEAD_SIZE * sizeof(float)); 
    pipe->InitBuffer(inQueueO, BUFFER_NUM, this->tileLength * this->HEAD_SIZE * sizeof(float));
    // h 每次搬入[tileLength, HEAD_SIZE, HEAD_SIZE]大小的tensor
    pipe->InitBuffer(inQueueH, BUFFER_NUM, (this->tileLength+1) * this->HEAD_ELEMENTS * sizeof(float));

    // dk,dv,dw,dr,da,db,dh每次搬出[tileLength, N]大小的tensor
    pipe->InitBuffer(outQueueDK, BUFFER_NUM, this->tileLength * this->HEAD_SIZE * sizeof(float)); 
    pipe->InitBuffer(outQueueDV, BUFFER_NUM, this->tileLength * this->HEAD_SIZE * sizeof(float)); 
    pipe->InitBuffer(outQueueDW, BUFFER_NUM, this->tileLength * this->HEAD_SIZE * sizeof(float)); 
    pipe->InitBuffer(outQueueDQ, BUFFER_NUM, this->tileLength * this->HEAD_SIZE * sizeof(float)); 
    pipe->InitBuffer(outQueueDAB, BUFFER_NUM, 2 * this->tileLength * this->HEAD_SIZE * sizeof(float)); 
    // dh每次搬出[HEAD_SIZE, HEAD_SIZE]大小的tensor
    pipe->InitBuffer(outQueueDH, BUFFER_NUM, this->HEAD_ELEMENTS * sizeof(float));

    // state及中间变量，每个中间变量大小为[N, N]
    pipe->InitBuffer(stateBuf, 3 * this->HEAD_ELEMENTS * sizeof(float));
    // dstate及中间变量，每个中间变量大小为[N, N]
    pipe->InitBuffer(dstateBuf, this->HEAD_ELEMENTS * sizeof(float));
    // q, k, v, w, a, b pipe
    // pipe->InitBuffer(vectorBufq, this->tileLength * this->HEAD_SIZE * sizeof(float));
    // pipe->InitBuffer(vectorBufk, this->tileLength * this->HEAD_SIZE * sizeof(float));
    // pipe->InitBuffer(vectorBufv, this->tileLength * this->HEAD_SIZE * sizeof(float));
    // pipe->InitBuffer(vectorBufw, this->tileLength * this->HEAD_SIZE * sizeof(float));
    // pipe->InitBuffer(vectorBufa, this->tileLength * this->HEAD_SIZE * sizeof(float));
    // pipe->InitBuffer(vectorBufb, this->tileLength * this->HEAD_SIZE * sizeof(float));
    // 用于储存broadcast结果
    pipe->InitBuffer(vectorBufexp, this->tileLength * this->HEAD_SIZE * sizeof(float));
    pipe->InitBuffer(vectorBuf0, this->HEAD_SIZE * sizeof(float));
    pipe->InitBuffer(vectorBuf1, this->HEAD_SIZE * sizeof(float));
    pipe->InitBuffer(broadBuf0, this->HEAD_ELEMENTS * sizeof(float));
    pipe->InitBuffer(broadBuf1, this->HEAD_ELEMENTS * sizeof(float));
    // pipe->InitBuffer(broadBuf2, this->HEAD_ELEMENTS * sizeof(float));
    // 设置broadcast shape参数
    SetBroadShapes();

    // matmul

    this->tiling = tiling;
    // A.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(x1), tiling.M * tiling.Ka);
    // B.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(x2), tiling.Kb * tiling.N);
    // C.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(y), tiling.M * tiling.N);

    if (GetSysWorkSpacePtr() == nullptr) {
        // printf("error:GetSysWorkSpacePtr() == nullptr\r\n");
        return;
    }
}

/**
  * @brief  Main process of matmul calculation
  * @param  pipe: Global memory and sync management TPipe object.
  * @retval None
  */
template <typename aType, typename bType, typename cType>
__aicore__ inline void KernelMatMulWKV7Grad<aType, bType, cType>::Process()
{
        // size 64*64=4096, total: 4096 *3 = 12288
        LocalTensor<float> broadLocal0 = broadBuf0.Get<float>();
        LocalTensor<float> broadLocal1 = broadBuf1.Get<float>();
        // LocalTensor<float> broadLocal2 = broadBuf2.Get<float>();
        // size 64*64*3= 12288
        LocalTensor<float> stateLocal = stateBuf.Get<float>();
        LocalTensor<float> dstateLocal = dstateBuf.Get<float>();
        for (uint32_t h = 0; h < this->headPerCore; h++)
        {
            uint32_t h_offset = h * this->HEAD_ELEMENTS;
            for (int32_t tile = this->tileNum - 1; tile >= 0; tile--)
            {   
                printf("tile: %d \n", tile);
                // copy states [b, h, 16*tile*HEAD_ELEMENTS:16*(tile-1)*HEAD_ELEMENTS, :]
                // 加载当前头的初始 h 到 stateLocal[0]
                CopyInh(h, tile);
                
                
                // DataCopy(stateLocal[0], hLocal[0], this->tileLength * this->HEAD_ELEMENTS);
                // DumpTensor(stateLocal[0], 0, 4096); 
                printf("hello world \n ");
                // copy tensor k,v,w,r,a,b,o[b, h, tile * tileLength:(tile-1)*tileLength, :]
                CopyInKVWRABO(h, tile, this->tileLength);
                LocalTensor<float> kLocal = inQueueK.DeQue<float>();
                LocalTensor<float> vLocal = inQueueV.DeQue<float>();
                LocalTensor<float> wLocal = inQueueW.DeQue<float>();
                LocalTensor<float> rLocal = inQueueR.DeQue<float>();
                LocalTensor<float> abLocal = inQueueAB.DeQue<float>();
                LocalTensor<float> oLocal = inQueueO.DeQue<float>();
                Compute(kLocal, vLocal, wLocal, rLocal, abLocal, oLocal, stateLocal, dstateLocal, broadLocal0, broadLocal1,
                        h, tile, this->tileLength);
                CopyOutDKVWRAB(h, tile, this->tileLength);

                
            }
            
            // // 处理余数
            // if (this->hasRemainer)
            // {
            //     // copy states [b, h, 16*tile*HEAD_ELEMENTS:16*(tile-1)*HEAD_ELEMENTS, :]
            //     // 加载当前头的初始 h 到 stateLocal[0]
            //     CopyInh(h, tile);
            //     LocalTensor<float> hLocal = inQueueH.DeQue<float>();
            //     DataCopy(stateLocal[0], hLocal[(this->tileLength -1) * this->HEAD_ELEMENTS], this->HEAD_ELEMENTS);

            //     CopyInKVWRABO(h, this->tileNum, this->tileNumRemainLength);
            //     LocalTensor<float> kLocal = inQueueK.DeQue<float>();
            //     LocalTensor<float> vLocal = inQueueV.DeQue<float>();
            //     LocalTensor<float> wLocal = inQueueW.DeQue<float>();
            //     LocalTensor<float> rLocal = inQueueR.DeQue<float>();
            //     LocalTensor<float> abLocal = inQueueAB.DeQue<float>();
            //     LocalTensor<float> oLocal = inQueueO.DeQue<float>();
            //     Compute(kLocal, vLocal, wLocal, rLocal, abLocal, hLocal, oLocal, stateLocal, broadLocal0, broadLocal1, broadLocal2, h, 
            //             this->tileNum, this->tileNumRemainLength);
            //     CopyOutDKVWRAB(h, this->tileNum, this->tileNumRemainLength);
            // }

            // copy out stateLocal[0] to dh[b, h, :, :]
            LocalTensor<float> dhOutLocal = outQueueDH.AllocTensor<float>();
            DataCopy(dhOutLocal, dstateLocal, this->HEAD_ELEMENTS);
            // DumpTensor(dhOutLocal, 11, this->HEAD_ELEMENTS);
            outQueueDH.EnQue<float>(dhOutLocal);      
            CopyOutdh(h, h_offset);
        }
}

template <typename aType, typename bType, typename cType>
__aicore__ inline void KernelMatMulWKV7Grad<aType, bType, cType>::SetBroadShapes()
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
template <typename aType, typename bType, typename cType>
__aicore__ inline void KernelMatMulWKV7Grad<aType, bType, cType>::CopyInKVWRABO(uint32_t progress_h, uint32_t progress_tile, uint32_t currentTileLength)
{
    // copy k,v,w,r,a,b,o[b, h, tile*tileLength:(tile+1)*tileLength, :]
        
    uint32_t offset = progress_h * this->T * this->HEAD_SIZE + progress_tile * this->tileLength * this->HEAD_SIZE;
    LocalTensor<float> kLocal = inQueueK.AllocTensor<float>();
    LocalTensor<float> vLocal = inQueueV.AllocTensor<float>();
    LocalTensor<float> wLocal = inQueueW.AllocTensor<float>();
    LocalTensor<float> rLocal = inQueueR.AllocTensor<float>();
    LocalTensor<float> abLocal = inQueueAB.AllocTensor<float>();
    LocalTensor<float> oLocal = inQueueO.AllocTensor<float>();
    DataCopy(kLocal, kGm[offset], currentTileLength * this->HEAD_SIZE);
    DataCopy(vLocal, vGm[offset], currentTileLength * this->HEAD_SIZE);
    DataCopy(wLocal, wGm[offset], currentTileLength * this->HEAD_SIZE);
    DataCopy(rLocal, rGm[offset], currentTileLength * this->HEAD_SIZE);
    DataCopy(abLocal, aGm[offset], currentTileLength * this->HEAD_SIZE);
    DataCopy(abLocal[this->tileLength * this->HEAD_SIZE], bGm[offset], currentTileLength * this->HEAD_SIZE);
    DataCopy(oLocal, oGm[offset], currentTileLength * this->HEAD_SIZE);
    inQueueK.EnQue<float>(kLocal);
    inQueueV.EnQue<float>(vLocal);
    inQueueW.EnQue<float>(wLocal);
    inQueueR.EnQue<float>(rLocal);
    inQueueAB.EnQue<float>(abLocal);
    inQueueO.EnQue<float>(oLocal);
}

template <typename aType, typename bType, typename cType>
__aicore__ inline void KernelMatMulWKV7Grad<aType, bType, cType>::CopyInh(uint32_t progress_h, uint32_t progress_tile)
{
    // copy in h[b, h, HEAD_SIZE, HEAD_SIZE]
    uint32_t offset = progress_h * this->tileNum * this->HEAD_ELEMENTS + progress_tile * (this->tileLength+1) * this->HEAD_ELEMENTS;
    LocalTensor<float> hLocal = inQueueH.AllocTensor<float>();
    DataCopy(hLocal, hGm[offset], (this->tileLength+1) * this->HEAD_ELEMENTS);
    inQueueH.EnQue<float>(hLocal);
}

template <typename aType, typename bType, typename cType>
__aicore__ inline void KernelMatMulWKV7Grad<aType, bType, cType>::CopyOutDKVWRAB(uint32_t progress_h, uint32_t progress_tile, uint32_t currentTileLength)
{
    // copy out o[b, h, tile*tileLength:(tile+1)*tileLength,:]
    uint32_t offset = progress_h * this->T * this->HEAD_SIZE + progress_tile * this->tileLength * this->HEAD_SIZE;
    LocalTensor<float> dkOutLocal = outQueueDK.DeQue<float>();
    LocalTensor<float> dvOutLocal = outQueueDV.DeQue<float>();
    LocalTensor<float> dwOutLocal = outQueueDW.DeQue<float>();
    LocalTensor<float> drOutLocal = outQueueDQ.DeQue<float>();
    LocalTensor<float> dabOutLocal = outQueueDAB.DeQue<float>();

    DataCopy(dkGm[offset], dkOutLocal, currentTileLength * this->HEAD_SIZE);
    DataCopy(dvGm[offset], dvOutLocal, currentTileLength * this->HEAD_SIZE);
    DataCopy(dwGm[offset], dwOutLocal, currentTileLength * this->HEAD_SIZE);
    DataCopy(drGm[offset], drOutLocal, currentTileLength * this->HEAD_SIZE);
    DataCopy(daGm[offset], dabOutLocal, currentTileLength * this->HEAD_SIZE);
    DataCopy(dbGm[offset], dabOutLocal[this->tileLength * this->HEAD_SIZE], currentTileLength * this->HEAD_SIZE);
    outQueueDK.FreeTensor(dkOutLocal);
    outQueueDV.FreeTensor(dvOutLocal);
    outQueueDW.FreeTensor(dwOutLocal);
    outQueueDQ.FreeTensor(drOutLocal);
    outQueueDAB.FreeTensor(dabOutLocal);
}

template <typename aType, typename bType, typename cType>
__aicore__ inline void KernelMatMulWKV7Grad<aType, bType, cType>::CopyOutdh(uint32_t progress_h, uint32_t offset)
{
    LocalTensor<float> dhOutLocal = outQueueDH.DeQue<float>();
    DataCopy(dhGm[offset], dhOutLocal, this->HEAD_ELEMENTS); 
    outQueueDH.FreeTensor(dhOutLocal);   
}

template <typename aType, typename bType, typename cType>
__aicore__ inline void KernelMatMulWKV7Grad<aType, bType, cType>::Compute(LocalTensor<float> kLocal, LocalTensor<float> vLocal, LocalTensor<float> wLocal, LocalTensor<float> rLocal, 
                                                                                    LocalTensor<float> abLocal, LocalTensor<float> oLocal,
                                                                                    LocalTensor<float> stateLocal, LocalTensor<float> dstateLocal,LocalTensor<float> broadLocal0, LocalTensor<float> broadLocal1, 
                                                                                    uint32_t progress_h, uint32_t progress_tile,
                                                                                    uint32_t currentTileLength)
{
    LocalTensor<float> dqOutLocal = outQueueDQ.AllocTensor<float>();
    LocalTensor<float> dkOutLocal = outQueueDK.AllocTensor<float>();
    LocalTensor<float> dvOutLocal = outQueueDV.AllocTensor<float>();
    LocalTensor<float> dwOutLocal = outQueueDW.AllocTensor<float>();
    LocalTensor<float> dabOutLocal = outQueueDAB.AllocTensor<float>();

    // size 64*8 = 512
    // LocalTensor<float> vectorLocaldq = vectorBufq.Get<float>();
    // LocalTensor<float> vectorLocaldk = vectorBufk.Get<float>();
    // LocalTensor<float> vectorLocaldv = vectorBufv.Get<float>();
    // LocalTensor<float> vectorLocaldw = vectorBufw.Get<float>();
    // LocalTensor<float> vectorLocalda = vectorBufb.Get<float>();
    LocalTensor<float> vectorLocalexp = vectorBufexp.Get<float>();
    LocalTensor<float> vectorLocal0 = vectorBuf0.Get<float>();
    LocalTensor<float> vectorLocal1 = vectorBuf1.Get<float>();

    uint32_t offset0 = 0; // reserved for state vectors
    uint32_t offset1 = this->HEAD_ELEMENTS;
    uint32_t offset2 = this->HEAD_ELEMENTS * 2;

    LocalTensor<float> hLocal = inQueueH.DeQue<float>();
    DataCopy(stateLocal[0], hLocal[this->tileLength * this->HEAD_ELEMENTS], this->HEAD_ELEMENTS);
    // DumpTensor(stateLocal[offset0], 3, 64); 
    // DumpTensor(abLocal, 6, 4096); 
    // DumpTensor(abLocal[currentTileLength * this->HEAD_SIZE], 7, 4096); 
    Exp(vectorLocalexp, wLocal, this->HEAD_SIZE *currentTileLength);
    Muls(wLocal, vectorLocalexp, (float)-1.0, this->HEAD_SIZE *currentTileLength);
    Exp(wLocal, wLocal, this->HEAD_SIZE *currentTileLength);
    

    for (int32_t t = currentTileLength - 1 ; t >= 0; t--)
    {
        // dq
        matmulObj.SetTensorA(oLocal[t * this->HEAD_SIZE]);
        matmulObj.SetTensorB(stateLocal[offset0]);
        matmulObj.IterateAll<true>(vectorLocal0);
        matmulObj.End();

        Add(dqOutLocal[t * this->HEAD_SIZE], dqOutLocal[t * this->HEAD_SIZE], vectorLocal0, this->HEAD_SIZE);
        
        // dstate_from_out
        Brcb(broadLocal0, oLocal[t * this->HEAD_SIZE], 8, {1,8});
        Brcb(stateLocal[offset1], broadLocal0, 64, {1,8});
        Mul(stateLocal[offset2], rLocal[t * this->HEAD_SIZE], stateLocal[offset1], 64, 64, { 1, 1, 1, 8, 0, 8 });
        // dstate_curr
        Add(stateLocal[offset1], dstateLocal, stateLocal[offset2], this->HEAD_ELEMENTS);
        DumpTensor(stateLocal[offset1], 11, 4096);
       
        // bwd_state
        DataCopy(stateLocal[offset0], hLocal[t * this->HEAD_ELEMENTS], this->HEAD_ELEMENTS);

        // dk
        matmulObj.SetTensorA(vLocal[t * this->HEAD_SIZE]);
        matmulObj.SetTensorB(stateLocal[offset1]);
        matmulObj.IterateAll<true>(dkOutLocal[t * this->HEAD_SIZE]);
        matmulObj.End();
        DumpTensor(dkOutLocal[t * this->HEAD_SIZE], 3, 64); 
        // dv
        Mul(broadLocal0, kLocal[t * this->HEAD_SIZE], stateLocal[offset1], 64, 64, { 1, 1, 1, 8, 0, 8 });
        WholeReduceSum(dvOutLocal[t * this->HEAD_SIZE], broadLocal0, this->HEAD_SIZE, this->HEAD_SIZE, 1, 1, this->HEAD_SIZE * sizeof(float) / 32);
        DumpTensor(dvOutLocal[t * this->HEAD_SIZE], 4, 64); 

        // sa
        Mul(broadLocal0, abLocal[t * this->HEAD_SIZE], stateLocal[offset0], 64, 64, { 1, 1, 1, 8, 0, 8 });
        WholeReduceSum(vectorLocal1, broadLocal0, this->HEAD_SIZE, this->HEAD_SIZE, 1, 1, this->HEAD_SIZE * sizeof(float) / 32);
        DumpTensor(vectorLocal1, 5, 64); 

        // db
        matmulObj.SetTensorA(vectorLocal1);
        matmulObj.SetTensorB(stateLocal[offset1]);
        matmulObj.IterateAll<true>(dabOutLocal[(this->tileLength+t) * this->HEAD_SIZE]);
        matmulObj.End();
        DumpTensor(dabOutLocal[(this->tileLength+t) * this->HEAD_SIZE], 6, 64); 

        // dsa
        Mul(broadLocal0, abLocal[(this->tileLength+t) * this->HEAD_SIZE], stateLocal[offset1], 64, 64, { 1, 1, 1, 8, 0, 8 });
        WholeReduceSum(vectorLocal1, broadLocal0, this->HEAD_SIZE, this->HEAD_SIZE, 1, 1, this->HEAD_SIZE * sizeof(float) / 32);
        DumpTensor(vectorLocal1, 7, 64); 

        // da
        matmulObj.SetTensorA(vectorLocal1);
        matmulObj.SetTensorB(stateLocal[offset0]);
        matmulObj.IterateAll<true>(dabOutLocal[t * this->HEAD_SIZE]);
        matmulObj.End();
        DumpTensor(dabOutLocal[t * this->HEAD_SIZE], 8, 64); 

        // dstate_from_sa
        Brcb(broadLocal0, vectorLocal1, 8, {1,8});
        Brcb(stateLocal[offset2], broadLocal0, 64, {1,8});
        Mul(broadLocal0, abLocal[t * this->HEAD_SIZE], stateLocal[offset2], 64, 64, { 1, 1, 1, 8, 0, 8 });
        DumpTensor(broadLocal0, 9, 4096); 

        // dstate_from_decay
        Mul(stateLocal[offset2], wLocal[t * this->HEAD_SIZE], stateLocal[offset1], 64, 64, { 1, 1, 1, 8, 0, 8 });
        DumpTensor(stateLocal[offset2], 10, 4096); 

        // dstate = dstate_from_sa + dstate_from_decay
        Add(dstateLocal, broadLocal0, stateLocal[offset2], this->HEAD_ELEMENTS);
        DumpTensor(dstateLocal, 66, 64); 

        // dw
        Mul(broadLocal0, stateLocal[offset1], stateLocal[offset0], this->HEAD_ELEMENTS);
        Add(vectorLocal1, broadLocal0, vectorLocal1, 64, 64, { 1, 1, 1, 0, 8, 0});
        Muls(vectorLocal1, vectorLocal1, (float)-1.0, this->HEAD_SIZE);
        // DumpTensor(vectorLocalexp[t * this->HEAD_SIZE], 98, 64);
        Mul(vectorLocal0, wLocal[t * this->HEAD_SIZE], vectorLocal1, this->HEAD_SIZE);
        Mul(dwOutLocal[t * this->HEAD_SIZE], vectorLocal0, vectorLocalexp[t * this->HEAD_SIZE], this->HEAD_SIZE);
        // DumpTensor(dwOutLocal[t * this->HEAD_SIZE], 99, 64);
        
    }
    // outque EnQue
    outQueueDQ.EnQue<float>(dqOutLocal);
    outQueueDK.EnQue<float>(dkOutLocal);
    outQueueDV.EnQue<float>(dvOutLocal);
    outQueueDW.EnQue<float>(dwOutLocal);
    outQueueDAB.EnQue<float>(dabOutLocal);

    
    // inque free
    inQueueH.FreeTensor(hLocal);
    inQueueO.FreeTensor(oLocal);
    inQueueK.FreeTensor(kLocal);
    inQueueV.FreeTensor(vLocal);
    inQueueW.FreeTensor(wLocal);
    inQueueR.FreeTensor(rLocal);
    inQueueAB.FreeTensor(abLocal);
}

extern "C" __global__ __aicore__ void wkv7grad(GM_ADDR k, GM_ADDR v, GM_ADDR w, GM_ADDR r, 
                                                GM_ADDR a, GM_ADDR b, GM_ADDR h, GM_ADDR o,
                                                GM_ADDR dk, GM_ADDR dv, GM_ADDR dw, GM_ADDR dr,
                                                GM_ADDR da, GM_ADDR db, GM_ADDR dh,
                                                GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelMatMulWKV7Grad<float, float, float> op;
    // TODO: user kernel impl
    // KernelWKV7Grad op;
    TPipe pipe;
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), op.matmulObj, &tiling_data.cubeTilingData); // Initialize the matmul object.

    op.Init(k, v, w, r, a, b, h, o, dk, dv, dw, dr, da, db, dh, workspace, tiling_data.cubeTilingData,
        tiling_data.tileNum, tiling_data.tileNumRemainLength, tiling_data.totalHeads, tiling_data.T,
        tiling_data.tileLength, tiling_data.HEAD_SIZE, tiling_data.HEAD_NUMS, tiling_data.hasRemainer, &pipe);
    op.Process();
}