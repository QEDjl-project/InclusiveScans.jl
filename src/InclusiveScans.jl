module InclusiveScans

using CUDA
using CUDA: i32

const BLOCK_SIZE::Int32 = 1024
const NUM_BANKS::Int32 = 32

@inline function _conflict_free_access(n::TIdx) where {TIdx<:Integer}
    return n + (n ÷ TIdx(NUM_BANKS))
end

function _scanBlockKernel!(
    g_odata::CuDeviceVector{T},
    g_idata::CuDeviceVector{T},
    blockSums::Union{CuDeviceVector{T},Nothing},
    n::TIdx,
) where {T,TIdx<:Integer}
    temp = CuDynamicSharedArray(T, _conflict_free_access(blockDim().x * TIdx(2)))

    tx::TIdx = threadIdx().x - one(TIdx)
    bx::TIdx = blockIdx().x - one(TIdx)

    start::TIdx = bx * (blockDim().x * TIdx(2))

    i1::TIdx = start + tx * TIdx(2)
    i2::TIdx = i1 + one(TIdx)

    if tx >= zero(TIdx) && TIdx(2) * tx < TIdx(2) * blockDim().x
        if i1 < n
            @inbounds temp[_conflict_free_access(TIdx(2) * tx + one(TIdx))] =
                g_idata[i1+one(TIdx)]
        else
            @inbounds temp[_conflict_free_access(TIdx(2) * tx + one(TIdx))] = zero(T)
        end
        if i2 < n
            @inbounds temp[_conflict_free_access(TIdx(2) * tx + TIdx(2))] =
                g_idata[i2+one(TIdx)]
        else
            @inbounds temp[_conflict_free_access(TIdx(2) * tx + TIdx(2))] = zero(T)
        end
    end
    sync_threads()

    # Up-sweep (reduce)
    offset::TIdx = one(TIdx)
    d::TIdx = blockDim().x
    while d > zero(TIdx)
        sync_threads()
        if tx < d
            ai::TIdx = offset * (TIdx(2) * tx + TIdx(1)) - one(TIdx)
            bi::TIdx = offset * (TIdx(2) * tx + TIdx(2)) - one(TIdx)
            @inbounds temp[_conflict_free_access(bi + one(TIdx))] +=
                temp[_conflict_free_access(ai + one(TIdx))]
        end
        offset <<= one(TIdx)
        d >>= one(TIdx)
    end

    sync_threads()

    # Save sum of block -> blockSums[bx]
    if tx == zero(TIdx)
        if !isnothing(blockSums)
            @inbounds blockSums[bx+one(TIdx)] =
                temp[_conflict_free_access(TIdx(2) * blockDim().x)]
        end
        @inbounds temp[_conflict_free_access(TIdx(2) * blockDim().x)] = zero(T)
    end
    sync_threads()

    # Down-sweep
    d = one(TIdx)
    while d < TIdx(2) * blockDim().x
        offset >>= one(TIdx)
        sync_threads()
        if tx < d
            ai = offset * (TIdx(2) * tx + one(TIdx)) - one(TIdx)
            bi = offset * (TIdx(2) * tx + TIdx(2)) - one(TIdx)
            @inbounds t = temp[_conflict_free_access(ai + one(TIdx))]
            @inbounds temp[_conflict_free_access(ai + one(TIdx))] =
                temp[_conflict_free_access(bi + one(TIdx))]
            @inbounds temp[_conflict_free_access(bi + one(TIdx))] += t
        end
        d <<= one(TIdx)
    end
    sync_threads()

    if tx >= zero(TIdx) && TIdx(2) * tx < TIdx(2) * blockDim().x
        if i1 < n
            @inbounds g_odata[i1+one(TIdx)] =
                temp[_conflict_free_access(TIdx(2) * tx + one(TIdx))]
        end
        if i2 < n
            @inbounds g_odata[i2+one(TIdx)] =
                temp[_conflict_free_access(TIdx(2) * tx + one(TIdx) + one(TIdx))]
        end
    end

    return nothing
end

# Add prefix sum from previous blocks (d_increments[blockIdx.x]) to all elements processing by this block
function _addIncrementsKernel!(
    g_odata::CuDeviceVector{T},
    incr::CuDeviceVector{T},
    n::TIdx,
) where {T,TIdx<:Integer}
    tx::TIdx = threadIdx().x - one(TIdx)
    bx::TIdx = blockIdx().x - one(TIdx)

    start::TIdx = bx * (blockDim().x * TIdx(2))
    i::TIdx = start + tx * TIdx(2)

    if i < n
        @inbounds g_odata[i+one(TIdx)] += incr[bx+one(TIdx)]
        if i + one(TIdx) < n
            @inbounds g_odata[i+TIdx(2)] += incr[bx+one(TIdx)]
        end
    end
    return nothing
end

function largeArrayScanExclusive!(
    d_out::CuVector{T},
    d_in::CuVector{T},
    n::TIdx,
) where {T,TIdx<:Integer}
    shmem_size::TIdx = BLOCK_SIZE * TIdx(2) * TIdx(sizeof(T))

    numBlocks = TIdx(cld(n, BLOCK_SIZE * TIdx(2)))

    if numBlocks > 1 # recursive case
        # allocate intermediate value vectors (no value initialization necessary)
        d_blockSums = CuArray{T}(undef, numBlocks)
        d_increments = CuArray{T}(undef, numBlocks)

        # Block-scan (EXCLUSIVE Blelloch):
        @cuda threads = BLOCK_SIZE blocks = numBlocks shmem = shmem_size always_inline =
            true _scanBlockKernel!(d_out, d_in, d_blockSums, n)
        CUDA.synchronize()

        # Recurse to get exclusive sums of the block sums
        largeArrayScanExclusive!(d_increments, d_blockSums, numBlocks)
        CUDA.synchronize()

        # Add prefixes to each block
        @cuda threads = BLOCK_SIZE blocks = numBlocks always_inline = true _addIncrementsKernel!(
            d_out,
            d_increments,
            n,
        )
        CUDA.synchronize()
    else
        # Base case: 1 block scan:
        @cuda threads = BLOCK_SIZE blocks = numBlocks shmem = shmem_size always_inline =
            true _scanBlockKernel!(d_out, d_in, nothing, n)
        CUDA.synchronize()
    end

    return nothing
end

function largeArrayScanInclusive!(
    d_out::CuVector{T},
    d_in::CuVector{T},
    n::TIdx,
) where {T,TIdx<:Integer}
    largeArrayScanExclusive!(d_out, d_in, n)

    # Turning an EXCLUSIVE result into an INCLUSIVE one by adding input elements
    d_out .+= d_in
    return nothing
end

end
