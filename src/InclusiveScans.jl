module InclusiveScans

using CUDA
using CUDA: i32

const BLOCK_SIZE::Int32 = 1024

function _scanBlockKernel!(
    ::Type{T},
    g_odata,
    g_idata,
    blockSums,
    n::Int32,
) where {T}
    temp = CuDynamicSharedArray(T, blockDim().x * 2i32)

    tx = threadIdx().x - 1i32
    bx = blockIdx().x - 1i32

    start = bx * (blockDim().x * 2i32)

    i1 = start + tx * 2i32
    i2 = i1 + 1

    if tx >= 0i32 && 2i32 * tx < 2i32 * blockDim().x
        if i1 < n
            temp[2i32*tx+1i32] = g_idata[i1+1i32]
        else
            temp[2i32*tx+1i32] = 0.0f0
        end
        if i2 < n
            temp[2i32*tx+2i32] = g_idata[i2+1i32]
        else
            temp[2i32*tx+2i32] = 0.0f0
        end
    end
    sync_threads()

    # Up-sweep (reduce)
    offset = 1i32
    d = blockDim().x
    while d > 0i32
        sync_threads()
        if tx < d
            ai = offset * (2i32 * tx + 1i32) - 1i32
            bi = offset * (2i32 * tx + 2i32) - 1i32
            temp[bi+1i32] += temp[ai+1i32]
        end
        offset <<= 1i32
        d >>= 1i32
    end

    sync_threads()

    # Save sum of block -> blockSums[bx]
    if tx == 0i32
        if blockSums !== nothing
            blockSums[bx+1i32] = temp[2i32*blockDim().x]
        end
        temp[2i32*blockDim().x] = 0.0f0
    end
    sync_threads()

    # Down-sweep
    d = 1i32
    while d < 2i32 * blockDim().x
        offset >>= 1i32
        sync_threads()
        if tx < d
            ai = offset * (2i32 * tx + 1i32) - 1i32
            bi = offset * (2i32 * tx + 2i32) - 1i32
            t = temp[ai+1i32]
            temp[ai+1i32] = temp[bi+1i32]
            temp[bi+1i32] += t
        end
        d <<= 1i32
    end
    sync_threads()

    if tx >= 0i32 && 2i32 * tx < 2i32 * blockDim().x
        if i1 < n
            g_odata[i1+1i32] = temp[2i32*tx+1i32]
        end
        if i2 < n
            g_odata[i2+1i32] = temp[2i32*tx+2i32]
        end
    end

    return nothing
end

# Add prefix sum from previous blocks (d_increments[blockIdx.x]) to all elements processing by this block
function _addIncrementsKernel!(g_odata, incr, n::Int32)
    tx = threadIdx().x - 1i32
    bx = blockIdx().x - 1i32

    start = bx * (blockDim().x * 2i32)
    i = start + tx * 2i32

    if i < n
        g_odata[i+1i32] += incr[bx+1i32]
        if i + 1i32 < n
            g_odata[i+2i32] += incr[bx+1i32]
        end
    end
    return nothing
end

function prepareScanBlocks!(::Type{T}, n::Int32) where {T}
    numBlocks = Int32(cld(n, BLOCK_SIZE * 2i32))

    d_blockSums = CUDA.zeros(T, numBlocks)
    d_increments = CUDA.zeros(T, numBlocks)
    return (numBlocks, d_blockSums, d_increments)
end

function largeArrayScanInclusive!(
    d_out::CuArray{T},
    d_in::CuArray{T},
    n::Int32,
    d_blockSums::CuArray{T},
    d_increments::CuArray{T},
    numBlocks::Int32,
) where {T}
    shmem_size = BLOCK_SIZE * 2i32 * Int32(sizeof(T))

    # Block-scan (EXCLUSIVE Blelloch):
    @cuda threads = BLOCK_SIZE blocks = numBlocks shmem = shmem_size _scanBlockKernel!(
        T,
        d_out,
        d_in,
        d_blockSums,
        n,
    )
    CUDA.synchronize()

    # Add the block prefixes (also EXCLUSIVE) into one block
    @cuda threads = BLOCK_SIZE blocks = 1i32 shmem = shmem_size _scanBlockKernel!(
        T,
        d_increments,
        d_blockSums,
        nothing,
        numBlocks,
    )
    CUDA.synchronize()

    # Add prefixes to each block
    @cuda threads = BLOCK_SIZE blocks = numBlocks _addIncrementsKernel!(
        d_out,
        d_increments,
        n,
    )
    CUDA.synchronize()

    # Turning an EXCLUSIVE result into an INCLUSIVE one by adding input elements
    d_out .+= d_in
    return nothing
end

function largeArrayScanInclusive!(d_out::CuArray{T}, d_in::CuArray{T}, n::Int32) where {T}
    numBlocks, d_blockSums, d_increments = prepareScanBlocks!(T, n)
    largeArrayScanInclusive!(d_out, d_in, n, d_blockSums, d_increments, numBlocks)
    return nothing
end

end
