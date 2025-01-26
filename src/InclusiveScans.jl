module InclusiveScans

using CUDA

const BLOCK_SIZE::Int32 = 1024

function _scanBlockKernel!(
    g_odata::CuDeviceVector{Float32,1},
    g_idata::CuDeviceVector{Float32,1},
    blockSums::Union{CuDeviceVector{Float32,1},Nothing},
    n::Int32,
)
    temp = CuDynamicSharedArray(Float32, blockDim().x * 2)

    tx = threadIdx().x - 1
    bx = blockIdx().x - 1

    start = bx * (blockDim().x * 2)

    i1 = start + tx * 2
    i2 = i1 + 1

    if tx >= 0 && 2 * tx < 2 * blockDim().x
        if i1 < n
            temp[2*tx+1] = g_idata[i1+1]
        else
            temp[2*tx+1] = 0.0f0
        end
        if i2 < n
            temp[2*tx+2] = g_idata[i2+1]
        else
            temp[2*tx+2] = 0.0f0
        end
    end
    sync_threads()

    # Up-sweep (reduce)
    offset = 1
    d = blockDim().x
    while d > 0
        sync_threads()
        if tx < d
            ai = offset * (2 * tx + 1) - 1
            bi = offset * (2 * tx + 2) - 1
            temp[bi+1] += temp[ai+1]
        end
        offset <<= 1
        d >>= 1
    end

    sync_threads()

    # Save sum of block -> blockSums[bx]
    if tx == 0
        if blockSums !== nothing
            blockSums[bx+1] = temp[2*blockDim().x]
        end
        temp[2*blockDim().x] = 0.0f0
    end
    sync_threads()

    # Down-sweep
    d = 1
    while d < 2 * blockDim().x
        offset >>= 1
        sync_threads()
        if tx < d
            ai = offset * (2 * tx + 1) - 1
            bi = offset * (2 * tx + 2) - 1
            t = temp[ai+1]
            temp[ai+1] = temp[bi+1]
            temp[bi+1] += t
        end
        d <<= 1
    end
    sync_threads()

    if tx >= 0 && 2 * tx < 2 * blockDim().x
        if i1 < n
            g_odata[i1+1] = temp[2*tx+1]
        end
        if i2 < n
            g_odata[i2+1] = temp[2*tx+2]
        end
    end

    return nothing
end

# Add prefix sum from previous blocks (d_increments[blockIdx.x]) to all elements processing by this block
function _addIncrementsKernel!(
    g_odata::CuDeviceVector{Float32,1},
    incr::CuDeviceVector{Float32,1},
    n::Int32,
)
    tx = threadIdx().x - 1
    bx = blockIdx().x - 1

    start = bx * (blockDim().x * 2)
    i = start + tx * 2

    if i < n
        g_odata[i+1] += incr[bx+1]
        if i + 1 < n
            g_odata[i+2] += incr[bx+1]
        end
    end
    return nothing
end

function largeArrayScanInclusive!(d_out::CuArray{Float32}, d_in::CuArray{Float32}, n::Int32)
    numBlocks = Int32(cld(n, BLOCK_SIZE * 2))

    d_blockSums = CUDA.zeros(Float32, numBlocks)
    d_increments = CUDA.zeros(Float32, numBlocks)

    shmem_size = BLOCK_SIZE * 2 * sizeof(Float32)

    # Block-scan (EXCLUSIVE Blelloch):
    @cuda threads = BLOCK_SIZE blocks = numBlocks shmem = shmem_size _scanBlockKernel!(
        d_out,
        d_in,
        d_blockSums,
        n,
    )
    CUDA.synchronize()

    # Add the block prefixes (also EXCLUSIVE) into one block
    @cuda threads = BLOCK_SIZE blocks = 1 shmem = shmem_size _scanBlockKernel!(
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

end
