module InclusiveScans

using CUDA

# CUDA scan block size
const BLOCK_SIZE = 1024

@inline function _ceil_div(a, b)
    return cld(a, b)
end

# Execute BLELLOCH EXCLUSIVE‐prefix-sum for current block.
# - Each block processes 2 * BLOCK_SIZE elements (2 per thread).
# - If `blockSums !== nothing`, writes the "block sum" to blockSums[blockIdx.x].
# - Otherwise skips this step (when we scan blockSums itself).
function _scanBlockKernel!(
    g_odata::CuDeviceVector{Float32,1},
    g_idata::CuDeviceVector{Float32,1},
    blockSums::Union{CuDeviceVector{Float32,1},Nothing},
    n::Int,
)
    # Dynamic shared memory on 2*BLOCK_SIZE float
    temp = @cuDynamicSharedMem(Float32, blockDim().x * 2)

    tx = threadIdx().x - 1
    bx = blockIdx().x - 1

    # Every block evaluate 2*BLOCK_SIZE => start = bx*(2*BLOCK_SIZE)
    start = bx * (blockDim().x * 2)

    function sm(i::Int)
        return temp[i+1]
    end
    function set_sm(i::Int, v::Float32)
        temp[i+1] = v
    end

    i1 = start + tx * 2
    i2 = i1 + 1

    if tx >= 0 && 2 * tx < 2 * blockDim().x
        if i1 < n
            set_sm(2 * tx, g_idata[i1+1])
        else
            set_sm(2 * tx, 0.0f0)
        end
        if i2 < n
            set_sm(2 * tx + 1, g_idata[i2+1])
        else
            set_sm(2 * tx + 1, 0.0f0)
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
            set_sm(bi, sm(bi) + sm(ai))
        end
        offset <<= 1
        d >>= 1
    end

    sync_threads()

    # Save sum of block -> blockSums[bx], 
    # Set last to 0 (to get EXCLUSIVE)
    if tx == 0
        if blockSums !== nothing
            blockSums[bx+1] = sm(2 * blockDim().x - 1)
        end
        set_sm(2 * blockDim().x - 1, 0.0f0)
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
            t = sm(ai)
            set_sm(ai, sm(bi))
            set_sm(bi, sm(bi) + t)
        end
        d <<= 1
    end
    sync_threads()

    if tx >= 0 && 2 * tx < 2 * blockDim().x
        if i1 < n
            g_odata[i1+1] = sm(2 * tx)
        end
        if i2 < n
            g_odata[i2+1] = sm(2 * tx + 1)
        end
    end

    return nothing
end

# Add prefix sum from previous blocks (d_increments[blockIdx.x]) to all elements processing by this block
function _addIncrementsKernel!(
    g_odata::CuDeviceVector{Float32,1},
    incr::CuDeviceVector{Float32,1},
    n::Int,
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

function _exclusiveToInclusiveKernel!(
    d_out::CuDeviceVector{Float32,1},
    d_in::CuDeviceVector{Float32,1},
    n::Int,
)
    idx = (blockIdx().x - 1) * blockDim().x + (threadIdx().x - 1)
    if idx >= 0 && idx < n
        d_out[idx+1] += d_in[idx+1]
    end
    return nothing
end

# EXCLUSIVE‐scan (Blelloch) to inclusive and write in d_out.
# d_out[i] = sum of elements to i (include i) d_in array.
function largeArrayScanInclusive!(d_out::CuArray{Float32}, d_in::CuArray{Float32}, n::Int)
    # Blocks scan
    numBlocks = _ceil_div(n, BLOCK_SIZE * 2)

    d_blockSums = CUDA.zeros(Float32, numBlocks)
    d_increments = CUDA.zeros(Float32, numBlocks)

    shmem_size = BLOCK_SIZE * 2 * sizeof(Float32)
    @cuda threads = BLOCK_SIZE blocks = numBlocks shmem = shmem_size _scanBlockKernel!(
        d_out,
        d_in,
        d_blockSums,
        Int(n),
    )
    CUDA.synchronize()

    # Scan d_blockSums in one block (exclusive)
    @cuda threads = BLOCK_SIZE blocks = 1 shmem = shmem_size _scanBlockKernel!(
        d_increments,
        d_blockSums,
        nothing,
        Int(numBlocks),
    )
    CUDA.synchronize()

    # Add increment in each block
    @cuda threads = BLOCK_SIZE blocks = numBlocks _addIncrementsKernel!(
        d_out,
        d_increments,
        Int(n),
    )
    CUDA.synchronize()

    numThreads = min(BLOCK_SIZE, n)
    numFullBlocks = _ceil_div(n, BLOCK_SIZE)
    @cuda threads = numThreads blocks = numFullBlocks _exclusiveToInclusiveKernel!(
        d_out,
        d_in,
        Int(n),
    )
    CUDA.synchronize()

    return nothing
end
end