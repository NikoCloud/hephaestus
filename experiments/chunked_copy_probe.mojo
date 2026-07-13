# Probe: verify ctx.enqueue_copy(dst_ptr, src_ptr, size) works for BOTH
# directions (host->device-at-offset, device-at-offset->host) before
# rewriting the loader to stream in chunks instead of holding the full
# ~8GB blob in host memory (the source of the 29.6GB peak VRAM finding,
# bench/1a-ab.md Finding 3).

from std.gpu.host import DeviceContext

comptime N = 1024  # total device arena elements
comptime CHUNK = 256  # chunk size


def main() raises:
    var ctx = DeviceContext()
    var dev = ctx.enqueue_create_buffer[DType.float32](N)
    var host_chunk = ctx.enqueue_create_host_buffer[DType.float32](CHUNK)
    var back_chunk = ctx.enqueue_create_host_buffer[DType.float32](CHUNK)

    # Fill each chunk on host with a distinct pattern, copy to the
    # corresponding OFFSET in the device arena.
    for c in range(N // CHUNK):
        var h = host_chunk.unsafe_ptr()
        for i in range(CHUNK):
            h[i] = Float32(c * 1000 + i)
        ctx.enqueue_copy(dev.unsafe_ptr() + c * CHUNK, host_chunk.unsafe_ptr(), CHUNK)
        ctx.synchronize()

    # Copy back chunk-by-chunk and verify.
    var errors = 0
    for c in range(N // CHUNK):
        ctx.enqueue_copy(back_chunk.unsafe_ptr(), dev.unsafe_ptr() + c * CHUNK, CHUNK)
        ctx.synchronize()
        var h = back_chunk.unsafe_ptr()
        for i in range(CHUNK):
            var expected = Float32(c * 1000 + i)
            if h[i] != expected:
                errors += 1
                if errors <= 3:
                    print("mismatch chunk", c, "i", i, "got", h[i], "want", expected)

    print("errors:", errors, "/", N)
    if errors == 0:
        print("EXP chunked_copy PASS: pointer+offset+size copy works both directions")
