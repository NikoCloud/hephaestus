# Isolate what actually drives peak VRAM: allocate ONLY an 8GB device
# buffer (no host staging buffers at all), to test whether the arena alone
# already shows an outsized footprint (pointing to DeviceContext/allocator
# behavior unrelated to the loader's own buffer choices).

from std.gpu.host import DeviceContext
from std.time import sleep

comptime ELEMS = 536870912  # 1GB bf16  # matches the 4B model's total BF16 elements


def main() raises:
    var ctx = DeviceContext()
    print("device buffer alone, no host buffers, no copies")
    var dev = ctx.enqueue_create_buffer[DType.bfloat16](ELEMS)
    ctx.synchronize()
    print("allocated", ELEMS * 2, "bytes on device -- holding for 3s")
    sleep(3.0)
    _ = dev^
