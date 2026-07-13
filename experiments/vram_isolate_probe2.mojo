# Does DeviceContext() ALONE, with zero user allocations, already reserve
# a large VRAM pool? If so, the 29.6GB figure has nothing to do with the
# loader's buffer choices at all.

from std.gpu.host import DeviceContext
from std.time import sleep


def main() raises:
    print("creating DeviceContext, allocating NOTHING")
    var ctx = DeviceContext()
    print("context created -- holding for 3s")
    sleep(3.0)
