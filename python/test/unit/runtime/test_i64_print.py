import triton
import triton.language as tl
import torch


def test_i64_printf(capfd):

    @triton.jit
    def ndscore_kernel(ptr):
        value = tl.load(ptr)
        print("value in kernel", value)
        tl.store(ptr, value + 1)

    ptr = torch.tensor(42, dtype=torch.int64).cuda()
    print("value before kernel", ptr.item())
    kernel = ndscore_kernel[(1, )](ptr)
    kernel
    print("value after kernel", ptr.item())
    captured = capfd.readouterr()
    assert "value in kernel: 42" in captured.out
    assert "value before kernel 42" in captured.out
    assert "value after kernel 43" in captured.out
