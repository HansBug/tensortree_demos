import time

import numpy as np
import torch

import tensortree.torch as ttorch

N, M, T = 300, 2, 50
S1, S2, S3 = 512, 2500, 1024


def test_min():
    a = ttorch.randn({f'a{i}': (S1, S2) for i in range(N // M)}, device='cuda')
    b = ttorch.randn({f'a{i}': (S2, S3) for i in range(N // M)}, device='cuda')

    result = []
    for i in range(T):
        _start_time = time.time()

        _ = ttorch.matmul(a, b)
        torch.cuda.synchronize()

        _end_time = time.time()
        result.append(_end_time - _start_time)

    print('time cost: mean({}) std({})'.format(np.mean(result), np.std(result)))


def test_native():
    a = {f'a{i}': torch.randn(S1, S2, device='cuda') for i in range(N)}
    b = {f'a{i}': torch.randn(S2, S3, device='cuda') for i in range(N)}

    result = []
    for i in range(T):
        _start_time = time.time()

        for key in a.keys():
            _ = torch.matmul(a[key], b[key])
        torch.cuda.synchronize()

        _end_time = time.time()
        result.append(_end_time - _start_time)

    print('time cost: mean({}) std({})'.format(np.mean(result), np.std(result)))


def test_linear():
    a = ttorch.randn({f'a{i}': (S1, S2) for i in range(N)}, device='cuda')
    b = ttorch.randn({f'a{i}': (S2, S3) for i in range(N)}, device='cuda')

    result = []
    for i in range(T):
        _start_time = time.time()

        _ = ttorch.matmul(a, b)
        torch.cuda.synchronize()

        _end_time = time.time()
        result.append(_end_time - _start_time)

    print('time cost: mean({}) std({})'.format(np.mean(result), np.std(result)))


def test_stream():
    a = ttorch.randn({f'a{i}': (S1, S2) for i in range(N)}, device='cuda')
    b = ttorch.randn({f'a{i}': (S2, S3) for i in range(N)}, device='cuda')

    ttorch.stream(M)
    result = []
    for i in range(T):
        _start_time = time.time()

        _ = ttorch.matmul(a, b)
        torch.cuda.synchronize()

        _end_time = time.time()
        result.append(_end_time - _start_time)

    print('time cost: mean({}) std({})'.format(np.mean(result), np.std(result)))


def warmup():
    # warm up
    a = torch.randn(1024, 1024).cuda()
    b = torch.randn(1024, 1024).cuda()
    for _ in range(20):
        c = torch.matmul(a, b)


if __name__ == '__main__':
    warmup()
    test_min()
    test_native()
    test_linear()
    test_stream()
