import torch
import time
from custom_ops_bn.ops import batchnorm

def batchnorm_cuda(x, g, b, rm, rv, training):
    """Wrapper for CUDA backend batchnorm"""
    return batchnorm(x, g, b, rm, rv, training, backend='cuda')

def batchnorm_triton(x, g, b, rm, rv, training):
    """Wrapper for Triton backend batchnorm"""
    return batchnorm(x, g, b, rm, rv, training, backend='triton')

def test_correctness(N=32, C=64, H=56, W=56, training=True):
    """Test correctness of Triton implementation"""
    print(f"\n{'='*70}")
    print(f"Correctness Test: N={N}, C={C}, H={H}, W={W}, Training={training}")
    print(f"{'='*70}")

    device = 'cuda'

    # Create test tensors
    input_test = torch.randn(N, C, H, W, device=device)
    gamma_test = torch.randn(C, device=device)
    beta_test = torch.randn(C, device=device)
    rm_test = torch.randn(C, device=device)
    rv_test = torch.abs(torch.randn(C, device=device))

    # PyTorch reference
    bn_pytorch = torch.nn.BatchNorm2d(C, momentum=0.1, eps=1e-5).to(device)
    bn_pytorch.weight.data = gamma_test.clone()
    bn_pytorch.bias.data = beta_test.clone()
    bn_pytorch.running_mean.data = rm_test.clone()
    bn_pytorch.running_var.data = rv_test.clone()
    bn_pytorch.train() if training else bn_pytorch.eval()

    with torch.no_grad():
        out_pytorch = bn_pytorch(input_test.clone())

        # Test Triton implementation
        triton_pass = False
        try:
            out_triton = batchnorm(
                input_test.clone(),
                gamma_test.clone(),
                beta_test.clone(),
                rm_test.clone(),
                rv_test.clone(),
                training=training,
                backend='triton'
            )
            diff_triton = torch.max(torch.abs(out_pytorch - out_triton)).item()
            triton_pass = diff_triton < 1e-3

            print(f"\n   Triton max diff: {diff_triton:.2e}")
            print(f"   Triton: {'✓ PASSED' if triton_pass else '✗ FAILED'}")

        except NotImplementedError:
            print(f"\n   Triton: ⚠ NOT IMPLEMENTED (TODO)")

        except Exception as e:
            print(f"\n   Triton: ✗ ERROR - {type(e).__name__}: {str(e)[:80]}")

    return triton_pass


def benchmark_forward_backward(func, input, gamma, beta, running_mean, running_var,
                                training=True, warmup=10, iterations=100):
    """Benchmark forward and backward separately"""
    # Warmup
    for _ in range(warmup):
        rm = running_mean.clone()
        rv = running_var.clone()
        inp = input.clone().detach().requires_grad_(True)
        g = gamma.clone().detach().requires_grad_(True)
        b = beta.clone().detach().requires_grad_(True)

        output = func(inp, g, b, rm, rv, training=training)
        loss = output.sum()
        loss.backward()

    torch.cuda.synchronize()

    # Benchmark - measure forward and backward separately
    total_forward_time = 0
    total_backward_time = 0

    for _ in range(iterations):
        rm = running_mean.clone()
        rv = running_var.clone()
        inp = input.clone().detach().requires_grad_(True)
        g = gamma.clone().detach().requires_grad_(True)
        b = beta.clone().detach().requires_grad_(True)

        # Measure forward
        torch.cuda.synchronize()
        start = time.time()
        output = func(inp, g, b, rm, rv, training=training)
        torch.cuda.synchronize()
        total_forward_time += time.time() - start

        # Measure backward
        torch.cuda.synchronize()
        start = time.time()
        loss = output.sum()
        loss.backward()
        torch.cuda.synchronize()
        total_backward_time += time.time() - start

    avg_forward_time = total_forward_time / iterations * 1000  # Convert to ms
    avg_backward_time = total_backward_time / iterations * 1000  # Convert to ms

    return avg_forward_time, avg_backward_time


def run_benchmark(N, C, H, W, training=True):
    """Run performance benchmark for specific input size"""
    print(f"\n{'='*70}")
    print(f"Performance Benchmark: N={N}, C={C}, H={H}, W={W}, Training={training}")
    print(f"{'='*70}")

    device = 'cuda'

    # Create base tensors
    input_base = torch.randn(N, C, H, W, device=device)
    gamma = torch.randn(C, device=device)
    beta = torch.randn(C, device=device)
    running_mean = torch.randn(C, device=device)
    running_var = torch.abs(torch.randn(C, device=device))

    # CUDA custom operator
    print("\n1. CUDA Custom Operator (Reference)")
    try:
        forward_time_cuda, backward_time_cuda = benchmark_forward_backward(
            batchnorm_cuda,
            input_base, gamma, beta,
            running_mean, running_var, training
        )
        print(f"   Forward:  {forward_time_cuda:.4f} ms")
        print(f"   Backward: {backward_time_cuda:.4f} ms")
    except Exception as e:
        print(f"   ✗ ERROR: {e}")
        return

    # Triton custom operator
    print("\n2. Triton Custom Operator (Your Implementation)")
    try:
        forward_time_triton, backward_time_triton = benchmark_forward_backward(
            batchnorm_triton,
            input_base, gamma, beta,
            running_mean, running_var, training
        )
        print(f"   Forward:  {forward_time_triton:.4f} ms")
        print(f"   Speedup vs CUDA: {forward_time_cuda / forward_time_triton:.2f}x")
        print(f"   Backward: {backward_time_triton:.4f} ms")
        print(f"   Speedup vs CUDA: {backward_time_cuda / backward_time_triton:.2f}x")
    except NotImplementedError:
        print(f"   ⚠ NOT IMPLEMENTED (TODO)")
    except Exception as e:
        print(f"   ✗ ERROR: {type(e).__name__}: {str(e)[:80]}")


def main():
    print("\n" + "="*70)
    print(" BatchNorm Custom Operators: Correctness & Performance Tests")
    print("="*70)

    # Test configurations
    configs = [
        # (8, 32, 224, 224),    
        # (32, 64, 112, 112),   
        (64, 128, 56, 56),    
    ]

    # Run correctness tests *************************************************************************
    print("\n" + "="*70)
    print(" CORRECTNESS TESTS")
    print("="*70)

    all_passed = True
    for N, C, H, W in configs:
        passed_train = test_correctness(N, C, H, W, training=True)
        passed_eval = test_correctness(N, C, H, W, training=False)
        all_passed = all_passed and passed_train and passed_eval

    print("\n" + "="*70)
    if all_passed:
        print(" ALL CORRECTNESS TESTS PASSED")
    else:
        print(" SOME CORRECTNESS TESTS FAILED")
    print("="*70)

    # Run performance benchmarks *************************************************************************
    print("\n" + "="*70)
    print(" PERFORMANCE BENCHMARKS")
    print("="*70)

    for N, C, H, W in configs:
        # Training mode
        run_benchmark(N, C, H, W, training=True)

        # Inference mode
        run_benchmark(N, C, H, W, training=False)

    print("\n" + "="*70)
    print(" Tests Complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
