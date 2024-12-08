import cutlass
import torch


class GroupedGemm(torch.autograd.Function):

    @staticmethod
    def forward(ctx, a, b, batch_sizes, trans_b):
        ctx.save_for_backward(a, b, batch_sizes)
        ctx.trans_b = trans_b
        return backend.gmm(a, b, batch_sizes, trans_a=False, trans_b=trans_b)

    @staticmethod
    def backward(ctx, grad):
        grad = grad.contiguous()
        a, b, batch_sizes = ctx.saved_tensors
        trans_b = ctx.trans_b

        agrad = None
        if ctx.needs_input_grad[0]:
            agrad = backend.gmm(
                grad, b, batch_sizes, trans_a=False, trans_b=not trans_b)

        bgrad = None
        if ctx.needs_input_grad[1]:
            lhs, rhs = (grad, a) if trans_b else (a, grad)
            bgrad = backend.gmm(
                lhs, rhs, batch_sizes, trans_a=True, trans_b=False)
        return agrad, bgrad, None, None


dtype = torch.float16
plan = cutlass.op.GroupedGemm(element=dtype, layout=cutlass.LayoutType.RowMajor)

def generate_problems(problems):
    valid_sizes = [128, 256, 512, 1024]
    As, Bs, Cs, Ds = [], [], [], []
    for _ in range(problems):
        M, N, K = [random.choice(valid_sizes) for _ in range(3)]
        A, B, C, D = initialize(dtype, M, N, K)
        As.append(A)
        Bs.append(B)
        Cs.append(C)
        Ds.append(D)
    return As, Bs, Cs, Ds

As, Bs, Cs, Ds, = generate_problems(20)

plan.run(As, Bs, Cs, Ds, print_module=True)
