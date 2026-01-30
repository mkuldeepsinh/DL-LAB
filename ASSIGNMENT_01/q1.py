import torch

t1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
t2 = torch.ones((2, 2))
t3 = torch.randn(2, 2)

add_res = t1 + t2
mul_res = t1 * 5 
sub_res = t1 - t3
idx_res = t1[:, 1]
reshaped = t1.view(4)

x = torch.tensor([2.0], requires_grad=True)
y = x**2 + 5*x
y.backward()
grad_val = x.grad