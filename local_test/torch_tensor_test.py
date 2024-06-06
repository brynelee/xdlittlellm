import torch

# torch.rand：返回一个张量，包含了从区间[0,1)的均匀分布中抽取一组随机数，形状由可变参数size定义。
print(torch.rand(5,3))
# torch.randn:返回一个张量，包含了从标准正态分布(Normal distribution)(均值为0，方差为1，即高斯白噪声)中抽取一组随机数，形状由可变参数sizes定义。
print(torch.randn(2,3))

# tensor.new_ones:返回一个与size大小相同的用1填充的张量;默认情况下，返回的Tensor具有与此张量相同的torch.dtype和torch.device;
tensor = torch.tensor((), dtype=torch.int32)
print(tensor.new_ones(5,3))

#  torch.rand_like：返回与输入相同大小的张量，该张量由区间[0,1)上均匀的随机数填充。
x = [[5,3],[2,1]]
x = torch.tensor(x)
y = torch.rand_like(x, dtype=torch.float32)
print(y)

x = [[5,3],[2,1],[3,4]]
x = torch.tensor(x)
y = x[0,:]
y += 1
print(y)
print(x[0,:])

# view()返回的新tensor与源tensor共享内存，实际上就是同一个tensor，也就是更改一个，另一个也会跟着改变。（顾名思义，view()仅仅改变了对这个张量的观察角度）
x = torch.tensor([[5,3],[2,1],[3,4]])
z = x.view(2,-1)
print(z)