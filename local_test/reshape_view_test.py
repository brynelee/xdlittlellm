import torch

# 创建一个形状为(2, 3)的张量
x = torch.tensor([[1, 2, 3], [4, 5, 6]])
print("原始张量 x:")
print(x)

# 使用 reshape() 改变形状为 (3, 2)
y = x.reshape((3, 2))
print("使用 reshape() 后的张量 y:")
print(y)


# 使用 view() 改变形状为 (3, 2)
z = x.view((3, 2))
print("使用 view() 后的张量 z:")
print(z)

# 尝试将非连续张量转换为新形状将失败
non_contiguous_x = x[:, [1, 0, 2]]  # 通过索引操作得到非连续张量
print("non_contiguous_x: ")
print(non_contiguous_x)
try:
    x1 = non_contiguous_x.view((3, 2))  # 如果参数是(3, 1)，尝试使用 view() 将会抛出错误, 
except RuntimeError as e:
    print("尝试使用 view() 时的错误:", e)
