# a = 1,2,3,4
# print(a[0])
# for data in ([1,2,3],[2,4,6]):
#     print(data)

import torch

# a = torch.LongTensor([[1,2],[3,4]])

# print(a.view(-1))
# print(a.view(list([2,2])+[-1]))
# print(list([2,2])+[-1])

# b = torch.ones((128,64,768))
# print(b.shape)
# print(b.view([128,-1,2,32,768]).shape)
# print(b.view([128,-1,2,32,768]).view([-1,2*768]).shape)


# c= torch.ones((128,768))
# print(c.shape)
# print(c.view([-1,768]).shape)

# b = b.view([128,-1,2,32,768]).view([-1,2*768])
# print(b.shape)
# d= torch.cat([b, c], dim=-1)
# print(d.shape)

# b = torch.ones((64,2))
# print(b)
# a = torch.ones((2,2))
# print(a)
# print(a+b)

a = torch.ones((128,64,1,768))
b = torch.ones((128,2,1,768))

a = a.view([128,2,-1,768])
print(a.view([128,2,-1,768]).shape)

c = torch.zeros(a.shape)
print(c)
c= c+b
print(c)
print(c.shape)
# for i in range(b.shape[1]):
#     for j in range(a.shape[2]):
#         a[:][i][]torch.cat(b)
print(torch.cat((a,c),dim = -1).shape)