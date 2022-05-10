import numpy as np
import torch

a = np.zeros([5, 5], dtype='<U32')
a.fill('O')

b = np.where(a == 'O', 0 , 1)
c = torch.from_numpy(b)
d = []
for i in range(9):
    d.append(c.unsqueeze(0))
e = torch.cat(d, dim=0)
print(e.shape)