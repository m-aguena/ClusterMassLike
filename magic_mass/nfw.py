import numpy as np


# In[28]:


def nfw_int(x):
    return np.log(1 + x) - (x / (1 + x))


def convert_mass(M1, c1, c2):
    return M1 * nfw_int(c2) / nfw_int(c1)
