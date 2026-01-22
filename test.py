from monai.networks.nets import UNet
import torch
from torchsummary import summary
import time
def iter_gen():
    for i in range(10):
        print(i)
        yield i

def list_gen():
    li = []
    for i in range(10):
        print(i)
        li.append(i)
        
    return li


for result in list_gen():
    print("Yiled: ", result)
    time.sleep(1.0)
