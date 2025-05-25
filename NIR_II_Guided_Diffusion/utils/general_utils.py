import torch
import math



def normalization(img):
    return torch.stack([(b-b.min())/(b.max()-b.min()) for b in img])


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
