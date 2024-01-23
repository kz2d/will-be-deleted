import numpy as np

class Lin:
    def __init__(self):
        self.w = []

    def fit(self, inp, tar):
        inp_tmp = np.hstack((inp, np.array([[1]] * len(inp))))
        self.w, _,_,_ = np.linalg.lstsq(inp_tmp, tar, rcond=None)

    def predict(self, inp):
        inp_tmp = np.hstack((inp, np.array([[1]] * len(inp))))
        return np.dot(inp_tmp,self.w)
