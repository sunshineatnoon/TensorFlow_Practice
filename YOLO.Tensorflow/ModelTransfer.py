import numpy as np

dt = np.dtype(("(3)float32,i4"))
testArray = np.fromfile('extraction.weights',dtype=dt)
print testArray
