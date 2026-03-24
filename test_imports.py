import torch
print("Torch version:", torch.__version__)
import numpy as np
print("Numpy version:", np.__version__)
import cv2
print("CV2 version:", cv2.__version__)
from core.sam_helper import SAM2Helper
print("SAM2Helper imported")
from core.tracker_helper import TrackerHelper
print("TrackerHelper imported")
