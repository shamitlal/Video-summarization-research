import random
import numpy as np
import math
from PIL import Image
import matplotlib.pyplot as plt
from randomizedPatchMatch import PatchMatch

class BDS:
    def __init__(self, Img_A, Img_B):
        self.Img_A = Img_A
        self.Img_B = Img_B
        self.completeness = PatchMatch(self.Img_A, self.Img_B)
        self.coherence = PatchMatch(self.Img_B, self.Img_A)
        self.completeScore = 0
        self.cohereScore = 0

    def calculateBDSScore(self):
        self.completeness.uniform_random_init_nnf()
        self.coherence.uniform_random_init_nnf()
        self.completeScore = self.completeness.improve_nnf_search()
        self.cohereScore = self.coherence.improve_nnf_search()
        self.totalSimilarityScore = np.sum(self.completeScore + self.cohereScore)
        print self.totalSimilarityScore

if __name__ == "__main__":
    Img_A = np.array(Image.open("bike_a.jpg"))
    Img_B = np.array(Image.open("bike_a.jpg"))
    test = BDS(Img_A, Img_B)
    test.calculateBDSScore()

