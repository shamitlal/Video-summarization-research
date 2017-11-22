import random
import numpy as np
import math
from PIL import Image
import matplotlib.pyplot as plt

class PatchMatch:

    def __init__(self, Img_A, Img_B):
        self.boxsize = 7
        self.iterations = 5
        self.infinity = 100000000000000
        self.alpha = 0.5
        self.boxsizeBy2 = self.boxsize/2
        self.Img_A = Img_A
        self.Img_B = Img_B
        print "shape = ", self.Img_A.shape
        self.window = max(self.Img_B.shape[0],self.Img_B.shape[1])
        #return correspondences in three channels: x_coordinates, y_coordinates, offsets
        #Here x represents dimension 0 and y represents dimension 1        
        self.nnf_x = np.zeros((self.Img_A.shape[0], self.Img_A.shape[1]))
        self.nnf_y = np.zeros((self.Img_A.shape[0], self.Img_A.shape[1]))
        self.nnf_D = np.zeros((self.Img_A.shape[0], self.Img_A.shape[1]))

    # calculate offset in terms of sum of square differences
    def cal_offset(self, ax, ay, bx, by):
        box_A = (int(ax-self.boxsizeBy2), int(ay-self.boxsizeBy2), int(ax+self.boxsizeBy2), int(ay+self.boxsizeBy2))
        box_B = (int(bx-self.boxsizeBy2), int(by-self.boxsizeBy2), int(bx+self.boxsizeBy2), int(by+self.boxsizeBy2))
        # self.patch_A = self.Img_A.crop(box_A)
        # self.patch_B = self.Img_B.crop(box_B)
        self.patch_A = self.Img_A[box_A[0]:box_A[2]+1 , box_A[1]:box_A[3]+1 , :]
        self.patch_B = self.Img_B[box_B[0]:box_B[2]+1 , box_B[1]:box_B[3]+1 , :]
        # print "patch shapes = ",self.patch_A.shape, self.patch_B.shape, ax, ay, bx, by
        # print "boxes = ",box_A, box_B
        self.abs_diff = np.array(self.patch_A, dtype=np.float32) - np.array(self.patch_B, dtype = np.float32)
        self.ssd = math.sqrt(np.sum(np.square(self.abs_diff)))
        return self.ssd    

    #Random initialization by sampling from uniform distribution for each pixel in image A
    def uniform_random_init_nnf(self):
        print "inside init nnf"
        for i in range(self.boxsizeBy2, self.nnf_x.shape[0]-self.boxsizeBy2):
            for j in range(self.boxsizeBy2, self.nnf_x.shape[1]-self.boxsizeBy2):
                self.nnf_x[i][j] = np.random.randint(self.boxsizeBy2, self.Img_A.shape[0]-self.boxsizeBy2)
                self.nnf_y[i][j] = np.random.randint(self.boxsizeBy2, self.Img_A.shape[1]-self.boxsizeBy2)
                self.nnf_D[i][j] = self.cal_offset(i, j, self.nnf_x[i][j], self.nnf_y[i][j])
        return self.nnf_x, self.nnf_y, self.nnf_D 

    def improveByCheckingLeftAndTopPixel(self):

        #x represents rows and y represents columns
        print "inside improveByCheckingLeftAndTopPixel"
        for x in xrange(self.boxsizeBy2, self.Img_A.shape[0]-self.boxsizeBy2):
            for y in xrange(self.boxsizeBy2, self.Img_A.shape[1]-self.boxsizeBy2):
                leftSSD = self.infinity
                topSSD = self.infinity
                currentSSD = self.nnf_D[x][y]
                if y > self.boxsizeBy2 and self.nnf_y[x][y-1] < self.Img_B.shape[1]-self.boxsizeBy2-1:
                    leftSSD = self.cal_offset(x, y, int(self.nnf_x[x][y-1]), int(self.nnf_y[x][y-1])+1)
                if x > self.boxsizeBy2 and self.nnf_x[x-1][y] < self.Img_B.shape[0]-self.boxsizeBy2-1:              
                    topSSD = self.cal_offset(x, y, int(self.nnf_x[x-1][y]+1), int(self.nnf_y[x-1][y]))

                if leftSSD == self.infinity and topSSD == self.infinity:
                    pass
                else:
                    tempSSD = 0
                    tempx = 0
                    tempy = 0
                    if leftSSD < topSSD:
                        tempSSD = leftSSD
                        tempx = self.nnf_x[x][y-1]
                        tempy = self.nnf_y[x][y-1]+1
                    else:
                        tempSSD = topSSD
                        tempx = self.nnf_x[x-1][y]+1
                        tempy = self.nnf_y[x-1][y]

                    if tempSSD < currentSSD:
                        self.nnf_D[x][y] = tempSSD
                        self.nnf_x[x][y] = tempx
                        self.nnf_y[x][y] = tempy

                self.randomSearch(x,y)




    def improveByCheckingRightAndDownPixel(self):
        #x represents rows and y represents columns
        print "inside improveByCheckingRightAndDownPixel"
        for x in xrange(self.Img_A.shape[0]-self.boxsizeBy2-1, self.boxsizeBy2-1, -1):
            for y in xrange(self.Img_A.shape[1]-self.boxsizeBy2-1, self.boxsizeBy2-1, -1):
                rightSSD = self.infinity
                downSSD = self.infinity
                currentSSD = self.nnf_D[x][y]
                if y < self.Img_A.shape[1]-self.boxsizeBy2-1 and self.nnf_y[x][y+1] > self.boxsizeBy2:
                    rightSSD = self.cal_offset(x, y, int(self.nnf_x[x][y+1]), int(self.nnf_y[x][y+1])-1)
                if x > self.Img_A.shape[0]-self.boxsizeBy2-1 and self.nnf_x[x+1][y] > self.boxsizeBy2:
                    downSSD = self.cal_offset(x, y, int(self.nnf_x[x+1][y]-1), int(self.nnf_y[x+1][y]))

                if rightSSD == self.infinity and downSSD == self.infinity:
                    pass
                else:
                    tempSSD = 0
                    tempx = 0
                    tempy = 0
                    if rightSSD < downSSD:
                        tempSSD = rightSSD
                        tempx = self.nnf_x[x][y+1]
                        tempy = self.nnf_y[x][y+1]-1
                    else:
                        tempSSD = downSSD
                        tempx = self.nnf_x[x+1][y]-1
                        tempy = self.nnf_y[x+1][y]

                    if tempSSD < currentSSD:
                        self.nnf_D[x][y] = tempSSD
                        self.nnf_x[x][y] = tempx
                        self.nnf_y[x][y] = tempy

                self.randomSearch(x,y)


    def randomSearch(self, x, y):
        radius = 1.0*self.window
        while radius > 1:
            randx = random.uniform(-1,1)*radius
            randy = random.uniform(-1,1)*radius

            examinePatchX = int(self.nnf_x[x][y] + randx)
            examinePatchY = int(self.nnf_y[x][y] + randy)
            #Clip the coordinates to acceptable range
            if examinePatchX < self.boxsizeBy2:
                examinePatchX = self.boxsizeBy2
            if examinePatchY < self.boxsizeBy2:
                examinePatchY = self.boxsizeBy2
            if examinePatchX > self.Img_B.shape[0] - self.boxsizeBy2 - 1:
                examinePatchX = self.Img_B.shape[0] - self.boxsizeBy2 - 1
            if examinePatchY > self.Img_B.shape[1] - self.boxsizeBy2 - 1:
                examinePatchY = self.Img_B.shape[1] - self.boxsizeBy2 - 1

            examinePatchSSSD = self.cal_offset(x, y, examinePatchX, examinePatchY)
            if examinePatchSSSD < self.nnf_D[x][y]:
                self.nnf_D[x][y] = examinePatchSSSD
                self.nnf_x[x][y] = examinePatchX
                self.nnf_y[x][y] = examinePatchY
            radius *= self.alpha


    # improve nnf offsets by searching and comparing neighbor's offsets
    def improve_nnf_search(self):
        for i in xrange(self.iterations):
            print "iteration:",i
            if i%2 == 0:
                self.improveByCheckingLeftAndTopPixel()
            else:
                self.improveByCheckingRightAndDownPixel()
            print np.sum(self.nnf_D)
        return self.nnf_D


if __name__ == "__main__":

    Img_A = np.array(Image.open("bike_a.jpg"))
    Img_B = np.array(Image.open("bike_b.jpg"))


    test=PatchMatch(Img_A, Img_B)
    
    a,b,c = test.uniform_random_init_nnf()
    print np.sum(c)
    a=test.improve_nnf_search() 
    print np.sum(a)
