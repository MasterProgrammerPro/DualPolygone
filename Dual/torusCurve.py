import numpy as np
from math import floor
from math import ceil
from PIL import Image

class TorusCurvePolygone:

    def __init__(self, edges, resolution=None, round_func=floor, tolerance=0):
        self.edges = edges
        self._round = round_func #?
        self.tolerance = tolerance
        self.bbx = self.bbox()
        if resolution != None:
            self.resolution = resolution
        else:
            self.resolution = 1 + 41 # a changer
        self.boundary_points = self.getGridBoundary()

    '''
    A, B, and C are points. 
    Returns two times of area of the Triangle ABC.
    '''
    @staticmethod
    def det(A, B, C):
        return A[0]*B[1]+B[0]*C[1]+C[0]*A[1]-C[0]*B[1]-A[0]*C[1]-B[0]*A[1]

    '''
    S is a segment.
    C is a point.
    Test if the point C is on the segments S. Returns True if they do
    and False if they dont't.
    '''
    @staticmethod
    def between(S, C):
        linear = (TorusCurvePolygone.det(S[0], S[1], C) == 0.0)
        if linear:    
            if ((S[0][0] < C[0]) and (C[0] < S[1][0])):
                return True
            elif ((S[1][0] < C[0]) and (C[0] < S[0][0])):
                return True
            else:
                return False
        else:
            return False

    '''
    S1 and S2 are segments.
    Test if two segments intersacts. Returns True if they do 
    and False if they don't.
    '''
    @staticmethod
    def testIntersectionOfTwoSegments(S1, S2):
        A = S1[0]
        B = S1[1]
        C = S2[0]
        D = S2[1]
        S1 = TorusCurvePolygone.det(A, B, C)
        S2 = TorusCurvePolygone.det(A, B, D)
        S3 = TorusCurvePolygone.det(C, D, A)
        S4 = TorusCurvePolygone.det(C, D, B)
        return (S1*S2 < 0) and (S3*S4 < 0)


    '''
    Returns intersection point of two segments
    S1 and S2 are segments.(defined by two points)
    '''
    @staticmethod
    def intersection(S1, S2):
        a = S1[0][0]
        b = S1[0][1]
        c = S1[1][0]
        d = S1[1][1]
        e = S2[0][0]
        f = S2[0][1]
        g = S2[1][0]
        h = S2[1][1]
        x= ((a*d-b*c)*(e-g)-(a-c)*(e*h-f*g))/((a-c)*(f-h)-(b-d)*(e-g)) 
        y= ((a*d-b*c)*(f-h)-(b-d)*(e*h-f*g))/((a-c)*(f-h)-(b-d)*(e-g))
        return [x, y]
        

    '''
    C is a square (Array with 4 points where successive points
    represent square's sides.).
    S is a segment.
    Test if a segments and a square intersact. 
    '''
    @staticmethod
    def testIntersectionSegmentSquare(S, C):
        SC1 = np.array([C[0], C[1]])
        SC2 = np.array([C[1], C[2]])
        SC3 = np.array([C[2], C[3]])
        SC4 = np.array([C[3], C[0]])
        E = np.array([SC1, SC2, SC3, SC4])
        result = 0
        counter = 0
        for i in E:
            if TorusCurvePolygone.testIntersectionOfTwoSegments(i, S):
                result = result | (1<<counter)
            counter += 1
        return result

    '''
    Test if a point p is a boundary point of a polygon E.
    p is a Point.
    E is an array contains our polygon's corners where consecutive elements
    represent polygones sides.
    '''
    @staticmethod
    def isBoundary(p, E):
        carre = np.array([[p[0]-1, p[1]-1],
                          [p[0]  , p[1]-1],
                          [p[0]  , p[1]  ],
                          [p[0]-1, p[1]  ]])
        
        nbE = np.shape(E)[0]
        sides = np.zeros((2*nbE, 2), dtype = int)
        
        for i in range(nbE):
            sides[2*i] = E[i]
            sides[2*i+1] = E[i]
        
        sides = np.reshape(np.roll(sides, -1, axis = 0), (nbE, 2, 2))
        
        result = False
        for i in sides:
            if TorusCurvePolygone.testIntersectionSegmentSquare(i, carre) != 0:
                result = True
                break
        return result


    def gpToGp01(Tips):
        x_aux, y_aux = Tips.min(axis=0)
        x = floor(x_aux)
        y = floor(y_aux)
        a = np.array([[x, y], [x, y]])
        return np.subtract(Tips, a)

    '''
    Test if pixel c is on gamma_p of segment s.
    c is coordinates of a pixel in range [0, resolution).
    s is a segment.
    '''
    def testMembershipGamma_pEtPixel(self, c, s):
        unit = 1/(self.resolution)
        carre = np.array([[(c[0])*unit  , (c[1])*unit  ],
                          [(c[0]+1)*unit, (c[1])*unit  ],
                          [(c[0]+1)*unit, (c[1]+1)*unit],
                          [(c[0])*unit  , (c[1]+1)*unit]])
        s01 = self.gpToGp01(s)
        r = TorusCurvePolygone.testIntersectionSegmentSquare(s01, carre)
        return r


    '''
    Returns the boundary box.
    '''
    def bbox(self):
        x_min, y_min = self.edges.min(axis=0)
        x_max, y_max = self.edges.max(axis=0)
        return floor(x_min)-1, floor(y_min)-1, ceil(x_max)+1, ceil(y_max)+1


    '''
    Returns the boundary points of Polygon.
    '''
    def getGridBoundary(self):
        b_list = []
        (x_min, y_min, x_max, y_max) = self.bbox()
        for y in range(y_min, y_max):
            for x in range(x_min, x_max):
                if (self.isBoundary([x, y], self.edges)):
                    b_list.append([x, y])
        return np.array(b_list)

    def getKernel(self): #directly from the original code
        k = set()
        for s in self.ker.values():
            k = k.union(s)
        l = list(k)
        l.sort()
        return l

    def showTorus(self): #directly from the original code
        img = np.zeros((self.resolution,self.resolution), dtype=np.int8)
        curve = [p[1] for p in self.plot]
        for (x,y) in curve:
           img[y,x] = 255
        torus = Image.fromarray(img, mode="L")
        torus.show()

    def saveTorus(self, myfile): #directly from the original code
        img = np.zeros((self.resolution,self.resolution), dtype=np.int8)
        curve = [p[1] for p in self.plot]
        for (x,y) in curve:
           img[y,x] = 255
        torus = Image.fromarray(img, mode="L")
        torus.save(myfile + ".png")

    def showdig(self): #directly from the original code
        img = np.zeros((self.bbx[2]-self.bbx[0],self.bbx[3]-self.bbx[1]), dtype=np.int8)         
        for (y,x) in self.getKernel():
            img[x - self.bbx[0], y - self.bbx[1]] = 3
        for (y,x) in self.getGridBoundary():
            img[x - self.bbx[0], y - self.bbx[1]] = 1
        for (y,x) in self.getDig():
            img[x - self.bbx[0], y - self.bbx[1]] = 2
            
        dig = Image.fromarray(img, mode="L")
        dig.putpalette([0,0,0,255,0,0,0,255,0,0,0,255])
        dig.show()

    def saveDig(self, myfile): #directly from the original code
        img = np.zeros((self.bbx[2]-self.bbx[0],self.bbx[3]-self.bbx[1]), dtype=np.int8)         
        for (y,x) in self.getKernel():
            img[x - self.bbx[0], y - self.bbx[1]] = 3
        for (y,x) in self.getGridBoundary():
            img[x - self.bbx[0], y - self.bbx[1]] = 1
        for (y,x) in self.getDig():
            img[x - self.bbx[0], y - self.bbx[1]] = 2
            
        dig = Image.fromarray(img, mode="L")
        dig.putpalette([0,0,0,255,0,0,0,255,0,0,0,255])
        dig.save(myfile + ".png")


if __name__ == '__main__':
    
    edges = np.array([[-2, 0], [-1, 1], [2, 0], [0, -1]])
    res = 256
    quad = TorusCurvePolygone(edges=edges, resolution=res)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    