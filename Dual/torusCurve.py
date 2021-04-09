import numpy as np
from math import floor
from math import ceil
from PIL import Image
import matplotlib.pyplot as plt

class TorusCurvePolygone:

    def __init__(self, vertices, resolution=None, round_func=floor, tolerance=0):
        self.vertices = vertices
        self._round = round_func #?
        self.tolerance = tolerance
        self.bbx = self.bbox()
        if resolution != None:
            self.resolution = resolution
        else:
            self.resolution = 1 + 41 # a changer
        self.boundary_points = self.getGridBoundary()
        self.gamma_p = self.list_gamma_p()
        self.torus = self.draw_torus()

    
    # A, B, and C are points. 
    # Returns two times of area of the Triangle ABC.
    @staticmethod
    def det(A, B, C):
        return A[0]*B[1]+B[0]*C[1]+C[0]*A[1]-C[0]*B[1]-A[0]*C[1]-B[0]*A[1]

    # S is a segment.
    # C is a point.
    # Test if the point C is on the segments S. Returns True if they do
    # and False if they dont't.
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


    # S1 and S2 are segments. S1 = 
    # Test if two segments intersacts. Returns True if they do 
    # and False if they don't.
    @staticmethod
    def testIntersectionOfTwoSegments(S1, S2):
        a = S1[0][0]
        b = S1[0][1]
        c = S1[1][0]
        d = S1[1][1]
        e = S2[0][0]
        f = S2[0][1]
        g = S2[1][0]
        h = S2[1][1]
        if (((a-c)*(f-h)-(b-d)*(e-g)) != 0):
            t = ((a-e)*(f-h)-(b-f)*(e-g))/((a-c)*(f-h)-(b-d)*(e-g))
            u = ((c-a)*(b-f)-(d-b)*(a-e))/((a-c)*(f-h)-(b-d)*(e-g))
            return ((t >= 0) and (t < 1) and (u >= 0) and (u < 1))
        else:
            #print("Possible error in testIntersection two segments")
            return False

    # Returns intersection point of two segments
    # S1 and S2 are segments.(defined by two points)
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
        return np.array([x, y])
        

    # C is a square (Array with 4 points where successive points
    # represent square's sides.).
    # S is a segment.
    # Test if a segments and a square intersact. 
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

    
    # Test if a point p is a boundary point of a polygon E.
    # p is a Point.
    # E is an array contains our polygon's corners where consecutive elements
    # represent polygones sides.
    @staticmethod
    def isBoundary(p, E):
        carre = np.array([[p[0]  , p[1]  ],
                          [p[0]+1, p[1]  ],
                          [p[0]+1, p[1]+1],
                          [p[0]  , p[1]+1]])
        
        sides = TorusCurvePolygone.getEdgesFromVertices(E)
        
        result = False
        for i in sides:
            if TorusCurvePolygone.testIntersectionSegmentSquare(i, carre) != 0:
                result = True
                break
        return result

    @staticmethod
    def gpToGp01(Tips):
        x_aux, y_aux = Tips.min(axis=0)
        x = floor(x_aux)
        y = floor(y_aux)
        a = np.array([[x, y], [x, y]])
        return np.subtract(Tips, a)

    @staticmethod
    def getEdgesFromVertices(V):
        nbE = np.shape(V)[0]
        sides = np.zeros((2*nbE, 2), dtype = int)
        for i in range(nbE):
            sides[2*i] = V[i]
            sides[2*i+1] = V[i]
        sides = np.reshape(np.roll(sides, -1, axis = 0), (nbE, 2, 2))
        return sides
    
    # Test if pixel c is on gamma_p of segment s.
    # c is coordinates of a pixel in range [0, resolution).
    # s is a segment.
    def testMembershipGamma_pEtPixel(self, c, s):
        unit = 1/(self.resolution)
        carre = np.array([[(c[0])*unit  , (c[1])*unit  ],
                          [(c[0]+1)*unit, (c[1])*unit  ],
                          [(c[0]+1)*unit, (c[1]+1)*unit],
                          [(c[0])*unit  , (c[1]+1)*unit]])
        s01 = TorusCurvePolygone.gpToGp01(s)
        r = TorusCurvePolygone.testIntersectionSegmentSquare(s01, carre)
        return r

    def findMembershipOfPixel(self, c):
        m_list = []
        counter = 0
        for i in self.gamma_p:
            if self.testMembershipGamma_pEtPixel(c, i):
                m_list.append(counter)
            counter = counter+1
        return m_list

    # Returns the boundary box.
    def bbox(self):
        x_min, y_min = self.vertices.min(axis=0)
        x_max, y_max = self.vertices.max(axis=0)
        return floor(x_min)-1, floor(y_min)-1, ceil(x_max)+1, ceil(y_max)+1


    # Returns the boundary points of Polygon.
    def getGridBoundary(self):
        b_list = []
        (x_min, y_min, x_max, y_max) = self.bbox()
        for y in range(y_min, y_max):
            for x in range(x_min, x_max):
                if (self.isBoundary([x, y], self.vertices)):
                    b_list.append([x, y])
        return np.array(b_list)
    
    # after this point functions are not complete
    def list_gamma_p(self):
        gp_list = np.empty((0,2,2))
        

        #nbE = np.shape(self.vertices)[0]
        sides = TorusCurvePolygone.getEdgesFromVertices(self.vertices)

        for i in self.boundary_points:
            gp_list_aux = np.empty((0,2))
            carre = np.array([[i[0]  , i[1]  ],
                              [i[0]+1, i[1]  ],
                              [i[0]+1, i[1]+1],
                              [i[0]  , i[1]+1]])
            for j in sides:
                intersectingSides = TorusCurvePolygone.testIntersectionSegmentSquare(j, carre)
                for k in range(4):
                    if (intersectingSides & 1 << k) :
                        gp_list_aux = np.append(gp_list_aux, [TorusCurvePolygone.intersection(j, [carre[k], carre[(k+1)%4]])], axis = 0)
            if len(gp_list_aux) == 1 :
                gp_list = np.append(gp_list, [np.append(gp_list_aux, gp_list_aux, axis = 0)], axis = 0)
            else:
                gp_list = np.append(gp_list, [gp_list_aux], axis = 0)
        return gp_list

            
    def draw_line(self, t, segment, current):
        taux = np.copy(t)
        for i in range(-1, 2):
            for j in range(-1, 2):
                if((i != 0 or j != 0) and (current[0]+i >= 0) and (current[1]+j >= 0) and (current[0]+i < self.resolution) and (current[1]+j < self.resolution) and (self.testMembershipGamma_pEtPixel(([current[0]+i, current[1]+j]), segment)) and (t[current[0]+i][current[1]+j] == 0)):
                    taux[current[0]][current[1]] = 255    
                    taux = self.draw_line(taux, segment, [current[0]+i, current[1]+j] )
        return taux
        
    def draw_torus(self):
        t = np.zeros((self.resolution, self.resolution), dtype = np.int16)
        for i in self.gamma_p:
            taux = np.zeros((self.resolution, self.resolution), dtype = np.int64)
            a = TorusCurvePolygone.gpToGp01(i)
            [c1, c2] = [floor(a[0][0]*self.resolution), floor(a[0][1]*self.resolution)]
            [c3, c4] = [floor(a[1][0]*self.resolution), floor(a[1][1]*self.resolution)]
            if c1 == self.resolution:
                c1 -= 1
            if c2 == self.resolution:
                c2 -= 1
            if c3 == self.resolution:
                c3 -= 1
            if c4 == self.resolution:
                c4 -= 1

            taux[c1][c2] = 255
            taux[c3][c4] = 255
            taux = self.draw_line(taux, i, [c1, c2])

            t = np.maximum(t, taux)
        return t


if __name__ == '__main__':
    
    vertices = np.array([[-2, 0], [-1, 1], [2, 0], [0, -1]])
    res = 512
    quad = TorusCurvePolygone(vertices=vertices, resolution=res)
    # torus = np.zeros((res, res))
    
    # for i in range(res):
    #     for j in range(res):
    #         c = np.array([i, j])
    #         r = quad.findMembershipOfPixel(c)
    #         if len(r) != 0:
    #             torus[i][j] = 255
    
    print(quad.gamma_p)
    plt.imshow(quad.torus)
    plt.show()