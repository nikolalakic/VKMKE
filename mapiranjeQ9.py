import sympy as sp
import numpy as np

class KoordinateTacaka:
    def __init__(self):
        #df = pd.read_csv('Q9Koordinate.csv', delimiter=';', encoding='UTF-8', skipinitialspace=True)
        #self.df = df
        self.a, self.b = sp.symbols('a b')
        self.x, self.y = sp.symbols('x y')
        self.qx1, self.qx2, self.qx3, self.qx4, self.qx5, self.qx6, self.qx7, self.qx8, self.qx9 = sp.symbols('qx1 qx2 qx3 qx4 qx5 qx6 qx7 qx8 qx9')
        self.qy1, self.qy2, self.qy3, self.qy4, self.qy5, self.qy6, self.qy7, self.qy8, self.qy9 = sp.symbols('qy1 qy2 qy3 qy4 qy5 qy6 qy7 qy8 qy9')

# Interpolacione funkcije

    def N9(self):
        N9 = (1 - self.x ** 2) * (1 - self.y ** 2)
        return N9

    def N8(self):
        N8 = 0.5 * (1 - self.x) * (1 - self.y ** 2) - 0.5 * self.N9()
        return N8

    def N7(self):
        N7 = 0.5 * (1 - self.x ** 2) * (1 + self.y) - 0.5 * self.N9()
        return N7

    def N6(self):
        N6 = 0.5 * (1 + self.x) * (1 - self.y ** 2) - 0.5 * self.N9()
        return N6

    def N5(self):
        N5 = 0.5 * (1 - self.x ** 2) * (1 - self.y) - 0.5 * self.N9()
        return N5

    def N4(self):
        N4 = 0.25 * (1 - self.x) * (1 + self.y) - 0.5 * self.N7() - 0.5 * self.N8() - 0.25 * self.N9()
        return N4

    def N3(self):
        N3 = 0.25 * (1 + self.x) * (1 + self.y) - 0.5 * self.N6() - 0.5 * self.N7() - 0.25 * self.N9()
        return N3

    def N2(self):
        N2 = 0.25 * (1 + self.x) * (1 - self.y) - 0.5 * self.N5() - 0.5 * self.N6()- 0.25 * self.N9()
        return N2

    def N1(self):
        N1 = 0.25 * (1 - self.x) * (1 - self.y) - 0.5 * self.N5() - 0.5 * self.N8() - 0.25 * self.N9()
        return N1

    def x_niz(self): # Formiranje koordinata tacaka Xi
        niz = sp.Matrix([0, self.a, self.a, 0, 0.5 * self.a, self.a, 0.5 * self.a, 0, 0.5 * self.a])
        return niz

    def y_niz(self): # Formiranje koordinata tacaka Yi
        niz = sp.Matrix([0, 0, self.b, self.b, 0, 0.5 * self.b, self.b, 0.5 * self.b, 0.5 * self.b])
        return niz

    def Jakobijan(self):
        niz_interpolacionih = sp.Matrix([self.N1(), self.N2(), self.N3(), self.N4(), self.N5(), self.N6(), self.N7(), self.N8(), self.N9()])
        x_niz = self.x_niz()
        X = sp.transpose(x_niz)*niz_interpolacionih # Polje pomeranja u X pravcu
        X = X[0]
        y_niz = self.y_niz()
        Y = sp.transpose(y_niz)*niz_interpolacionih # Polje pomeranja u Y pravcu
        Y = Y[0]
        J = sp.Matrix([[sp.diff(X, self.x), sp.diff(Y, self.x)], [sp.diff(X, self.y), sp.diff(Y, self.y)]])
        return J
    
    def JakobijanInv(self):
        J = self.Jakobijan()
        J = J.inv()
        return J

    def JakobijanDet(self):
        J = self.Jakobijan()
        J = J.det()
        return J
    
    #Kolone matrice B pri cemu su kolone matrice reda 3x2
    def B1(self):
        vektor = self.JakobijanInv()*sp.Matrix([[sp.diff(self.N1(), self.x)], [sp.diff(self.N1(), self.y)]])
        x = vektor[0]
        y = vektor[1]
        matrica = sp.Matrix([[x, 0], [0, y], [y, x]])
        return matrica

    def B2(self):
        vektor = self.JakobijanInv()*sp.Matrix([[sp.diff(self.N2(), self.x)], [sp.diff(self.N2(), self.y)]])
        x = vektor[0]
        y = vektor[1]
        matrica = sp.Matrix([[x, 0], [0, y], [y, x]])
        return matrica

    def B3(self):
        vektor = self.JakobijanInv()*sp.Matrix([[sp.diff(self.N3(), self.x)], [sp.diff(self.N3(), self.y)]])
        x = vektor[0]
        y = vektor[1]
        matrica = sp.Matrix([[x, 0], [0, y], [y, x]])
        return matrica

    def B4(self):
        vektor = self.JakobijanInv()*sp.Matrix([[sp.diff(self.N4(), self.x)], [sp.diff(self.N4(), self.y)]])
        x = vektor[0]
        y = vektor[1]
        matrica = sp.Matrix([[x, 0], [0, y], [y, x]])
        return matrica

    def B5(self):
        vektor = self.JakobijanInv()*sp.Matrix([[sp.diff(self.N5(), self.x)], [sp.diff(self.N5(), self.y)]])
        x = vektor[0]
        y = vektor[1]
        matrica = sp.Matrix([[x, 0], [0, y], [y, x]])
        return matrica

    def B6(self):
        vektor = self.JakobijanInv()*sp.Matrix([[sp.diff(self.N6(), self.x)], [sp.diff(self.N6(), self.y)]])
        x = vektor[0]
        y = vektor[1]
        matrica = sp.Matrix([[x,0], [0, y], [y, x]])
        return matrica

    def B7(self):
        vektor = self.JakobijanInv()*sp.Matrix([[sp.diff(self.N7(), self.x)],[sp.diff(self.N7(), self.y)]])
        x = vektor[0]
        y = vektor[1]
        matrica = sp.Matrix([[x, 0], [0, y], [y, x]])
        return matrica

    def B8(self):
        vektor = self.JakobijanInv()*sp.Matrix([[sp.diff(self.N8(), self.x)],[sp.diff(self.N8(), self.y)]])
        x = vektor[0]
        y = vektor[1]
        matrica = sp.Matrix([[x, 0], [0, y], [y, x]])
        return matrica
    
    def B9(self):
        vektor = self.JakobijanInv()*sp.Matrix([[sp.diff(self.N9(), self.x)], [sp.diff(self.N9(), self.y)]])
        x = vektor[0]
        y = vektor[1]
        matrica = sp.Matrix([[x, 0], [0, y], [y, x]])
        return matrica

    def polje_pomeranja_x(self):
        polje_pomeranja = self.N1()*self.qx1 + self.N2()*self.qx2 + self.N3()*self.qx3 + self.N4()*self.qx4 + self.N5()*self.qx5 + self.N6()*self.qx6 + self.N7()*self.qx7 + self.N8()*self.qx8 + self.N9()*self.qx9
        return polje_pomeranja

    def polje_pomeranja_y(self):
        polje_pomeranja = self.N1()*self.qy1 + self.N2()*self.qy2 + self.N3()*self.qy3 + self.N4()*self.qy4 + self.N5()*self.qy5 + self.N6()*self.qy6 + self.N7()*self.qy7 + self.N8()*self.qy8 + self.N9()*self.qy9
        return polje_pomeranja

    def epsilon1(self):
        Jakobijan = sp.Matrix([[sp.diff(self.polje_pomeranja_x(), self.x)], [sp.diff(self.polje_pomeranja_x(), self.y)], [sp.diff(self.polje_pomeranja_y(), self.x)], [sp.diff(self.polje_pomeranja_y(), self.y)]])
        inv_J = self.JakobijanInv()
        M = sp.Matrix([[inv_J[0], inv_J[1], 0, 0], [inv_J[2], inv_J[3], 0, 0], [0, 0, inv_J[0], inv_J[1]], [0, 0, inv_J[2], inv_J[3]]])
        t_matrica = sp.Matrix([[1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 1, 0]])
        epsilon = t_matrica*M*Jakobijan
        epsilon = epsilon.subs({self.x: -1, self.y: -1})
        epsilon = np.array([epsilon[0], epsilon[1], epsilon[2]], dtype=str)
        return epsilon

    def epsilon2(self):
        Jakobijan = sp.Matrix([[sp.diff(self.polje_pomeranja_x(), self.x)], [sp.diff(self.polje_pomeranja_x(), self.y)], [sp.diff(self.polje_pomeranja_y(), self.x)], [sp.diff(self.polje_pomeranja_y(), self.y)]])
        inv_J = self.JakobijanInv()
        M = sp.Matrix([[inv_J[0], inv_J[1], 0, 0], [inv_J[2], inv_J[3], 0, 0], [0, 0, inv_J[0], inv_J[1]], [0, 0, inv_J[2], inv_J[3]]])
        t_matrica = sp.Matrix([[1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 1, 0]])
        epsilon = t_matrica*M*Jakobijan
        epsilon = epsilon.subs({self.x: 1, self.y: -1})
        epsilon = np.array([epsilon[0], epsilon[1], epsilon[2]], dtype=str)
        return epsilon

    def epsilon3(self):
        Jakobijan = sp.Matrix([[sp.diff(self.polje_pomeranja_x(), self.x)], [sp.diff(self.polje_pomeranja_x(), self.y)], [sp.diff(self.polje_pomeranja_y(), self.x)], [sp.diff(self.polje_pomeranja_y(), self.y)]])
        inv_J = self.JakobijanInv()
        M = sp.Matrix([[inv_J[0], inv_J[1], 0, 0], [inv_J[2], inv_J[3], 0, 0], [0, 0, inv_J[0], inv_J[1]], [0, 0, inv_J[2], inv_J[3]]])
        t_matrica = sp.Matrix([[1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 1, 0]])
        epsilon = t_matrica*M*Jakobijan
        epsilon = epsilon.subs({self.x: 1, self.y: 1})
        epsilon = np.array([epsilon[0], epsilon[1], epsilon[2]], dtype=str)
        return epsilon

    def epsilon4(self):
        Jakobijan = sp.Matrix([[sp.diff(self.polje_pomeranja_x(), self.x)], [sp.diff(self.polje_pomeranja_x(), self.y)], [sp.diff(self.polje_pomeranja_y(), self.x)], [sp.diff(self.polje_pomeranja_y(), self.y)]])
        inv_J = self.JakobijanInv()
        M = sp.Matrix([[inv_J[0], inv_J[1], 0, 0], [inv_J[2], inv_J[3], 0, 0], [0, 0, inv_J[0], inv_J[1]], [0, 0, inv_J[2], inv_J[3]]])
        t_matrica = sp.Matrix([[1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 1, 0]])
        epsilon = t_matrica*M*Jakobijan
        epsilon = epsilon.subs({self.x: -1, self.y: 1})
        epsilon = np.array([epsilon[0], epsilon[1], epsilon[2]], dtype=str)
        return epsilon

    def epsilon5(self):
        Jakobijan = sp.Matrix([[sp.diff(self.polje_pomeranja_x(), self.x)], [sp.diff(self.polje_pomeranja_x(), self.y)], [sp.diff(self.polje_pomeranja_y(), self.x)], [sp.diff(self.polje_pomeranja_y(), self.y)]])
        inv_J = self.JakobijanInv()
        M = sp.Matrix([[inv_J[0], inv_J[1], 0, 0], [inv_J[2], inv_J[3], 0, 0], [0, 0, inv_J[0], inv_J[1]], [0, 0, inv_J[2], inv_J[3]]])
        t_matrica = sp.Matrix([[1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 1, 0]])
        epsilon = t_matrica*M*Jakobijan
        epsilon = epsilon.subs({self.x: 0, self.y: -1})
        epsilon = np.array([epsilon[0], epsilon[1], epsilon[2]], dtype=str)
        return epsilon

    def epsilon6(self):
        Jakobijan = sp.Matrix([[sp.diff(self.polje_pomeranja_x(), self.x)], [sp.diff(self.polje_pomeranja_x(), self.y)], [sp.diff(self.polje_pomeranja_y(), self.x)], [sp.diff(self.polje_pomeranja_y(), self.y)]])
        inv_J = self.JakobijanInv()
        M = sp.Matrix([[inv_J[0], inv_J[1], 0, 0], [inv_J[2], inv_J[3], 0, 0], [0, 0, inv_J[0], inv_J[1]], [0, 0, inv_J[2], inv_J[3]]])
        t_matrica = sp.Matrix([[1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 1, 0]])
        epsilon = t_matrica*M*Jakobijan
        epsilon = epsilon.subs({self.x: 1, self.y: 0})
        epsilon = np.array([epsilon[0], epsilon[1], epsilon[2]], dtype=str)
        return epsilon

    def epsilon7(self):
        Jakobijan = sp.Matrix([[sp.diff(self.polje_pomeranja_x(), self.x)], [sp.diff(self.polje_pomeranja_x(), self.y)], [sp.diff(self.polje_pomeranja_y(), self.x)], [sp.diff(self.polje_pomeranja_y(), self.y)]])
        inv_J = self.JakobijanInv()
        M = sp.Matrix([[inv_J[0], inv_J[1], 0, 0], [inv_J[2], inv_J[3], 0, 0], [0, 0, inv_J[0], inv_J[1]], [0, 0, inv_J[2], inv_J[3]]])
        t_matrica = sp.Matrix([[1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 1, 0]])
        epsilon = t_matrica*M*Jakobijan
        epsilon = epsilon.subs({self.x: 0, self.y: 1})
        epsilon = np.array([epsilon[0], epsilon[1], epsilon[2]], dtype=str)
        return epsilon

    def epsilon8(self):
        Jakobijan = sp.Matrix([[sp.diff(self.polje_pomeranja_x(), self.x)], [sp.diff(self.polje_pomeranja_x(), self.y)], [sp.diff(self.polje_pomeranja_y(), self.x)], [sp.diff(self.polje_pomeranja_y(), self.y)]])
        inv_J = self.JakobijanInv()
        M = sp.Matrix([[inv_J[0], inv_J[1], 0, 0], [inv_J[2], inv_J[3], 0, 0], [0, 0, inv_J[0], inv_J[1]], [0, 0, inv_J[2], inv_J[3]]])
        t_matrica = sp.Matrix([[1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 1, 0]])
        epsilon = t_matrica*M*Jakobijan
        epsilon = epsilon.subs({self.x: -1, self.y: 0})
        epsilon = np.array([epsilon[0], epsilon[1], epsilon[2]], dtype=str)
        return epsilon

    def epsilon9(self):
        Jakobijan = sp.Matrix([[sp.diff(self.polje_pomeranja_x(), self.x)], [sp.diff(self.polje_pomeranja_x(), self.y)], [sp.diff(self.polje_pomeranja_y(), self.x)], [sp.diff(self.polje_pomeranja_y(), self.y)]])
        inv_J = self.JakobijanInv()
        M = sp.Matrix([[inv_J[0], inv_J[1], 0, 0], [inv_J[2], inv_J[3], 0, 0], [0, 0, inv_J[0], inv_J[1]], [0, 0, inv_J[2], inv_J[3]]])
        t_matrica = sp.Matrix([[1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 1, 0]])
        epsilon = t_matrica*M*Jakobijan
        epsilon = epsilon.subs({self.x: 0, self.y: 0})
        epsilon = np.array([epsilon[0], epsilon[1], epsilon[2]], dtype=str)
        return epsilon



