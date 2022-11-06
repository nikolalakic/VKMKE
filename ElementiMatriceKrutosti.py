# -*- coding: utf-8 -*-

import sympy as sp
import mapiranjeQ9 as mp
import numpy as np
import pandas as pd
import os



class MatricaKrutostiQ9:

    def __init__(self):
        self.x, self.y = sp.symbols('x y') # x i y su u stvari ksi i eta, vezane za izoparametarski koordinatni sistem!
        self.E, self.ni, self.t = sp.symbols('E ni t')
        self.a, self.b = sp.symbols('a b')
        #Ravno stanje deformacije:
        #self.Ematrica = sp.Matrix([[self.E*(1-self.ni)/((1+self.ni)*(1-2*self.ni)), self.E*(1-self.ni)*self.ni/((1-self.ni**2)*(1-2*self.ni)), 0],
        # [self.E*(1-self.ni)*self.ni/((1-self.ni**2)*(1-2*self.ni)), self.E*(1-self.ni)/((1+self.ni)*(1-2*self.ni)), 0],
        # [0, 0, self.E/(2*(1 + self.ni))]
        # ])

        #Ravno stanje napona:
        self.Ematrica = sp.Matrix([[self.E/(1-self.ni**2), self.ni*self.E/(1-self.ni**2), 0],
                                   [self.ni*self.E/(1-self.ni**2), self.E/(1-self.ni**2), 0],
                                   [0, 0, (1-self.ni)/2*self.E/(1-self.ni**2)]
                                   ])

# Interpolacione funkcije:
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

# Jakobijeva matrica
    def Jakobijan(self):
        kt = mp.KoordinateTacaka() # ucitavanje klase iz skripte mapiranjeQ9.py
        niz_interpolacionih = sp.Matrix([self.N1(), self.N2(), self.N3(), self.N4(), self.N5(), self.N6(), self.N7(), self.N8(), self.N9()])
        x_niz = kt.x_niz()
        X = sp.transpose(x_niz)*niz_interpolacionih # Polje pomeranja u X pravcu
        X = X[0]
        y_niz = kt.y_niz()
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

# Formiranje matrice B spajanjem matrica Bi:
    def Bmatrica(self):
        b = self.B1().col_insert(2, self.B2())
        b = b.col_insert(4, self.B3())
        b = b.col_insert(6, self.B4())
        b = b.col_insert(8, self.B5())
        b = b.col_insert(10, self.B6())
        b = b.col_insert(12, self.B7())
        b = b.col_insert(14, self.B8())
        b = b.col_insert(16, self.B9())
        return b

# Transponovana B matrica:
    def BT(self):
        BT = sp.transpose(self.Bmatrica())
        return BT

    def Provera(self):
        fajlovi = os.listdir()
        if 'ElementiMatriceKrutosti.csv' in fajlovi:
            os.remove('ElementiMatriceKrutosti.csv')
            os.system('echo "Elementi" > ElementiMatriceKrutosti.csv')
        else:
            os.system('echo "Elementi" > ElementiMatriceKrutosti.csv')


# Formiranje podintegralne f-je i integracija:
    def integracija(self):
        self.Provera()
        BT = self.BT()
        B = self.Bmatrica()
        K = np.array([], dtype=str)
        for i in np.arange(1,19):
            red= np.array([], dtype=str)
            for j in np.arange(1,19):
                podintegralna = BT.row(i - 1)*self.Ematrica*B.col(j - 1)*self.t*self.JakobijanDet()
                integral = sp.integrate(sp.integrate(podintegralna, (self.x, -1, 1)), (self.y, -1, 1))
                koeficijent_matrice = '=' + str(integral[0])
                koeficijent_matrice = koeficijent_matrice.replace('**','^')
                red = np.append(red, koeficijent_matrice)
            K = np.asarray(red)
            df = pd.DataFrame(K)
            df.to_csv('ElementiMatriceKrutosti.csv', encoding='UTF-8', mode='a', sep=';', header=False, index=False)

obj = MatricaKrutostiQ9()
obj.integracija()
