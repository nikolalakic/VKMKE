# -*- coding: utf-8 -*-

import sympy as sp
import mapiranjeQ9 as mp


class MatricaKrutostiQ9:

    def __init__(self):
        self.x, self.y = sp.symbols('x y') # x i y su u stvari ksi i eta, vezane za izoparametarski koordinatni sistem!
        self.E, self.ni, self.t = sp.symbols('E ni t')
        self.a, self.b = sp.symbols('a b')
        self.mp = mp.KoordinateTacaka()
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


# Formiranje matrice B spajanjem matrica Bi:
    def Bmatrica(self):
        b = self.mp.B1().col_insert(2, self.mp.B2())
        b = b.col_insert(4, self.mp.B3())
        b = b.col_insert(6, self.mp.B4())
        b = b.col_insert(8, self.mp.B5())
        b = b.col_insert(10, self.mp.B6())
        b = b.col_insert(12, self.mp.B7())
        b = b.col_insert(14, self.mp.B8())
        b = b.col_insert(16, self.mp.B9())
        return b

# Transponovana B matrica:
    def BT(self):
        BT = sp.transpose(self.Bmatrica())
        return BT

# Formiranje podintegralne f-je i integracija:
    def integracija(self):
        BT = self.BT()
        B = self.Bmatrica()
        upit = str(input('Numericka ili simbolicka vrednost koeficijenta krutosti matrice? (recima unesi "numericka" ili "simbolicka", bez navodnika): '))
        upit = upit.lower()
        if upit == 'simbolicka':
            i = input('Unesi red matrice i: ')
            j = input('Unesi kolonu matrice j: ')
            if i == 'sve' and j == 'sve':
                podintegralna = BT*self.Ematrica*B*self.t*self.mp.JakobijanDet()
                integral = sp.integrate(sp.integrate(podintegralna, (self.x, -1, 1)), (self.y, -1, 1))
                print('\nK = ', integral)
            else:
                i = int(i)
                j = int(j)
                if i > 18 and j > 18:
                    print('\ni i j moraju biti manji od 18! ')
                    exit()
                else:
                    podintegralna = BT.row(i - 1)*self.Ematrica*B.col(j - 1)*self.t*self.mp.JakobijanDet()
                    integral = sp.integrate(sp.integrate(podintegralna, (self.x, -1, 1)), (self.y, -1, 1))
                    koeficijent_matrice = integral[0]
                    print(f'\nk{i}_{j} =', koeficijent_matrice)
        elif upit == 'numericka':
            i = input('Unesi red matrice i: ')
            j = input('Unesi kolonu matrice j: ')
            if i == 'sve' and j == 'sve':
                a = float(input('Unesi dimenziju konacnog elementa a [m]: '))
                b = float(input('Unesi dimenziju konacnog elementa b [m]: '))
                E = float(input('Unesi moduo elasticnosti E [GPa]: '))
                E = E * 10 ** 6
                ni = float(input('Unesi Poasonov koeficijent \u03BD: '))
                t = float(input('Unesi debljinu konacnog elementa t [m]: '))
                podintegralna = BT*self.Ematrica*B*self.t*self.mp.JakobijanDet()
                integral = sp.integrate(sp.integrate(podintegralna, (self.x, -1, 1)), (self.y, -1, 1))
                integral = integral.subs({self.E: E, self.ni: ni, self.t: t, self.a: a, self.b: b})
                print('\nK = ', integral)

            elif i != 'sve' and j != 'sve':
                i = int(i)
                j = int(j)
                if i > 18 or j > 18:
                    print('\ni i j moraju biti manji od 18! ')
                    exit()
                else:
                    a = float(input('Unesi dimenziju konacnog elementa a [m]: '))
                    b = float(input('Unesi dimenziju konacnog elementa b [m]: '))
                    E = float(input('Unesi moduo elasticnosti E [GPa]: '))
                    E = E*10**6
                    ni = float(input('Unesi Poasonov koeficijent \u03BD: '))
                    t = float(input('Unesi debljinu konacnog elementa t [m]: '))
                    podintegralna = BT.row(i - 1) * self.Ematrica * B.col(j - 1) * self.t*self.mp.JakobijanDet()
                    integral = sp.integrate(sp.integrate(podintegralna, (self.x, -1, 1)), (self.y, -1, 1))
                    integral = integral.subs({self.E: E, self.ni: ni, self.t: t, self.a: a, self.b: b})
                    koeficijent_matrice = integral[0]
                    print(f'\nk{i}_{j} =', koeficijent_matrice)
            else:
                print('\n i i j moraju biti rec "sve" ili broj manje ili jednak 18!')
                exit()
        else:
            print('\nUnesi lepo "simbolicka" ili "numericka" pri upitu.')
            exit()


obj = MatricaKrutostiQ9()
obj.integracija()
