# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mapiranjeQ9 as mp


class Sistem:

    def __init__(self):
        df = pd.read_csv('ElementiMK.csv', encoding='UTF-8', sep=';', index_col=None)
        niz_parametara = df['Vrednost'].to_numpy()
        self.E = float(niz_parametara[0])
        self.ni = float(niz_parametara[1])
        self.t = float(niz_parametara[2])
        self.a = float(niz_parametara[3])
        self.b = float(niz_parametara[4])
        self.niz_elemenata = df['Elementi'].to_numpy()
        self.matrica_krutosti = self.niz_elemenata.reshape(18, 18)
        self.mp = mp.KoordinateTacaka()

    def index(self):
        matrica = np.array([
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],  # KE 1
            [3, 4, 19, 20, 21, 22, 5, 6, 23, 24, 25, 26, 27, 28, 11, 12, 29, 30],  # KE 2
            [19, 20, 31, 32, 33, 34, 21, 22, 35, 36, 37, 38, 39, 40, 25, 26, 41, 42],  # KE 3
            [31, 32, 43, 44, 45, 46, 33, 34, 47, 48, 49, 50, 51, 52, 37, 38, 53, 54]  # KE 4
            ])
        return matrica

    def E_matrica(self):
        Ematrica = np.array([[self.E / (1 - self.ni ** 2), self.ni * self.E / (1 - self.ni ** 2), 0],
                                   [self.ni * self.E / (1 - self.ni ** 2), self.E / (1 - self.ni ** 2), 0],
                                   [0, 0, (1 - self.ni) / 2 * self.E / (1 - self.ni ** 2)]
                                   ])
        return Ematrica

    def index2x(self):
        matrica = np.array([
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],  # KE 1
            [3, 4, 19, 20, 21, 22, 5, 6, 23, 24, 25, 26, 27, 28, 11, 12, 29, 30],  # KE 2
            [19, 20, 31, 32, 33, 34, 21, 22, 35, 36, 37, 38, 39, 40, 25, 26, 41, 42],  # KE 3
            [31, 32, 43, 44, 45, 46, 33, 34, 47, 48, 49, 50, 51, 52, 37, 38, 53, 54],  # KE 4
            [43, 44, 55, 56, 57, 58, 45, 46, 59, 60, 61, 62, 63, 64, 49, 50, 65, 66],  # KE 5
            [55, 56, 67, 68, 69, 70, 57, 58, 71, 72, 73, 74, 75, 76, 61, 62, 77, 78],  # KE 6
            [67, 68, 79, 80, 81, 82, 69, 70, 83, 84, 85, 86, 87, 88, 73, 74, 89, 90],  # KE 7
            [79, 80, 91, 92, 93, 94, 81, 82, 95, 96, 97, 98, 99, 100, 85, 86, 101, 102],  # KE 8
            [103, 104, 105, 106, 3, 4, 1, 2, 107, 108, 109, 110, 9, 10, 111, 112, 113, 114],  # KE 9
            [105, 106, 115, 116, 19, 20, 3, 4, 117, 118, 119, 120, 23, 24, 109, 110, 121, 122],   # KE 10
            [115, 116, 123, 124, 31, 32, 19, 20, 125, 126, 127, 128, 35, 36, 119, 120, 129, 130],  # KE 11
            [123, 124, 131, 132, 43, 44, 31, 32, 133, 134, 135, 136, 47, 48, 127, 128, 137, 138],  # KE 12
            [131, 132, 139, 140, 55, 56, 43, 44, 141, 142, 143, 144, 59, 60, 135, 136, 145, 146],  # KE 13
            [139, 140, 147, 148, 67, 68, 55, 56, 149, 150, 151, 152, 71, 72, 143, 144, 153, 154],  # KE 14
            [147, 148, 155, 156, 79, 80, 67, 68, 157, 158, 159, 160, 83, 84, 151, 152, 161, 162],  # KE 15
            [155, 156, 163, 164, 91, 92, 79, 80, 165, 166, 167, 168, 95, 96, 159, 160, 169, 170],  # KE 16
        ])
        return matrica

    def indeksiranematrice2x(self):
        matrica_krutosti = self.niz_elemenata.reshape(18, 18)
        #indeks_matrica = pd.DataFrame(self.index())  # 4 KE
        #konacni_elementi = np.arange(0, 4)  # 4 KE
        indeks_matrica = pd.DataFrame(self.index2x())  # 16 KE
        konacni_elementi = np.arange(0, 16)  # 16 KE
        niz_matrica = []
        for i in konacni_elementi:
            indeks_matrice_i = np.array(indeks_matrica.loc[i])
            matrica = pd.DataFrame(matrica_krutosti, index=indeks_matrice_i, columns=indeks_matrice_i)
            niz_matrica.append(matrica)
        return niz_matrica

    def indeksiranematrice(self):
        matrica_krutosti = self.niz_elemenata.reshape(18, 18)
        indeks_matrica = pd.DataFrame(self.index())  # 4 KE
        konacni_elementi = np.arange(0, 4)  # 4 KE
        niz_matrica = []
        for i in konacni_elementi:
            indeks_matrice_i = np.array(indeks_matrica.loc[i])
            matrica = pd.DataFrame(matrica_krutosti, index=indeks_matrice_i, columns=indeks_matrice_i)
            niz_matrica.append(matrica)
        return niz_matrica

    def ksystem2x(self):
        #pomeranja = np.array(np.arange(1, 55))  # 4 KE
        #ksistema = pd.DataFrame(np.zeros((54, 54)), columns=pomeranja, index=pomeranja) # 4 KE
        pomeranja = np.array(np.arange(1, 171))
        ksistema = pd.DataFrame(np.zeros((170, 170)), columns=pomeranja, index=pomeranja)
        matrice_krutosti = self.indeksiranematrice2x()
        for m in matrice_krutosti:
            for i in ksistema.index:
                for j in ksistema.columns:
                    try:
                        ksistema.loc[i][j] = ksistema.loc[i][j] + m.loc[i][j]
                    except Exception:
                        pass
        return ksistema

    def ksystem(self):
        pomeranja = np.array(np.arange(1, 55))  # 4 KE
        ksistema = pd.DataFrame(np.zeros((54, 54)), columns=pomeranja, index=pomeranja) # 4 KE
        matrice_krutosti = self.indeksiranematrice()
        for m in matrice_krutosti:
            for i in ksistema.index:
                for j in ksistema.columns:
                    try:
                        ksistema.loc[i][j] = ksistema.loc[i][j] + m.loc[i][j]
                    except Exception:
                        pass
        return ksistema

    def unknown_displacements2x(self):
        pomeranja = np.array(np.arange(1, 171))  # 16 KE
        vektor_cvornih_sila = np.array(np.zeros(170), dtype=np.int64)  # 16 KE
        cvorne_sile = pd.Series(vektor_cvornih_sila, index=pomeranja)
        for i in cvorne_sile.index:
            if i == 92:  # i za 4 KE je 50
                cvorne_sile.loc[92] = cvorne_sile.loc[92] - 40
        poznata_pomeranja = np.array([1, 2, 15, 16, 7, 8, 111, 112, 103, 104])
        nepoznata_pomeranja = pd.Series(pomeranja, index=pomeranja).drop(poznata_pomeranja)
        pnn = cvorne_sile.drop(poznata_pomeranja)
        knn = self.ksystem2x().drop(index=poznata_pomeranja, columns=poznata_pomeranja)
        qnn = np.dot(np.linalg.inv(knn), pnn)
        qnn = pd.Series(qnn, index=nepoznata_pomeranja)
        return qnn

    def unknown_displacements(self):
        vektor_cvornih_sila = np.array(np.zeros(54), dtype=np.int64)  # 4 KE
        pomeranja = np.array(np.arange(1, 55))  # 4 KE
        cvorne_sile = pd.Series(vektor_cvornih_sila, index=pomeranja)
        for i in cvorne_sile.index:
            if i == 50:  # i za 4 KE je 50
                cvorne_sile.loc[50] = cvorne_sile.loc[50] - 40
        poznata_pomeranja = np.array([1, 2, 15, 16, 7, 8])  # 4 KE
        nepoznata_pomeranja = pd.Series(pomeranja, index=pomeranja).drop(poznata_pomeranja)
        pnn = cvorne_sile.drop(poznata_pomeranja)
        knn = self.ksystem().drop(index=poznata_pomeranja, columns=poznata_pomeranja)
        qnn = np.dot(np.linalg.inv(knn), pnn)
        qnn = pd.Series(qnn, index=nepoznata_pomeranja)
        return qnn

    def displacements2x(self):
        qnn = self.unknown_displacements2x()
        #q = pd.Series(np.zeros(54), index=np.arange(1, 55))  #  4 KE
        q = pd.Series(np.zeros(170), index=np.arange(1, 171)) #  16 KE
        for i in q.index:
            try:
                q.loc[i] = q.loc[i] + qnn.loc[i]
            except Exception:
                pass
        return q

    def displacements(self):
        qnn = self.unknown_displacements2x()
        q = pd.Series(np.zeros(54), index=np.arange(1, 55))  #  4 KE
        for i in q.index:
            try:
                q.loc[i] = q.loc[i] + qnn.loc[i]
            except Exception:
                pass
        return q

    def stress2x(self, ke, tacka, q, napon):
        indeksi = self.index2x()
        Q = [q[j] for j in indeksi[ke - 1]]
        X = [Q[2 * j] for j in range(9)]
        Y = [Q[2 * j + 1] for j in range(9)]
        a = self.a
        b = self.b
        qx1 = X[0]
        qx2 = X[1]
        qx3 = X[2]
        qx4 = X[3]
        qx5 = X[4]
        qx6 = X[5]
        qx7 = X[6]
        qx8 = X[7]
        qx9 = X[8]
        qy1 = Y[0]
        qy2 = Y[1]
        qy3 = Y[2]
        qy4 = Y[3]
        qy5 = Y[4]
        qy6 = Y[5]
        qy7 = Y[6]
        qy8 = Y[7]
        qy9 = Y[8]
        epsilon1 = eval(tacka[0])
        epsilon2 = eval(tacka[1])
        epsilon3 = eval(tacka[2])
        konacno = np.dot(self.E_matrica(), np.array([[epsilon1], [epsilon2], [epsilon3]]))
        konacno = konacno[napon][0]
        return konacno

    def stress(self, ke, tacka, q, napon):
        indeksi = self.index()
        Q = [q[j] for j in indeksi[ke - 1]]
        X = [Q[2 * j] for j in range(9)]
        Y = [Q[2 * j + 1] for j in range(9)]
        a = self.a
        b = self.b
        qx1 = X[0]
        qx2 = X[1]
        qx3 = X[2]
        qx4 = X[3]
        qx5 = X[4]
        qx6 = X[5]
        qx7 = X[6]
        qx8 = X[7]
        qx9 = X[8]
        qy1 = Y[0]
        qy2 = Y[1]
        qy3 = Y[2]
        qy4 = Y[3]
        qy5 = Y[4]
        qy6 = Y[5]
        qy7 = Y[6]
        qy8 = Y[7]
        qy9 = Y[8]
        epsilon1 = eval(tacka[0])
        epsilon2 = eval(tacka[1])
        epsilon3 = eval(tacka[2])
        konacno = np.dot(self.E_matrica(), np.array([[epsilon1], [epsilon2], [epsilon3]]))
        konacno = konacno[napon][0]
        return konacno

    def average(self, a, b):
        prosek = (a + b)/2
        return prosek

    def plot2x(self):
        q = self.displacements2x()
        kt = mp.KoordinateTacaka()
        pomeranja = pd.Series(q, index=np.arange(1, 171))
        #q = pd.Series(q, index=np.arange(1, 55))
        #q = [q.loc[16], q.loc[18], q.loc[12], q.loc[30], q.loc[26], q.loc[42], q.loc[38], q.loc[54], q.loc[50]]
        ugib = np.array([pomeranja.loc[2], pomeranja.loc[10], pomeranja.loc[4], pomeranja.loc[24], pomeranja.loc[20], pomeranja.loc[36],
                pomeranja.loc[32], pomeranja.loc[48], pomeranja.loc[44], pomeranja.loc[60], pomeranja.loc[56], pomeranja.loc[72],
                pomeranja.loc[68], pomeranja.loc[84], pomeranja.loc[80], pomeranja.loc[96], pomeranja.loc[92]])*1000
        sigma_x = np.array([self.stress2x(ke=1, tacka=kt.epsilon4(), q=q, napon=0),
                   self.stress2x(ke=1, tacka=kt.epsilon7(), q=q, napon=0),
                   self.average(a=self.stress2x(ke=1, tacka=kt.epsilon3(), q=q, napon=0),
                                b=self.stress2x(ke=2, tacka=kt.epsilon4(), q=q, napon=0)),
                   self.stress2x(ke=2, tacka=kt.epsilon7(), q=q, napon=0),
                   self.average(a=self.stress2x(ke=2, tacka=kt.epsilon3(), q=q, napon=0),
                                b=self.stress2x(ke=3, tacka=kt.epsilon4(), q=q, napon=0)),
                   self.stress2x(ke=3, tacka=kt.epsilon7(), q=q, napon=0),
                   self.average(a=self.stress2x(ke=3, tacka=kt.epsilon3(), q=q, napon=0),
                                b=self.stress2x(ke=4, tacka=kt.epsilon4(), q=q, napon=0)),
                   self.stress2x(ke=4, tacka=kt.epsilon7(), q=q, napon=0),
                   self.average(a=self.stress2x(ke=4, tacka=kt.epsilon3(), q=q, napon=0),
                                b=self.stress2x(ke=5, tacka=kt.epsilon4(), q=q, napon=0)),
                   self.stress2x(ke=5, tacka=kt.epsilon7(), q=q, napon=0),
                   self.average(a=self.stress2x(ke=5, tacka=kt.epsilon3(), q=q, napon=0),
                                b=self.stress2x(ke=6, tacka=kt.epsilon4(), q=q, napon=0)),
                   self.stress2x(ke=6, tacka=kt.epsilon7(), q=q, napon=0),
                   self.average(a=self.stress2x(ke=6, tacka=kt.epsilon3(), q=q, napon=0),
                                b=self.stress2x(ke=7, tacka=kt.epsilon4(), q=q, napon=0)),
                   self.stress2x(ke=7, tacka=kt.epsilon7(), q=q, napon=0),
                   self.average(a=self.stress2x(ke=7, tacka=kt.epsilon3(), q=q, napon=0),
                                b=self.stress2x(ke=8, tacka=kt.epsilon4(), q=q, napon=0)),
                   self.stress2x(ke=8, tacka=kt.epsilon7(), q=q, napon=0),
                   self.stress2x(ke=8, tacka=kt.epsilon3(), q=q, napon=0)
                   ])
        sigma_y = np.array([self.stress2x(ke=1, tacka=kt.epsilon4(), q=q, napon=1),
                   self.stress2x(ke=1, tacka=kt.epsilon7(), q=q, napon=1),
                   self.average(a=self.stress2x(ke=1, tacka=kt.epsilon3(), q=q, napon=1),
                                b=self.stress2x(ke=2, tacka=kt.epsilon4(), q=q, napon=1)),
                   self.stress2x(ke=2, tacka=kt.epsilon7(), q=q, napon=1),
                   self.average(a=self.stress2x(ke=2, tacka=kt.epsilon3(), q=q, napon=1),
                                b=self.stress2x(ke=3, tacka=kt.epsilon4(), q=q, napon=1)),
                   self.stress2x(ke=3, tacka=kt.epsilon7(), q=q, napon=1),
                   self.average(a=self.stress2x(ke=3, tacka=kt.epsilon3(), q=q, napon=1),
                                b=self.stress2x(ke=4, tacka=kt.epsilon4(), q=q, napon=1)),
                   self.stress2x(ke=4, tacka=kt.epsilon7(), q=q, napon=1),
                   self.average(a=self.stress2x(ke=4, tacka=kt.epsilon3(), q=q, napon=1),
                                b=self.stress2x(ke=5, tacka=kt.epsilon4(), q=q, napon=1)),
                   self.stress2x(ke=5, tacka=kt.epsilon7(), q=q, napon=1),
                   self.average(a=self.stress2x(ke=5, tacka=kt.epsilon3(), q=q, napon=1),
                                b=self.stress2x(ke=6, tacka=kt.epsilon4(), q=q, napon=1)),
                   self.stress2x(ke=6, tacka=kt.epsilon7(), q=q, napon=1),
                   self.average(a=self.stress2x(ke=6, tacka=kt.epsilon3(), q=q, napon=1),
                                b=self.stress2x(ke=7, tacka=kt.epsilon4(), q=q, napon=1)),
                   self.stress2x(ke=7, tacka=kt.epsilon7(), q=q, napon=1),
                   self.average(a=self.stress2x(ke=7, tacka=kt.epsilon3(), q=q, napon=1),
                                b=self.stress2x(ke=8, tacka=kt.epsilon4(), q=q, napon=1)),
                   self.stress2x(ke=8, tacka=kt.epsilon7(), q=q, napon=1),
                   self.stress2x(ke=8, tacka=kt.epsilon3(), q=q, napon=1)
                   ])
        tau_xy = np.array([self.stress2x(ke=1, tacka=kt.epsilon4(), q=q, napon=2),
                   self.stress2x(ke=1, tacka=kt.epsilon7(), q=q, napon=2),
                   self.average(a=self.stress2x(ke=1, tacka=kt.epsilon3(), q=q, napon=2),
                                b=self.stress2x(ke=2, tacka=kt.epsilon4(), q=q, napon=2)),
                   self.stress2x(ke=2, tacka=kt.epsilon7(), q=q, napon=2),
                   self.average(a=self.stress2x(ke=2, tacka=kt.epsilon3(), q=q, napon=2),
                                b=self.stress2x(ke=3, tacka=kt.epsilon4(), q=q, napon=2)),
                   self.stress2x(ke=3, tacka=kt.epsilon7(), q=q, napon=2),
                   self.average(a=self.stress2x(ke=3, tacka=kt.epsilon3(), q=q, napon=2),
                                b=self.stress2x(ke=4, tacka=kt.epsilon4(), q=q, napon=2)),
                   self.stress2x(ke=4, tacka=kt.epsilon7(), q=q, napon=2),
                   self.average(a=self.stress2x(ke=4, tacka=kt.epsilon3(), q=q, napon=2),
                                b=self.stress2x(ke=5, tacka=kt.epsilon4(), q=q, napon=2)),
                   self.stress2x(ke=5, tacka=kt.epsilon7(), q=q, napon=2),
                   self.average(a=self.stress2x(ke=5, tacka=kt.epsilon3(), q=q, napon=2),
                                b=self.stress2x(ke=6, tacka=kt.epsilon4(), q=q, napon=2)),
                   self.stress2x(ke=6, tacka=kt.epsilon7(), q=q, napon=2),
                   self.average(a=self.stress2x(ke=6, tacka=kt.epsilon3(), q=q, napon=2),
                                b=self.stress2x(ke=7, tacka=kt.epsilon4(), q=q, napon=2)),
                   self.stress2x(ke=7, tacka=kt.epsilon7(), q=q, napon=2),
                   self.average(a=self.stress2x(ke=7, tacka=kt.epsilon3(), q=q, napon=2),
                                b=self.stress2x(ke=8, tacka=kt.epsilon4(), q=q, napon=2)),
                   self.stress2x(ke=8, tacka=kt.epsilon7(), q=q, napon=2),
                   self.stress2x(ke=8, tacka=kt.epsilon3(), q=q, napon=2)
                   ])
        #x_osa = np.linspace(0, 5, num=9)  # 4 KE
        x_osa = np.linspace(0, 5, num=17)  # 16 KE
        #plt.figure(1, figsize=(15, 5))
        fig, axs = plt.subplots(2, 2)
        axs[0, 0].plot(x_osa, ugib)
        axs[0, 0].set_title('Ugib duz grede [mm]')
        axs[0, 0].set_xlabel('L [m]')
        axs[0, 0].set_xticks(np.arange(0, 6))
        axs[0, 1].plot(x_osa, sigma_x)
        axs[0, 1].set_title('\u03C3x [KPa]')
        axs[0, 1].set_xlabel('L [m]')
        axs[0, 1].set_xticks(np.arange(0, 6))
        axs[1, 0].plot(x_osa, sigma_y)
        axs[1, 0].set_title('\u03C3y [KPa]')
        axs[1, 0].set_xlabel('L [m]')
        axs[1, 0].set_xticks(np.arange(0, 6))
        axs[1, 1].plot(x_osa, tau_xy)
        axs[1, 1].set_title('\u03C4xy [KPa]')
        axs[1, 1].set_xlabel('L [m]')
        axs[1, 1].set_xticks(np.arange(0, 6))
        fig.tight_layout()
        plt.show()

    def plot(self):
        q = self.displacements()
        kt = mp.KoordinateTacaka()
        q = pd.Series(q, index=np.arange(1, 55))
        ugib = np.array([q.loc[16], q.loc[18], q.loc[12], q.loc[30], q.loc[26], q.loc[42], q.loc[38], q.loc[54], q.loc[50]], dtype=float)*1000
        sigma_x = np.array([self.stress(ke=1, tacka=kt.epsilon4(), q=q, napon=0),
                            self.stress(ke=1, tacka=kt.epsilon7(), q=q, napon=0),
                            self.average(a=self.stress(ke=1, tacka=kt.epsilon3(), q=q, napon=0),
                                         b=self.stress(ke=2, tacka=kt.epsilon4(), q=q, napon=0)),
                            self.stress(ke=2, tacka=kt.epsilon7(), q=q, napon=0),
                            self.average(a=self.stress(ke=2, tacka=kt.epsilon3(), q=q, napon=0),
                                         b=self.stress(ke=3, tacka=kt.epsilon4(), q=q, napon=0)),
                            self.stress(ke=3, tacka=kt.epsilon7(), q=q, napon=0),
                            self.average(a=self.stress(ke=3, tacka=kt.epsilon3(), q=q, napon=0),
                                         b=self.stress(ke=4, tacka=kt.epsilon4(), q=q, napon=0)),
                            self.stress(ke=4, tacka=kt.epsilon7(), q=q, napon=0),
                            self.stress(ke=4, tacka=kt.epsilon3(), q=q, napon=0)
                            ])
        sigma_y = np.array([self.stress(ke=1, tacka=kt.epsilon4(), q=q, napon=1),
                            self.stress(ke=1, tacka=kt.epsilon7(), q=q, napon=1),
                            self.average(a=self.stress(ke=1, tacka=kt.epsilon3(), q=q, napon=1),
                                         b=self.stress(ke=2, tacka=kt.epsilon4(), q=q, napon=1)),
                            self.stress(ke=2, tacka=kt.epsilon7(), q=q, napon=1),
                            self.average(a=self.stress(ke=2, tacka=kt.epsilon3(), q=q, napon=1),
                                         b=self.stress(ke=3, tacka=kt.epsilon4(), q=q, napon=1)),
                            self.stress(ke=3, tacka=kt.epsilon7(), q=q, napon=1),
                            self.average(a=self.stress(ke=3, tacka=kt.epsilon3(), q=q, napon=1),
                                         b=self.stress(ke=4, tacka=kt.epsilon4(), q=q, napon=1)),
                            self.stress(ke=4, tacka=kt.epsilon7(), q=q, napon=1),
                            self.stress(ke=4, tacka=kt.epsilon3(), q=q, napon=1)
                            ])
        tau_xy = np.array([self.stress(ke=1, tacka=kt.epsilon4(), q=q, napon=2),
                           self.stress(ke=1, tacka=kt.epsilon7(), q=q, napon=2),
                           self.average(a=self.stress(ke=1, tacka=kt.epsilon3(), q=q, napon=2),
                                        b=self.stress(ke=2, tacka=kt.epsilon4(), q=q, napon=2)),
                           self.stress(ke=2, tacka=kt.epsilon7(), q=q, napon=2),
                           self.average(a=self.stress(ke=2, tacka=kt.epsilon3(), q=q, napon=2),
                                        b=self.stress(ke=3, tacka=kt.epsilon4(), q=q, napon=2)),
                           self.stress(ke=3, tacka=kt.epsilon7(), q=q, napon=2),
                           self.average(a=self.stress(ke=3, tacka=kt.epsilon3(), q=q, napon=2),
                                        b=self.stress(ke=4, tacka=kt.epsilon4(), q=q, napon=2)),
                           self.stress(ke=4, tacka=kt.epsilon7(), q=q, napon=2),
                           self.stress(ke=4, tacka=kt.epsilon3(), q=q, napon=2)
                           ])
        x_osa = np.linspace(0, 5, num=9)  # 4 KE
        fig, axs = plt.subplots(2, 2)
        axs[0, 0].plot(x_osa, ugib)
        axs[0, 0].set_title('Ugib duz grede [mm]')
        axs[0, 0].set_xlabel('L [m]')
        axs[0, 0].set_xticks(np.arange(0, 6))
        axs[0, 1].plot(x_osa, sigma_x)
        axs[0, 1].set_title('\u03C3x [KPa]')
        axs[0, 1].set_xlabel('L [m]')
        axs[0, 1].set_xticks(np.arange(0, 6))
        axs[1, 0].plot(x_osa, sigma_y)
        axs[1, 0].set_title('\u03C3y [KPa]')
        axs[1, 0].set_xlabel('L [m]')
        axs[1, 0].set_xticks(np.arange(0, 6))
        axs[1, 1].plot(x_osa, tau_xy)
        axs[1, 1].set_title('\u03C4xy [KPa]')
        axs[1, 1].set_xlabel('L [m]')
        axs[1, 1].set_xticks(np.arange(0, 6))
        fig.tight_layout()
        plt.show()

obj = Sistem()
#obj.plot2x()
obj.plot()
