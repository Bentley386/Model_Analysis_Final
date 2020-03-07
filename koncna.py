# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 21:35:07 2018

@author: Admin
"""
from xml.dom import minidom
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.animation as animation
import numpy as np
import matplotlib
import scipy.linalg as lin
import scipy
import scipy.integrate
import time
import matplotlib.pyplot as plt
from matplotlib import transforms

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
plt.rc("text",usetex=True)
matplotlib.rcParams["text.latex.unicode"] = True
plt.close("all")
pi = np.pi

class Plaketa():
    def __init__(self,levi,desni):
        self.levi = levi
        self.desni = desni
        self.sredina = (levi+desni)/2
        self.tangenta = desni - levi
        self.dolzina = lin.norm(self.tangenta)
        self.tangenta = self.tangenta/self.dolzina
        self.normala = np.array([-self.tangenta[1],self.tangenta[0]]) #prej tle minus
        self.ex = self.transformirajNaPlaketo(1,0,True)
        self.ey = self.transformirajNaPlaketo(0,1,True)
        self.napetost = 0
        
    def transformirajNaPlaketo(self,x,y,nitocka=False): #iz laboratorijskega sistema na plaketo
        if nitocka:
            vektor = np.array([x,y])
            #return self.transformirajNaPlaketo(x,y) - self.transformirajNaPlaketo(0,0)
            return np.array([np.dot(vektor,self.tangenta),np.dot(vektor,self.normala)])
        else:
            vektor = np.array([x,y]) - self.sredina 
        return np.array([np.dot(vektor,self.tangenta),np.dot(vektor,self.normala)]) #prva komp. minus prej
    
    def transformirajIzPlakete(self,x,y,nitocka=False): #obratno
        vektor2 = np.array([np.dot([x,y],self.ex),np.dot([x,y],self.ey)])
        if nitocka:
            return vektor2
        return vektor2 + self.sredina       



def narediPlakete(n,N,a,L): #naredi seznam objektov plaketa, n-st na elektrodo, N st elektrod, a razmik med elektrodami
    l = L/n
    levi = np.array([np.array([k1*a-0.5*L+ k2*l,0]) for k1 in range(-N,N+1) for k2 in range(n)])
    desni = np.array([np.array([i[0]+l,0]) for i in levi])
    return [Plaketa(levi[i],desni[i]) for i in range((2*N+1)*n)]
    #return np.array([-1/2 + i,0])
    
    
    
# ========================= Naivna metoda ===========================0    
    
    
def potencialZaradiPlakete(xx,yy,plaketa,N,sigma): #tista formula iz navodil
    #x =xx - (-1/2 + 1/(2*N) + i/N)
    x,y = plaketa.transformirajNaPlaketo(xx,yy)
    l = plaketa.dolzina
    if y==0:
        prvi = -l
    else:
        prvi = -l + y*(np.arctan((x+l/2)/y)-np.arctan((x-l/2)/y))
    ostali = (x+l/2)/2*np.log((x+l/2)**2+y*y) - (x-l/2)/2*np.log((x-l/2)**2+y*y)
    #return x+y+1/(2*N)
    return sigma/(2*pi)*(prvi+ostali)

def poljeZaradiPlakete(xx,yy,plaketa,N,sigma): #formula iz navodil
    #x = xx-(-1/2 + 1/(2*N)+i/N)
    x, y = plaketa.transformirajNaPlaketo(xx,yy)
    l = plaketa.dolzina
    xmin= x+l/2
    xpl = x-l/2
    if y==0:
        iks1 = 0
    else:
        iks1 = 1/(1+(xmin/y)**2) - 1/(1+(xpl/y)**2)
    iks2 = 0.5*np.log(xmin*xmin+y*y) + 1/(1+(y/xmin)**2) - 0.5*np.log(xpl*xpl+y*y) - 1/(1+(y/xpl)**2)
    iks = -sigma/(2*pi)*(iks1+iks2)
    if y==0:
        ips1 = pi/2*(np.sign(xmin)-np.sign(xpl))
    else:
        ips1 = np.arctan(xmin/y) - np.arctan(xpl/y)
    ips = -sigma/(2*pi)*ips1
    return np.array([iks,ips])

def najdiGostotePotencial(plakete,U): #rešitev sistema A*sigma = 1
    N = len(plakete)
    nicla = np.array([0,1])
    #U = np.ones(N+1)
    #U[-1]=0
    def naredi(i,j):
        if j==N:
            return 1
        if i==N:
            return potencialZaradiPlakete(nicla[0],nicla[1],plakete[j],N,1)            
        if i==j:
            l = plakete[i].dolzina
            return  (l*np.log(l/2)-l)/(2*pi)
        return potencialZaradiPlakete(plakete[i].sredina[0],plakete[i].sredina[1],plakete[j],N,1)
    A = np.fromfunction(np.vectorize(naredi),(N+1,N+1),dtype=int)
    return lin.solve(A,U)


def vrednost(x,y,naboji,konst,plakete,polje=False): #izračun potenciala/polja
    rezult = 0
    if polje:
        rezult = np.zeros(2)
    N = len(naboji)
    for i in range(N):
        if polje:
            temp = poljeZaradiPlakete(x,y,plakete[i],N,naboji[i])
            rezult+= plakete[i].transformirajIzPlakete(temp[0],temp[1],True)
            continue
        rezult+= potencialZaradiPlakete(x,y,plakete[i],N,naboji[i])
    if not polje:
        rezult += konst
    return rezult

# ================================================================

# ======================== Periodicna Greenova ============================

def potencialZaradiPlakete2(x,y,plaketa,N,a,sigma): #Periodicna greenova funkcija
    #x =xx - (-1/2 + 1/(2*N) + i/N)
    leva = plaketa.levi
    desna = plaketa.desni
    A=lin.norm(desna-leva)/(4*pi)
    
    def pot(s):
        xx = (1-s)*leva[0]+s*desna[0]
        yy = (1-s)*leva[1]+s*desna[1]
        arg = np.sinh(pi/a*(y-yy))**2 + np.sin(pi/a*(x-xx))**2
        if arg < 0.00000001:
            #return 0
            arg = 0.00000001
        return np.log(arg)

    return A*sigma*scipy.integrate.quad(pot,0,1)[0]

def poljeZaradiPlakete2(x,y,plaketa,N,a,sigma): #formula iz navodil
    #x = xx-(-1/2 + 1/(2*N)+i/N)
    leva = plaketa.levi
    desna = plaketa.desni
    A = -lin.norm(desna-leva)*0.25/a
    
    def polje(s,smer):
        xx = (1-s)*leva[0]+s*desna[0]
        yy = (1-s)*leva[1]+s*desna[1]
        
        if smer == 0: #X
            if abs(x-xx-a) < 0.0000001:
                return 0
            return np.sin(2*pi*(x-xx)/a)/(np.sinh(pi*(y-yy)/a)**2 + np.sin(pi*(x-xx)/a)**2)
        
        if smer == 1: #y
            if abs(y-yy) < 0.0000001:
                return 0          
            return np.sinh(2*pi*(y-yy)/a)/(np.sinh(pi*(y-yy)/a)**2+np.sin(pi*(x-xx)/a)**2)
        
    Ex = scipy.integrate.quad(polje,0,1,0)[0]
    Ey = scipy.integrate.quad(polje,0,1,1)[0]
    return A*sigma*np.array([Ex,Ey])

def najdiGostotePotencial2(plakete,U,a): #rešitev sistema A*sigma = 1
    N = len(plakete)
    nicla = np.array([0,0])
    def naredi(i,j):
        if j==N:
            return 1
        if i==N:
            return potencialZaradiPlakete2(nicla[0],nicla[1],plakete[j],N,a,1) 
        return potencialZaradiPlakete2(plakete[i].sredina[0],plakete[i].sredina[1],plakete[j],N,a,1)
    A = np.fromfunction(np.vectorize(naredi),(N+1,N+1),dtype=int)
    return lin.solve(A,U)

def vrednost2(x,y,naboji,konst,plakete,a,polje=False): #izračun potenciala/polja
    rezult = 0
    if polje:
        rezult = np.zeros(2)
    N = len(naboji)
    for i in range(N):
        if polje:
            temp = poljeZaradiPlakete2(x,y,plakete[i],N,a,naboji[i])
            rezult+= temp
            continue
        rezult+= potencialZaradiPlakete2(x,y,plakete[i],N,a,naboji[i])
    if not polje:
        rezult+=konst
    return rezult


#==========================================================================0
def plaketeFromSVG(src,skaliraj=0):
    doc = minidom.parse(src)
    path_strings = [path.getAttribute('d') for path
                    in doc.getElementsByTagName('path')]
    doc.unlink()
    plakete = []
    tocke = []
    #print(path_strings)
    

    z=False
    
    stevec=0
    premakni = [] #kdaj se pisalo samo premakne in ne nariše plakete
    
    for i in path_strings:
        path = i.split()
        current = [0,0]
        cifri = [0,0]
        premakni.append(stevec)
        for char in path[1:]:
            if char=="z" or char=="Z":
                z=True
                continue
            if char=="m" or char=="M":
                premakni.append(stevec)
                continue
            current=cifri
            cifri = list(map(float,char.split(",")))
            cifri = [cifri[0]+current[0],-cifri[1]+current[1]]
            tocke.append(cifri)
            stevec+=1

    tocke = np.array(tocke)
    #da je elektroda v okolici (0,0)
    iksi = tocke[:,0]-np.sum(tocke[:,0])/len(tocke[:,0])
    ipsi = tocke[:,1]-np.sum(tocke[:,1])/len(tocke[:,1])
    
    #da je elektroda reskalirana na kvadrat s stranico 10
    maxx = np.amax(np.abs(iksi))
    maxy = np.amax(np.abs(ipsi))
    if skaliraj==0:
        iksi = iksi*0.5/maxx
        ipsi = ipsi*1/maxy
    else:
        iksi = iksi*skaliraj/maxx
        ipsi = ipsi*skaliraj/maxy
    
    
    for i in range(0,len(tocke)-1):
        if i+1 in premakni:
            continue
        levi = np.array([iksi[i],ipsi[i]])
        desni = np.array([iksi[i+1],ipsi[i+1]])
        plakete.append(Plaketa(levi,desni))
        
    if z:
        plakete.append(Plaketa(np.array([iksi[-1],ipsi[-1]]),np.array([iksi[0],ipsi[0]])))
        
    return plakete   

def rotirajPlakete(plakete,kot):
    nove = []
    for plak in plakete:
        levi = np.array([np.cos(kot)*plak.levi[0] - np.sin(kot)*plak.levi[1],np.sin(kot)*plak.levi[0] + np.cos(kot)*plak.levi[1]])
        desni = np.array([np.cos(kot)*plak.desni[0] - np.sin(kot)*plak.desni[1],np.sin(kot)*plak.desni[0] + np.cos(kot)*plak.desni[1]])
        nove.append(Plaketa(levi,desni))
    return nove

def shiftajPlakete(plakete,x,y):
    nove = []
    vek= np.array([x,y])
    for plak in plakete:
        levi = plak.levi + vek
        desni = plak.desni + vek
        nove.append(Plaketa(levi,desni))
    return nove

if 1:
    #plot - all inclusive
    xlim=5
    ylim=4
    a = 5
    L = 3
    n = 50
    plak1 = shiftajPlakete(rotirajPlakete(plaketeFromSVG("CrkaF.svg"),pi/4*(np.random.rand()-pi/8)),-1.5,1.5)
    plak2 = shiftajPlakete(rotirajPlakete(plaketeFromSVG("CrkaM.svg"),pi/4*(np.random.rand()-pi/8)),0,1.5)
    plak3 = shiftajPlakete(rotirajPlakete(plaketeFromSVG("CrkaF.svg"),pi/4*(np.random.rand()-pi/8)),1.5,1.5)
    plakete = plak1+plak2+plak3
    for plak in plakete:
        plak.napetost = 1
    plak1 = shiftajPlakete(rotirajPlakete(plaketeFromSVG("CrkaF.svg"),pi/4*(np.random.rand()-pi/8)),-1.5,-1.5)
    plak2 = shiftajPlakete(rotirajPlakete(plaketeFromSVG("CrkaM.svg"),pi/4*(np.random.rand()-pi/8)),0,-1.5)
    plak3 = shiftajPlakete(rotirajPlakete(plaketeFromSVG("CrkaF.svg"),pi/4*(np.random.rand()-pi/8)),1.5,-1.5)
    plakete2 = plak1+plak2+plak3
    for plak in plakete2:
        plak.napetost = -1
    plaketke = plakete+plakete2
    naboji = najdiGostotePotencial2(plaketke,[p.napetost for p in plaketke] +[0],a)
    nabojcki = naboji[:-1]
    konst = naboji[-1]
    
    fig = plt.figure()
    x = np.linspace(-xlim,xlim,100)
    y = np.linspace(-ylim,ylim,100)
    X,Y = np.meshgrid(x,y)
    XX,YY = np.meshgrid(np.linspace(-xlim,xlim,20),np.linspace(-ylim,ylim,20))
    Z = np.array([[vrednost2(x,y,nabojcki,konst,plaketke,a) for x in np.linspace(-xlim,xlim,100)] for y in np.linspace(-ylim,ylim,100)])
    polja = np.array([[vrednost2(xx,yy,nabojcki,konst,plaketke,a,True) for xx in np.linspace(-xlim,xlim,20)] for yy in np.linspace(-ylim,ylim,20)])
    
    ax1 = fig.add_axes([0.05,0.05,0.65,0.65])
    cs = ax1.contourf(X,Y,Z,levels=np.linspace(np.min(Z),np.max(Z),50),cmap="hot")

# This is the fix for the white lines between contour levels
    for c in cs.collections:
        c.set_edgecolor("face")
    for plak in plaketke:
        if plak.napetost>0:
            barva = "m"
        else:
            barva = "c"
        for i in range(-3,3):
            ax1.plot([plak.levi[0]+i*a,plak.desni[0]+i*a],[plak.levi[1],plak.desni[1]],barva)

    ax1.quiver(XX,YY,polja[:,:,0],polja[:,:,1])
    
    ax1.set_xlim(-xlim,xlim)
    
    ax2 = fig.add_axes([0.05,0.7,0.65,0.3])
    
    ax2.plot(x,[vrednost2(i,0,nabojcki,konst,plaketke,a) for i in x],label=r"$y=0$")
    ax2.plot(x,[vrednost2(i,-1,nabojcki,konst,plaketke,a) for i in x],label=r"$y=-1$")
    ax2.plot(x,[vrednost2(i,-2,nabojcki,konst,plaketke,a) for i in x],label=r"$y=-2$")
    ax2.plot(x,[vrednost2(i,1,nabojcki,konst,plaketke,a) for i in x],label=r"$y=1$")
    ax2.plot(x,[vrednost2(i,2,nabojcki,konst,plaketke,a) for i in x],label=r"$y=2$")
    ax2.plot(x,[vrednost2(i,3,nabojcki,konst,plaketke,a) for i in x],label=r"$y=3$")
    ax2.plot(x,[vrednost2(i,-3,nabojcki,konst,plaketke,a) for i in x],label=r"$y=-3$")

    
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines["bottom"].set_visible(False)
    ax2.yaxis.set_ticks_position('left')
    ax2.get_xaxis().set_visible(False)    
    

    ax3 = fig.add_axes([0.7,0.05,0.3,0.65])
    base = ax3.transData
    rot = transforms.Affine2D().rotate_deg(270)

    ax3.plot(y,[vrednost2(0,i,nabojcki,konst,plaketke,a) for i in y],label=r"$x=0$",transform=rot+base)
    ax3.plot(y,[vrednost2(1,i,nabojcki,konst,plaketke,a) for i in y],label=r"$x=1$",transform=rot+base)
    ax3.plot(y,[vrednost2(2,i,nabojcki,konst,plaketke,a) for i in y],label=r"$x=2$",transform=rot+base)

    ax3.spines['right'].set_visible(False)
    ax3.spines['left'].set_visible(False)
    ax3.spines["top"].set_visible(False)
    #ax3.spines["bottom"].set_visible(False)
    
    #ax3.yaxis.set_ticks_position('bottom')
    ax3.get_yaxis().set_visible(False)    
    ax3.legend()
    handles,labels = ax2.get_legend_handles_labels() 
    labax = fig.add_axes([0.7,0.7,0.15,0.3])
    labax.legend(handles,labels,loc="center")
    labax.spines['right'].set_visible(False)
    labax.spines['left'].set_visible(False)
    labax.spines["top"].set_visible(False)
    labax.spines["bottom"].set_visible(False)
    labax.get_xaxis().set_visible(False)    
    labax.get_yaxis().set_visible(False)    
    
    cax = fig.add_axes([0.85, 0.7, 0.025, 0.3])   
    cb = plt.colorbar(cs,cax=cax)
    #cb.ax.locator_params(nbins=9)

    plt.savefig("vseFMF.pdf",dpi=300,transparent=True)
    plt.show()


if 0:
    #plot - all inclusive animacija
    n=20
    a=5
    L=3
    
    fig = plt.figure()
    x = np.linspace(-5,5,200)
    y = np.linspace(-5,5,200)
    X,Y = np.meshgrid(x,y)
    XX,YY = np.meshgrid(np.linspace(-5,5,20),np.linspace(-5,5,20))
    
    ax1 = fig.add_axes([0.05,0.05,0.65,0.65])
    
    ax2 = fig.add_axes([0.05,0.7,0.65,0.3])


    
    ax3 = fig.add_axes([0.7,0.05,0.3,0.65])
    base = ax3.transData
    rot = transforms.Affine2D().rotate_deg(90)


    labax = fig.add_axes([0.7,0.7,0.15,0.3])

    cax = fig.add_axes([0.85, 0.7, 0.025, 0.3])   
    #yshift = np.linspace(-0.1,0.1,20)
    koti = np.linspace(0,2*pi,40)
    #skale = [0.1*i for i in range(1,19)]
    def animiraj(t):
        print(t)
        ax1.clear()
        ax2.clear()
        ax3.clear()
        cax.clear()
        labax.clear()
        #ysh = yshift[t]
        kot = koti[t]
        #skala = skale[t]
        
        plakete = shiftajPlakete(rotirajPlakete(narediPlakete(10,0,5,3),kot),0,2)
        for plak in plakete:
            plak.napetost = 1
        plakete2 = shiftajPlakete(rotirajPlakete(narediPlakete(10,0,5,3),-kot),0,-2)

        for plak in plakete2:
            plak.napetost = -1
        plaketke = plakete+plakete2
        
        naboji = najdiGostotePotencial2(plaketke,[p.napetost for p in plaketke] +[0],a)
        nabojcki = naboji[:-1]
        konst = naboji[-1]
        Z = [[vrednost2(i,j,nabojcki,konst,plaketke,a) for i in x] for j in y]
        polja = np.array([[vrednost2(xx,yy,nabojcki,konst, plaketke,a,True) for xx in np.linspace(-5,5,20)] for yy in np.linspace(-5,5,20)])

        cs = ax1.contourf(X,Y,Z,levels=np.linspace(np.min(Z),np.max(Z),50),cmap="hot")
        # This is the fix for the white lines between contour levels
        for c in cs.collections:
            c.set_edgecolor("face")

        for plak in plaketke:
            if plak.napetost>0:
                barva = "m"
            else:
                barva = "c"
            for i in range(-3,3):
                ax1.plot([plak.levi[0]+i*a,plak.desni[0]+i*a],[plak.levi[1],plak.desni[1]],barva)
        ax1.quiver(XX,YY,polja[:,:,0],polja[:,:,1])
        ax1.set_xlim(-5,5)


    
        ax2.plot(x,[vrednost2(i,0,nabojcki,konst,plaketke,a) for i in x],label=r"$y=0$")
        ax2.plot(x,[vrednost2(i,-1,nabojcki,konst,plaketke,a) for i in x],label=r"$y=-1$")
        ax2.plot(x,[vrednost2(i,-2,nabojcki,konst,plaketke,a) for i in x],label=r"$y=-2$")
        ax2.plot(x,[vrednost2(i,1,nabojcki,konst,plaketke,a) for i in x],label=r"$y=1$")
        ax2.plot(x,[vrednost2(i,2,nabojcki,konst,plaketke,a) for i in x],label=r"$y=2$")
        ax2.plot(x,[vrednost2(i,3,nabojcki,konst,plaketke,a) for i in x],label=r"$y=3$")
        ax2.plot(x,[vrednost2(i,-3,nabojcki,konst,plaketke,a) for i in x],label=r"$y=-3$")
        ax2.spines['right'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax2.spines["bottom"].set_visible(False)
        ax2.yaxis.set_ticks_position('left')
        ax2.get_xaxis().set_visible(False)    

    
        ax3.plot(y,[vrednost2(0,i,nabojcki,konst,plaketke,a) for i in y],label=r"$x=0$",transform=rot+base)
        ax3.plot(y,[vrednost2(1,i,nabojcki,konst,plaketke,a) for i in y],label=r"$x=1$",transform=rot+base)
        ax3.plot(y,[vrednost2(2,i,nabojcki,konst,plaketke,a) for i in y],label=r"$x=2$",transform=rot+base)

        ax3.spines['right'].set_visible(False)
        ax3.spines['left'].set_visible(False)
        ax3.spines["top"].set_visible(False)
        ax3.get_yaxis().set_visible(False)
        ax3.legend()

        handles,labels = ax2.get_legend_handles_labels() 
        labax.legend(handles,labels,loc="center")
        labax.spines['right'].set_visible(False)
        labax.spines['left'].set_visible(False)
        labax.spines["top"].set_visible(False)
        labax.spines["bottom"].set_visible(False)
        labax.get_xaxis().set_visible(False)    
        labax.get_yaxis().set_visible(False) 

        plt.colorbar(cs,cax=cax)
        #cb.ax.locator_params(nbins=5)
    
        
    ani = animation.FuncAnimation(fig,animiraj,range(40),interval=300)   
    #plt.show()
    ani.save("rotiranje2.mp4")  
    print("evo")


if 0:
    #test plaket
    plakete = plaketeFromSVG("CrkaF.svg")  #krog jih ma 32 miki pa 128
    plakete = shiftajPlakete(plakete,-2,0)
    for plak in plakete:
        plak.napetost = 1
    plakete2 = shiftajPlakete(plaketeFromSVG("CrkaM.svg"),2,0)
    for plak in plakete2:
        plak.napetost = -1
    
    plaketke = plakete+plakete2
    
    for plak in plaketke:
        if plak.napetost>0:
            barva = "m"
        else:
            barva = "c"
        for i in range(1):
            plt.plot([plak.levi[0]+i*a,plak.desni[0]+i*a],[plak.levi[1],plak.desni[1]],barva)


if 0:
    #animacija
    n=10
    N= 0
    a= 5
    L= 3
     
    fig, ax = plt.subplots(figsize=(20,5))
    
    x = np.linspace(-20,20,200)
    y = np.linspace(-5,5,200)
    X,Y = np.meshgrid(x,y)
    XX, YY = np.meshgrid(np.linspace(-20,20,40),np.linspace(-5,5,20))
    plaketke = narediPlakete(n,N,a,L)
    kot = 0
    cax =make_axes_locatable(ax).append_axes('right', '2%',"1%")
    
    Zji = []
    polje = []
    plakete=[]
    for i in range(21):
        print(i)
        plakete.append(plaketke)
        nabojcki = najdiGostotePotencial2(plaketke,a)
        Z = [[vrednost2(x,y,nabojcki,plaketke,a) for x in np.linspace(-20,20,200)] for y in np.linspace(-5,5,200)]
        Zji.append(Z)        
        polja = np.array([[vrednost2(xx,yy,nabojcki,plaketke,a,True) for xx in np.linspace(-20,20,40)] for yy in np.linspace(-5,5,20)])
        polje.append(polja)
        plaketke = rotirajPlakete(plaketke,pi/10)
    mini = np.min(Zji)
    maxi = np.max(Zji)
        
    def animiraj(t):
        global plaketke
        global kot
        ax.clear()
        print(t)
        

        cs = ax.contourf(X,Y,Zji[t],levels=np.linspace(mini,maxi,50),cmap="hot")
        for plak in plakete[t]:
            for i in range(-5,6):
                ax.plot([plak.levi[0]+i*a,plak.desni[0]+i*a],[plak.levi[1],plak.desni[1]],"g")
        ax.quiver(XX,YY,polje[t][:,:,0],polje[t][:,:,1])
        ax.set_xlim(-20,20)
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")        
        plt.suptitle(r"$n=10, a=5, L=3, \varphi={}°$".format(round(t*18,2)))
        cax.cla()
        fig.colorbar(cs,cax=cax)
        
    ani = animation.FuncAnimation(fig,animiraj,range(21),interval=100)   
    #plt.show()
    ani.save("rotiranje.mp4")  
    print("evo")
    
    
    
if 0:
    #U(y)
    a=5
    L=5
    n=5
    ipsiloni = np.linspace(0,20,300)
    referenca = -ipsiloni+np.ones(300)
    Nji = [i for i in range(1,10)]
    for i in range(9):
        print(i)
        plaketke=narediPlakete(5,Nji[i],a,L)
        naboji=najdiGostotePotencial(plaketke)
        nabojcki = naboji[:-1]
        konst=naboji[-1]
        vrednosti = np.array([vrednost(pi,i,nabojcki,konst,plaketke) for i in ipsiloni])
        plt.plot(ipsiloni,vrednosti,label=r"$N={}$".format(Nji[i]))

    plt.plot(ipsiloni,-ipsiloni + np.ones(300), "--",color="k")
    plt.legend(loc="best")
    print("kaj")
    plt.xlabel(r"$y$")
    plt.ylabel(r"$U$")
    plt.title(r"$U(\pi,y)$ Navadna Greenova $a=L=n=5$")
    plt.savefig("samogor3.pdf")

if 0:
    #N za 0.1% napako v od y
    a=5
    L=5
    n=5
    ipsiloni = [5+i for i in range(21)]
    odgovor = []
    for y in ipsiloni:
        print(y)
        ref = -y+1
        for N in range(1,100):
            plaketke=narediPlakete(5,N,a,L)
            naboji=najdiGostotePotencial(plaketke)
            nabojcki = naboji[:-1]
            konst=naboji[-1]
            trenutn = vrednost(pi,y,nabojcki,konst,plaketke)
            rel = np.abs((ref-trenutn)/ref)
            if rel <= 0.001:
                odgovor.append(N)
                break
            if N==99:
                print("wut")
    plt.plot(ipsiloni,odgovor,":")
    print("kaj")
    plt.xlabel(r"$y$")
    plt.ylabel(r"$N$")
    plt.title(r"Potreben $N$ za rel. napako $0.001$ v odv. od $y$, $x=\pi,a=L=n=5$")
    plt.savefig("samogor4.pdf")
    
if 0:
    #U(n) per. Green
    a = 5
    L=3
    N=1
    ref = -1.3150724453148268
    nji = [i for i in range(1,51)]
    rez = []
    for n in nji:
        print(n)
        l = L/n
        levi = np.array([np.array([-k1*a-0.5*L+ k2*l-0.5*a,0]) for k1 in range(N) for k2 in range(n)]) 
        levi2 = np.array([np.array([k1*a-0.5*L+ k2*l+0.5*a,0]) for k1 in range(N) for k2 in range(n)]) 
        levi = np.concatenate((levi[::-1],levi2))
        desni = np.array([np.array([i[0]+l,0]) for i in levi])
        plaketke  = [Plaketa(levi[i],desni[i]) for i in range(2*n*N)]      
        U=[]
        napetost = 1
        for i in range(2*N):
            for j in range(n):
                U.append(napetost)
                plaketke[i+j].napetost = napetost
            napetost *= (-1)
        U.append(0)        
        naboji = najdiGostotePotencial2(plaketke,U,2*a)
        nabojcki = naboji[:-1]
        konst = naboji[-1]
        rezu = vrednost2(pi,np.e,nabojcki,konst,plaketke,2*a)+vrednost2(np.sqrt(2),-pi,nabojcki,konst,plaketke,2*a)+vrednost2(np.sqrt(3),0,nabojcki,konst,plaketke,2*a)
        rez.append(rezu)
    plt.plot(nji,rez,ls=":",marker=".")
    ax = plt.gca()
    ax2 = ax.twinx()
    ax2.plot(nji,np.abs(np.array(rez)-ref),ls=":",marker=".",color="r")
    ax2.set_yscale("log")
    ax2.tick_params(axis="y",colors="r")
    ref = rez
    ax.set_xlabel(r"$n$")
    ax.set_ylabel(r"$U$",color="b")
    ax.tick_params(axis="y",colors="b")
    ax2.set_ylabel(r"$\Delta U$",color="r")
    plt.title(r"Potencial v odv. od $n$ - Period. Greenova $a=5,L=3$")
    plt.savefig("Napake3.pdf")
    
if 0:
    #hitrost od n
    a=5
    L=3
    casiPer = []
    casiGre = []
    nji = [5*i for i in range(1,41)]
    for n in nji:
        print(n)
        nav = True
        plaketke = narediPlakete(n,0,a,L)
        U = np.append(np.ones(n),0)
        start = time.time()
        naboji = najdiGostotePotencial(plaketke,U)
        nabojcki = naboji[:-1]
        konst = naboji[-1]
        vrednost(2,2,nabojcki,konst,plaketke)
        casiGre.append(time.time()-start)
        nav = False
        plaketke = narediPlakete(n,0,a,L)
        U = np.append(np.ones(n),0)
        start = time.time()
        naboji = najdiGostotePotencial2(plaketke,U,a)
        nabojcki = naboji[:-1]
        konst = naboji[-1]
        vrednost2(2,2,nabojcki,konst,plaketke,a)
        casiPer.append(time.time()-start)       
    plt.plot(nji,casiPer,ls=":",marker=".",label="Periodicna")
    plt.plot(nji,casiGre,ls=":",marker=".",label="Navadna")
    plt.xlabel(r"$n$")
    plt.ylabel(r"$t[s]$")
    plt.title(r"Potreben čas za pridobitev nabojev in izračun vrednosti v neki točki za obe metodi.")
    plt.legend()
    plt.savefig("Casi.pdf")

if 0:
    #contour napake od n,N
    a=5
    L=3
    ref = 1.3138817328741375
    #Nji = [5,10,15,20,25,30,35,40,45,50]
    #nji = [2,3,4,5,6,7,8,9,10]
    Nji = [2*i for i in range(1,21)]
    nji = [i for i in range(1,7)]
    rez = []
    for N in Nji:
        print(N)
        vred = []
        for n in nji:
            l = L/n
            levi = np.array([np.array([-k1*a-0.5*L+ k2*l-0.5*a,0]) for k1 in range(N) for k2 in range(n)]) 
            levi2 = np.array([np.array([k1*a-0.5*L+ k2*l+0.5*a,0]) for k1 in range(N) for k2 in range(n)]) 
            levi = np.concatenate((levi[::-1],levi2))
            desni = np.array([np.array([i[0]+l,0]) for i in levi])
            plaketke  = [Plaketa(levi[i],desni[i]) for i in range(2*n*N)]      
            U=[]
            napetost = 1
            for i in range(2*N):
                for j in range(n):
                    U.append(napetost)
                    plaketke[i+j].napetost = napetost
                napetost *= (-1)
            U.append(0)
            naboji = najdiGostotePotencial(plaketke,U)
            nabojcki = naboji[:-1]
            konst = naboji[-1]
            rezu = vrednost(pi,np.e,nabojcki,konst,plaketke)+vrednost(np.sqrt(2),-pi,nabojcki,konst,plaketke)+vrednost(np.sqrt(3),0,nabojcki,konst,plaketke)
            vred.append(np.abs(rezu-ref))
        rez.append(vred)
    X,Y = np.meshgrid(np.array(nji),np.array(Nji))
    cs = plt.contourf(X,Y,rez,levels=np.linspace(np.min(rez),np.max(rez),50),cmap="rainbow")
    for c in cs.collections:
        c.set_edgecolor("face")
    plt.xlabel(r"$n$")
    plt.ylabel(r"$N$")
    plt.title(r"Napaka potenciala glede na referenco pri $N=100,n=20$. $a=5,L=3$")
    plt.colorbar(cs)
    plt.savefig("Napake2.pdf", dpi=300)
    
if 0:
    #plot - contour in quiver
    n = 50
    N = 1
    a = 5
    L = 3
    if 0: #green za alt. s sredinskim
        l = L/n
        levi = np.array([np.array([k1*a-0.5*L+ k2*l-0.5*a,0]) for k1 in range(2) for k2 in range(50)])
        desni = np.array([np.array([i[0]+l,0]) for i in levi])
        plaketke = [Plaketa(levi[i],desni[i]) for i in range(100)]
    if 1: #green za alt brez sredinskega
        l = L/n
        levi = np.array([np.array([-k1*a-0.5*L+ k2*l-0.5*a,0]) for k1 in range(N) for k2 in range(n)]) 
        levi2 = np.array([np.array([k1*a-0.5*L+ k2*l+0.5*a,0]) for k1 in range(N) for k2 in range(n)]) 
        levi = np.concatenate((levi[::-1],levi2))
        desni = np.array([np.array([i[0]+l,0]) for i in levi])
        plaketke  = [Plaketa(levi[i],desni[i]) for i in range(2*n*N)]
    #return np.array([-1/2 + i,0])
    #plaketke=rotirajPakete(plaketke,1)
    #plaketke = narediPlakete(n,N,a,L)
    naboji = najdiGostotePotencial2(plaketke,2*a)
    nabojcki= naboji[:-1]
    konst = naboji[-1]
    print(vrednost2(pi,np.e,nabojcki,konst,plaketke,2*a)+vrednost2(np.sqrt(2),-pi,nabojcki,konst,plaketke,2*a)+vrednost2(np.sqrt(3),0,nabojcki,konst,plaketke,2*a))
    print(1/0)
    x = np.linspace(-20,20,200)
    y = np.linspace(-5,5,200)
    X,Y = np.meshgrid(x,y)
    XX,YY = np.meshgrid(np.linspace(-20,20,40),np.linspace(-5,5,20))
    Z = [[vrednost2(x,y,nabojcki,konst,plaketke,2*a) for x in np.linspace(-20,20,200)] for y in np.linspace(-5,5,200)]
    polja = np.array([[vrednost2(xx,yy,nabojcki,konst,plaketke,2*a,True) for xx in np.linspace(-20,20,40)] for yy in np.linspace(-5,5,20)])
    #print(polja)
    cs = plt.contourf(X,Y,Z,levels=np.linspace(np.min(Z),np.max(Z),50),cmap="hot")

# This is the fix for the white lines between contour levels
    for c in cs.collections:
        c.set_edgecolor("face")
    for plak in plaketke:
        if plak.napetost==1:
            barva = "m"
        else:
            barva = "c"
        for i in range(-3,3):
            plt.plot([plak.levi[0]+i*2*a,plak.desni[0]+i*2*a],[plak.levi[1],plak.desni[1]],barva)
    plt.quiver(XX,YY,polja[:,:,0],polja[:,:,1])
    plt.colorbar(cs)
    plt.xlim(-5,5)
    plt.title("$n=50, a=5, L=3 \ U(x,y)$")
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    plt.savefig("ContourGreen3.pdf",dpi=300)
    plt.show()




if 0: #potencial od N za vec x
    L=3
    a = 5
    n=4
    Y=2.3
    rez = [[],[],[],[],[]]
    konste = []
    Nji = [2*i for i in range(1,26)]
    for N in Nji:
        print(N)
        if 1: #green za alt brez sredinskega
            l = L/n
            levi = np.array([np.array([-k1*a-0.5*L+ k2*l-0.5*a,0]) for k1 in range(N) for k2 in range(n)]) 
            levi2 = np.array([np.array([k1*a-0.5*L+ k2*l+0.5*a,0]) for k1 in range(N) for k2 in range(n)]) 
            levi = np.concatenate((levi[::-1],levi2))
            desni = np.array([np.array([i[0]+l,0]) for i in levi])
            plaketke  = [Plaketa(levi[i],desni[i]) for i in range(2*n*N)]
        #plaketke = narediPlakete(n,N,a,L)
        naboji =  najdiGostotePotencial(plaketke)
        nabojcki=naboji[:-1]
        konst = naboji[-1]
        konste.append(konst)
        rez[0].append(vrednost(0.4-2*a,Y,nabojcki,konst,plaketke))
        rez[1].append(vrednost(0.4-a,Y,nabojcki,konst,plaketke))
        rez[2].append(vrednost(0.4,Y,nabojcki,konst,plaketke))
        rez[3].append(vrednost(0.4+a,Y,nabojcki,konst,plaketke))
        rez[4].append(vrednost(0.4+2*a,Y,nabojcki,konst,plaketke))

    plt.plot(Nji,rez[0],marker=".",ls=":",label=r"$x=0.4-2a$")
    plt.plot(Nji,rez[1],marker=".",ls=":",label=r"$x=0.4-a$")
    plt.plot(Nji,rez[2],marker=".",ls=":",label=r"$x=0.4$")
    plt.plot(Nji,rez[3],marker=".",ls=":",label=r"$x=0.4+a$")
    plt.plot(Nji,rez[4],marker=".",ls=":",label=r"$x=0.4+2a$")
    plt.legend(loc="right")
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    ax2.plot(Nji,konste,"--")
    ax2.plot(Nji,np.array(rez[0])-np.array(konste),"--")
    plt.xlabel(r"$N$")
    plt.ylabel(r"$U$")
    plt.title(r"$L=3,a=5,n=4,U_i=\pm 1,  U(x,2.3)$")
    plt.savefig("PeriodicnostVsotaAlt.pdf")
    

if 0: #profil na sredini za vec N
    L=3
    a = 5
    n=4
    x = [-40+i for i in range(81) if i!= 0]
    Nji = [10*i for i in range(1,10)]
    for N in Nji:
        print(N)
        l = L/n
        levi = np.array([np.array([-k1*a-0.5*L+ k2*l-0.5*a,0]) for k1 in range(N) for k2 in range(n)]) 
        levi2 = np.array([np.array([k1*a-0.5*L+ k2*l+0.5*a,0]) for k1 in range(N) for k2 in range(n)]) 
        levi = np.concatenate((levi[::-1],levi2))
        desni = np.array([np.array([i[0]+l,0]) for i in levi])
        plaketke  = [Plaketa(levi[i],desni[i]) for i in range(2*n*N)]
        
        nabojcki =  najdiGostotePotencial(plaketke)
        polovica = int(n*(2*N)/2)
        plt.plot(x,nabojcki[:-1][polovica-40:polovica+40],label=r"$N={}$".format(N))
    plt.legend(loc="best",ncol=4,fontsize="small",fancybox=True)
    plt.grid(True)
    #plt.yscale("log")
    plt.title(r"$L=3,a=5,n=4$ Naboji sredinskih 80 plaket za $U=\pm1$")        
    plt.savefig("NabojiVsotaAlt.pdf")
    
if 0: #celoten profil za vec N
    L=3
    a = 5
    n=4
    Nji = [10*i for i in range(1,5)]
    for N in Nji:
        print(N)
        plaketke = narediPlakete(n,N,a,L)
        nabojcki =  najdiGostotePotencial(plaketke)
        x = np.linspace(-1,1,len(nabojcki[:-1]))
        plt.plot(x,nabojcki[:-1],label=r"$N={}$".format(N))
    plt.legend(loc="best",fancybox=True)
    #plt.yscale("log")
    plt.title(r"$L=3,a=5,n=4$ Profil gostote naboja za več $N$ za $U=\pm 1$")        
    plt.savefig("NabojiVsota2Alt2.pdf")
    


