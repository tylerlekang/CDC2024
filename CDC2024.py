import numpy as np
import numpy.random as rnd
from numpy.linalg import norm
import matplotlib as mpl
import matplotlib.pyplot as plt
from time import perf_counter


# trick to make PDF outputs of figs acceptable for IEEE submission
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42


## function plotters

def plotFunc(func,xmin=-1,xmax=1,xN=1001,fsize=(12,3),pltTitle=None,*args,**kwargs):
    x = np.linspace(xmin,xmax,xN)
    # print(x[:10],x.shape) # debug
    fx = func(x,*args,**kwargs)
    fig = plt.figure(figsize=fsize)
    plt.plot(x,fx)
    if pltTitle is not None: plt.title(pltTitle)
    plt.xlabel('x')
    plt.tight_layout()
    plt.show()

def pltFunc3d(func3d,xmin=-1,xmax=1,xN=1001,fsize=(8,6),pltTitle=None):
    x = np.linspace(xmin,xmax,xN)
    y = np.linspace(xmin,xmax,xN)
    X,Y = np.meshgrid(x,y)
    xy = np.column_stack([X.ravel(),Y.ravel()]) # assuming func3d accepts sinlge (N,2) input
    
    fXY = func3d(xy)
    Z = fXY.reshape(X.shape)
    
    fig = plt.figure(figsize=fsize)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X,Y,Z)
    ax.view_init(elev=10,azim=45)
    ax.dist=25
    if pltTitle is not None: plt.title(pltTitle)
    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')
    plt.tight_layout()
    plt.show()


## non-analytic smooth flat bump function
## see https://en.wikipedia.org/wiki/Non-analytic_smooth_function#Smooth_transition_functions

def f_bump(x, tol=1e-6):
    ret = np.zeros_like(x)
    posIdx = np.where(x > 0+tol)
    ret[posIdx] = np.exp(-1/x[posIdx])
    return ret

def g_bump(x):
    fx = f_bump(x)
    return fx / (fx + f_bump(1 - x))

# scalar bump func that transitions 0 to 1 over [a,b], =1 over [b,c], transitions 1 to 0 over [c,d]
def bumpScalar(x,a=0,b=1,c=2,d=3):
    arg1 = (x-a)/(b-a)
    arg2 = (d-x)/(d-c)
    return g_bump(arg1)*g_bump(arg2)

# this returns the scalar bump function from given parameters a,b,c,d (see above) 
def set_bumpScalar_func(a,b,c,d):
    def bumpScalar(x):
        arg1 = (x-a)/(b-a)
        arg2 = (d-x)/(d-c)
        return g_bump(arg1)*g_bump(arg2)
    return bumpScalar

# this returns the B(x) function for x=[x_1,...,x_n] vector state, from given parameters a,b,c,d (see above)  
def set_bumpX_func(bumpScalars):
    # I am hardcoding this to n=2 for this work, it could be made more general
    bump1 = bumpScalars[0]
    bump2 = bumpScalars[1]
    def bumpX(X):
        return bump1(X[:,0])*bump2(X[:,1])
    return bumpX


## platform true dynamics
def setPlatformTrueLinearDyn(a1, a2, beta, prts=True):
    A = np.array([[0,1],
                  [a1,a2]])
    A_eig = np.sort(np.linalg.eig(A)[0])
    B = np.array([[0.],
                  [1.]])
    BL = B*beta

    if prts:
        print('A:\n',A,A.shape)
        print('eig:',A_eig)
        # print('B:\n',B,B.shape)
        print('B Lambda:\n',BL,BL.shape)

    return A,BL


## platform matched nonlinearity
def setPlatformNonlinearity(theta1, theta2, theta3, bumpFunc, prts=True):
    Theta = np.array([[theta1,theta2,theta3]]) # shape (1,3)

    def PsiFunc(state): # state assumed to be numpy shape (2,)
        Bx = bumpFunc(state[np.newaxis,:]) # needs (1,2) shaped input, output shape (1,)
        psi1 = Bx * state[0]**2 * state[1] # shape (1,)
        psi2 = Bx * abs(state[0]) * state[1] # shape (1,)
        psi3 = Bx * state[0]**3 # shape (1,)
        return np.array([psi1,
                         psi2,
                         psi3]) # shape (3,1)

    def ThetaPsiFunc(state): # state assumed to be numpy shape (2,)
        Bx = bumpFunc(state[np.newaxis,:]) # needs (1,2) shaped input, output shape (1,)
        psi1 = Bx * state[0]**2 * state[1] # shape (1,)
        psi2 = Bx * abs(state[0]) * state[1] # shape (1,)
        psi3 = Bx * state[0]**3 # shape (1,)
        Psi = np.array([psi1,
                        psi2,
                        psi3]) # shape (3,1)
        return Theta @ Psi # shape (1,1)

    if prts:
        print('Theta:\n',Theta,Theta.shape)
        print('Psi shape:',PsiFunc(np.array([1.,1.])).shape)
        print('ThetaPsi shape:',ThetaPsiFunc(np.array([1.,1.])).shape)
    
    return Theta,PsiFunc,ThetaPsiFunc

# same overall nonlinearity but without the bump functions
def setPlatformNLnoBump(theta1, theta2, theta3, prts=True):
    Theta = np.array([[theta1,theta2,theta3]]) # shape (1,3)
    
    def PsiNBFunc(state):
        psi1 = np.array([state[0]**2 * state[1]]) # shape (1,)
        psi2 = np.array([abs(state[0]) * state[1]]) # shape (1,)
        psi3 = np.array([state[0]**3]) # shape (1,)
        Psi = np.array([psi1,
                        psi2,
                        psi3]) # shape (3,1)
        return Psi # shape (3,1)
    
    def ThetaPsiNBFunc(state):
        psi1 = np.array([state[0]**2 * state[1]]) # shape (1,)
        psi2 = np.array([abs(state[0]) * state[1]]) # shape (1,)
        psi3 = np.array([state[0]**3]) # shape (1,)
        Psi = np.array([psi1,
                        psi2,
                        psi3]) # shape (3,1)
        return Theta @ Psi # shape (1,1)
    
    if prts:
        print('ThetaPsiNB shape:',ThetaPsiNBFunc(np.array([1.,1.])).shape)
    
    return PsiNBFunc,ThetaPsiNBFunc


## 2nd order linear damped oscillator
def set2ndLinDampedOsc(w0, xi, prts=True):
    Ar = np.array([[0,1],
                [-w0**2,-2*xi*w0]])
    Ar_eig = np.linalg.eig(Ar)[0]
    Br = np.array([[0.],
                [w0**2]])

    if prts:
        print('Ar:\n',Ar,Ar.shape)
        print('eig:',Ar_eig)
        print('Br:\n',Br,Br.shape)

    return Ar,Br


## Euler simulation of uncontrolled dynamics
def plotStateSim(ts, x, PsiVals = None, ThetaPsiVals = None, Ref=False):
    n = x.shape[0]
    tN = ts.size
    
    if PsiVals is None and ThetaPsiVals is None:
        plt.figure(figsize=(12,3))
        plt.axhline(0,c='k',lw=0.5) # axes lines
        plt.axvline(0,c='k',lw=0.5)
        for i in range(n):
            if Ref: labi = r'$x^r_{t,%d}$'%(i+1)
            else: labi = r'$x_{t,%d}$'%(i+1)
            plt.plot(ts,x[i],label=labi)
        leg = plt.legend(bbox_to_anchor=(0.995,1),loc='upper left',edgecolor='w')
        for line in leg.legendHandles: line.set_linewidth(2)
        plt.xlabel(r'time $t$')
        if Ref: plt.ylabel(r'ref states $x^r_t$')
        else: plt.ylabel(r'states $x_t$')
        # leg = plt.legend(bbox_to_anchor=(0.99,1),loc='upper left',fontsize=15,edgecolor='w')
        # for line in leg.legendHandles: line.set_linewidth(2)
        # plt.xlabel(r'time $t$',fontsize=15)
        # plt.ylabel(r'states $x_t$',fontsize=15)
        # plt.xticks(fontsize=13)
        # plt.yticks(fontsize=13)

    else: # PsiVals is not None OR ThetaPsiVals is not None
        fig,ax = plt.subplots(3,1,sharex=True,figsize=(12,8))
        
        # plot state values
        plt.sca(ax[0])
        plt.axhline(0,c='k',lw=0.5) # axes lines
        plt.axvline(0,c='k',lw=0.5)
        for i in range(n): plt.plot(ts,x[i],label=r'$x_{t,%d}$'%(i+1))
        leg = plt.legend(bbox_to_anchor=(0.995,1),loc='upper left',edgecolor='w')
        for line in leg.legendHandles: line.set_linewidth(2)
        plt.ylabel(r'states $x_t$')
        # leg = plt.legend(bbox_to_anchor=(0.99,1),loc='upper left',fontsize=15,edgecolor='w')
        # for line in leg.legendHandles: line.set_linewidth(2)
        # plt.xlabel(r'time $t$',fontsize=15)
        # plt.ylabel(r'states $x_t$',fontsize=15)
        # plt.xticks(fontsize=13)
        # plt.yticks(fontsize=13)
        
        # plot Psi functions values
        plt.sca(ax[1])
        plt.axhline(0,c='k',lw=0.5) # axes lines
        plt.axvline(0,c='k',lw=0.5)
        for i in range(3): plt.plot(ts,PsiVals[i],color='C%d'%(n+i),label=r'$\Psi_{%d}(x_t)$'%(i+1))
        leg = plt.legend(bbox_to_anchor=(0.995,1),loc='upper left',edgecolor='w')
        for line in leg.legendHandles: line.set_linewidth(2)
        plt.ylabel(r'Psi funcs')
        # plt.yticks(fontsize=13)
        
        # plot overall ThetaPsi function values
        plt.sca(ax[2])
        plt.axhline(0,c='k',lw=0.5) # axes lines
        plt.axvline(0,c='k',lw=0.5)
        plt.plot(ts,ThetaPsiVals.reshape((tN,)),color='C%d'%(n+3+i),label=r'$\Theta\Psi(x_t)$')
        leg = plt.legend(bbox_to_anchor=(0.995,1),loc='upper left',edgecolor='w')
        for line in leg.legendHandles: line.set_linewidth(2)
        # plt.yticks(fontsize=13)
        plt.ylabel(r'Nonlinearity')
        plt.xlabel(r'time $t$')
    
    plt.tight_layout()
    plt.show()
    

def simLinearUncontrolled(A, t0, tf, dt, x0=[1.,1.], Ref=False):
    ts = np.arange(t0,tf+dt,dt)
    print('time steps:',ts,ts.shape)
    
    n = A.shape[0]
    
    x = np.zeros((n,ts.size))
    x[:,0] = x0

    tr0 = perf_counter()
    for ti,t in enumerate(ts[:-1]):
        xDot = A@x[:,ti]
        x[:,ti+1] = x[:,ti] + xDot*dt

    print('Euler integration simulation time: %.1f sec\n'%(perf_counter()-tr0))
    
    plotStateSim(ts,x, Ref=Ref)


def simFullUncontrolled(A,BL,beta,PsiFunc,ThetaPsiFunc,t0,tf,dt,x0=[1.,1.]):
    ts = np.arange(t0,tf+dt,dt)
    print('time steps:',ts,ts.shape)
    
    n = A.shape[0]
    
    x = np.zeros((n,ts.size))
    x[:,0] = x0
    
    PsiVals = np.zeros((3,ts.size))
    ThetaPsiVals = np.zeros((1,ts.size))

    tr0 = perf_counter()
    for ti,t in enumerate(ts[:-1]):
        PsiVals[:,ti] = PsiFunc(x[:,ti]).reshape((3,))
        ThetaPsiVals[:,ti] = ThetaPsiFunc(x[:,ti])
        
        xDot = A@x[:,ti] + (BL*beta*ThetaPsiVals[:,ti]).reshape((n,)) # shape (n,)
        x[:,ti+1] = x[:,ti] + xDot*dt # shape (n,)
    
    # final vals
    PsiVals[:,-1] = PsiFunc(x[:,-1]).reshape((3,))
    ThetaPsiVals[:,-1] = ThetaPsiFunc(x[:,-1])

    print('Euler integration simulation time: %.1f sec\n'%(perf_counter()-tr0))
    
    plotStateSim(ts,x, PsiVals, ThetaPsiVals)


## random Psi ReLU initializing functions

def ReLU(u):
    return u * (u > 0)

def init_Psi(m, seedN=None, prts=True):
    rnd.seed(seedN)
    Alphas = rnd.uniform(-1,1,(2,m))
    Alphas /= norm(Alphas,axis=0)
    ts = rnd.uniform(-1,1,(m,1))
    
    # def PsiReLU(x):
    #     '''
    #     required shapes:
    #     x: (2,T)
    #     Alphas: (2,m) --> Alphas.T: (m,2)
    #     ts: (m,1)
    #     output shape: (m,T)
    #     '''
    #     return ReLU(Alphas.T @ x - ts)
    
    def Psi(x):
        '''
        required shapes:
        1: (1,T)
        x: (2,T)
        PsiReLU: (m,T)
        output shape: (m+2+1,T)
        '''
        Psi1s = np.ones((1,x.shape[1]))
        PsiReLU = ReLU(Alphas.T @ x - ts)
        
        return np.vstack((Psi1s,x,PsiReLU))
    
    if prts:
        xtest = np.array([[1.],[1.]]) # shape (2,1)
        print('Psi shape:',Psi(xtest).shape)
    
    return Psi

def getThetaApprox(fxFunc,psiRandFunc,xmin,xmax,xN, prtMaxErr=True):
    x = np.linspace(xmin,xmax,xN)
    y = np.linspace(xmin,xmax,xN)
    X,Y = np.meshgrid(x,y)
    xy = np.column_stack([X.ravel(),Y.ravel()]) # assuming funcs accept single (N,2) input
    
    fx = np.zeros((xy.shape[0],1))
    for i,xyi in enumerate(xy):
        # print(xyi,xyi.shape) ###############
        fx[i] = fxFunc(xyi)
        
    psiRand = psiRandFunc(xy.T)
    
    A = psiRand @ psiRand.T  # shape (m+3,4)
    b = psiRand @ fx  # shape (m+3,1)
    ThetaApprox = np.linalg.lstsq(A,b, rcond=None)[0].T # solves Theta.T = (psiRand @ psiRand.T)^-1 (psiRand@fx)
    
    # now apply those lstsq ThetaApprox to psiRand to get the approx func values on the grid
    ThetaPsiApprox = ThetaApprox @ psiRand

    approx_errors_overGrid = fx - ThetaPsiApprox.T # need .T to match shape!
    maxError = max(approx_errors_overGrid)
    
    if prtMaxErr:
        print('Maximum func approx error |f(x)-Theta@Psi(x)| over specified x grid: %.2f'%(maxError))

    # debug prints
    # print(fx.shape,psiRand.shape,ThetaApprox.shape)
    # print(ThetaApprox)
    # print(ThetaPsiApprox.shape)
    # print(approx_errors_overGrid.shape)

    return ThetaApprox,approx_errors_overGrid,maxError