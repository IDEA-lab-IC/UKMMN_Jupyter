from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output
from matplotlib import colors
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

def setup_topology(nelx, nely, rmin, Emin, Emax, nu, penal, volfrac):
    # element stiffness (8x8) -- keep your k/KE definition
    k = np.array([1/2-nu/6, 1/8+nu/8, -1/4-nu/12, -1/8+3*nu/8,
                  -1/4+nu/12, -1/8-nu/8, nu/6, 1/8-3*nu/8])
    KE = 1.0/(1-nu**2)*np.array([
        [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
        [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
        [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
        [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
        [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
        [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
        [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
        [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]]
    ])

    ndof = 2*(nelx+1)*(nely+1)

    edofMat = np.zeros((nelx*nely, 8), dtype=int)
    for elx in range(nelx):
        for ely in range(nely):
            el = ely + elx*nely
            n1 = (nely+1)*elx + ely
            n2 = (nely+1)*(elx+1) + ely
            edofMat[el, :] = np.array([2*n1+2, 2*n1+3, 2*n2+2, 2*n2+3,
                                       2*n2, 2*n2+1, 2*n1, 2*n1+1])

    # build iK, jK pattern for sparse assembly (coo)
    iK = np.kron(edofMat, np.ones((8,1))).flatten().astype(int)
    jK = np.kron(edofMat, np.ones((1,8))).flatten().astype(int)

    # BCs: fixed DOFs and free DOFs
    dofs = np.arange(ndof)
    fixed = np.union1d(dofs[0:2*(nely+1):2], np.array([2*(nelx+1)*(nely+1)-1]))
    free = np.setdiff1d(dofs, fixed)

    # load and solution vectors
    f = np.zeros((ndof, 1))
    u = np.zeros((ndof, 1))
    # default load same as your function (adjust as needed)
    f[1, 0] = -1.0

    # Filter assembly (H, Hs)
    nfilter = int(nelx*nely*((2*(np.ceil(rmin)-1)+1)**2))
    iH = np.zeros(nfilter, dtype=int)
    jH = np.zeros(nfilter, dtype=int)
    sH = np.zeros(nfilter, dtype=float)
    cc = 0
    for i in range(nelx):
        for j in range(nely):
            row = i*nely + j
            kk1 = int(max(i - (np.ceil(rmin)-1), 0))
            kk2 = int(min(i + np.ceil(rmin), nelx))
            ll1 = int(max(j - (np.ceil(rmin)-1), 0))
            ll2 = int(min(j + np.ceil(rmin), nely))
            for k in range(kk1, kk2):
                for l in range(ll1, ll2):
                    col = k*nely + l
                    fac = rmin - np.sqrt((i-k)**2 + (j-l)**2)
                    weight = max(0.0, fac)
                    iH[cc] = row
                    jH[cc] = col
                    sH[cc] = weight
                    cc += 1
    H = coo_matrix((sH, (iH, jH)), shape=(nelx*nely, nelx*nely)).tocsc()
    Hs = H.sum(1)

    # initial design variables (flat)
    x = volfrac * np.ones(nelx*nely, dtype=float)
    xold = x.copy()
    xPhys = x.copy()

    # bookkeeping arrays
    ce = np.zeros(nelx*nely, dtype=float)
    dc = np.zeros(nelx*nely, dtype=float)
    dv = np.ones(nelx*nely, dtype=float)

    # pack everything into state dict
    state = {
        'nelx': nelx, 'nely': nely, 'rmin': rmin,
        'Emin': Emin, 'Emax': Emax, 'nu': nu, 'penal': penal,
        'volfrac': volfrac,
        'ndof': ndof, 'KE': KE, 'edofMat': edofMat,
        'iK': iK, 'jK': jK,
        'fixed': fixed, 'free': free,
        'f': f, 'u': u,
        'H': H, 'Hs': Hs,
        'x': x, 'xold': xold, 'xPhys': xPhys,
        'ce': ce, 'dc': dc, 'dv': dv
    }
    return state
