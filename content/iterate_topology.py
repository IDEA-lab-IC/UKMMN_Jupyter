from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output
from matplotlib import colors
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

def iterate_topology(state, params):
    nelx = state['nelx']; nely = state['nely']
    KE = state['KE']; edofMat = state['edofMat']
    iK = state['iK']; jK = state['jK']
    free = state['free']; f = state['f']; u = state['u']
    H = state['H']; Hs = state['Hs']
    x = state['x']; xPhys = state['xPhys']
    Emin = params.get('Emin', state['Emin'])
    Emax = params.get('Emax', state['Emax'])
    penal = params.get('penal', state['penal'])
    move = params.get('move', 0.2)

    # 1) Assemble global stiffness K (sK vector) and solve
    # sK: element stiffness entries repeated by element
    sK = (KE.flatten(order='F')[:, np.newaxis] * (Emin + xPhys**penal * (Emax - Emin))).flatten(order='F')
    # build sparse K and solve (use same approach as your code)
    K = coo_matrix((sK, (iK, jK)), shape=(state['ndof'], state['ndof'])).tocsc()
    K_reduced = K[state['free'], :][:, state['free']]
    u[state['free'], 0] = spsolve(K_reduced, f[state['free'], 0])

    # 2) Element strain energy and objective
    # u[edofMat] gives (nelx*nely,8,1) style array via fancy indexing; replicate your approach:
    ue = u[edofMat].reshape(nelx*nely, 8)
    ce = (np.dot(ue, KE) * ue).sum(axis=1)   # scalar per element
    obj = ((Emin + xPhys**penal * (Emax - Emin)) * ce).sum()

    # 3) Sensitivities
    dc = (-penal * xPhys**(penal-1) * (Emax - Emin)) * ce
    dv = np.ones_like(dc)

    # 4) Filter sensitivities
    dc = np.asarray(H * (dc[np.newaxis].T / Hs))[:, 0]
    dv = np.asarray(H * (dv[np.newaxis].T / Hs))[:, 0]

    # 5) OC update
    l1 = 0.0; l2 = 1e9; eps = 1e-9
    xnew = np.zeros_like(x)
    while (l2 - l1) / (l1 + l2 + eps) > 1e-3:
        lmid = 0.5*(l2 + l1)
        x_candidate = np.maximum(0.0, np.maximum(x - move,
                             np.minimum(1.0, np.minimum(x + move, x * np.sqrt(-dc / dv / lmid)))))
        if (x_candidate.sum() - state['volfrac']*nelx*nely) > 0:
            l1 = lmid
        else:
            l2 = lmid
    xnew[:] = np.maximum(0.0, np.maximum(x - move,
                     np.minimum(1.0, np.minimum(x + move, x * np.sqrt(-dc / dv / lmid)))))

    # 6) Update state and filtered physical densities
    state['xold'] = x.copy()
    state['x'] = xnew
    state['xPhys'] = np.asarray(H * (xnew[np.newaxis].T / Hs))[:, 0]
    state['ce'] = ce
    state['dc'] = dc
    state['dv'] = dv

    change = np.max(np.abs(state['x'] - state['xold']))
    info = {'obj': float(obj), 'change': float(change)}
    return state, info
