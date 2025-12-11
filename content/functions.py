import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

# Element stiffness matrix for standard 4-node Q4 element
def element_stiffness_matrix(E=1.0, nu=0.3):
    k = np.array([
        [ 0.6667, -0.3333, -0.3333, -0.0   ],
        [-0.3333,  0.6667,  0.0   , -0.3333],
        [-0.3333,  0.0   ,  0.6667, -0.3333],
        [-0.0   , -0.3333, -0.3333,  0.6667]
    ])
    return k * E

# Generate edofMat for nelx x nely mesh
def generate_edofMat(nelx, nely):
    nodenrs = np.arange((nelx+1)*(nely+1)).reshape((nely+1, nelx+1))
    edofMat = np.zeros((nelx*nely, 8), dtype=int)
    for elx in range(nelx):
        for ely in range(nely):
            el = ely + elx*nely
            n1 = nodenrs[ely, elx]
            n2 = nodenrs[ely, elx+1]
            n3 = nodenrs[ely+1, elx+1]
            n4 = nodenrs[ely+1, elx]
            edofMat[el, :] = np.array([
                2*n1, 2*n1+1, 2*n2, 2*n2+1,
                2*n3, 2*n3+1, 2*n4, 2*n4+1
            ])
    return edofMat

# Build global stiffness matrix
def FE(nelx, nely, xPhys, edofMat, KE, freedofs, f, Emin, Emax, p):
    # Assembled using COO sparse style
    Kvals = []
    I = []
    J = []

    for el in range(nelx*nely):
        x_phys_el = xPhys.flatten()[el]
        E_el = Emin + (x_phys_el**p) * (Emax - Emin)
        ke = KE * E_el
        edofs = edofMat[el]
        for i in range(8):
            for j in range(8):
                I.append(edofs[i])
                J.append(edofs[j])
                Kvals.append(ke[i, j])

    K = coo_matrix((Kvals, (I, J))).tocsr()
    u = np.zeros((2*(nelx+1)*(nely+1), 1))
    u[freedofs, 0] = spsolve(K[freedofs][:, freedofs], f[freedofs, 0])
    return u

# Build density filter (sensitivity filter)
def build_filter(nelx, nely, rmin):
    n = nelx * nely
    iH = []
    jH = []
    sH = []

    for i in range(nelx):
        for j in range(nely):
            row = i * nely + j
            imin = int(max(i - (rmin - 1), 0))
            imax = int(min(i + (rmin - 1), nelx - 1))
            jmin = int(max(j - (rmin - 1), 0))
            jmax = int(min(j + (rmin - 1), nely - 1))
            for k in range(imin, imax + 1):
                for l in range(jmin, jmax + 1):
                    col = k * nely + l
                    weight = rmin - np.sqrt((i - k)**2 + (j - l)**2)
                    if weight > 0:
                        iH.append(row)
                        jH.append(col)
                        sH.append(weight)

    H = coo_matrix((sH, (iH, jH)), shape=(n, n)).tocsr()
    Hs = np.array(H.sum(axis=1)).flatten()
    return H, Hs

# Apply sensitivity filter
def filter_sensitivities(dc, H, Hs):
    dc_filtered = (H @ (dc.flatten())) / Hs
    return dc_filtered.reshape(dc.shape)

# Compliance objective and sensitivities
def compute_objective(nelx, nely, xPhys, u, KE, edofMat, p, Emin, Emax):
    obj = 0.0
    dc = np.zeros((nely, nelx))

    for elx in range(nelx):
        for ely in range(nely):
            el = elx*nely + ely
            edofs = edofMat[el]
            u_el = u[edofs, 0]
            xval = xPhys[ely, elx]
            dE = p * xval**(p-1) * (Emax - Emin)
            obj += (Emin + xval**p * (Emax - Emin)) * (u_el.T @ KE @ u_el)
            dc[ely, elx] = -dE * (u_el.T @ KE @ u_el)

    return obj, dc

# Optimality Criteria update
def OC_update(x, dc, volfrac, move, l1=0, l2=1e9):
    xnew = np.zeros_like(x)
    eps = 1e-3

    # Bisection to find lambda
    while (l2 - l1) / (l1 + l2 + eps) > 1e-3:
        lmid = 0.5 * (l2 + l1)
        x_candidate = np.maximum(0.0, np.maximum(x - move, np.minimum(1.0, np.minimum(x + move, x * np.sqrt(-dc / lmid)))))
        if x_candidate.mean() - volfrac > 0:
            l1 = lmid
        else:
            l2 = lmid

    xnew = np.maximum(0.0, np.maximum(x - move, np.minimum(1.0, np.minimum(x + move, x * np.sqrt(-dc / lmid)))))
    return xnew
