import numpy as np

# -----------------------------------------------------------
# Element stiffness matrix for 4-node quad
# -----------------------------------------------------------
def lk(E=1.0, nu=0.3):
    k = np.array([
        1/2 - nu/6, 1/8 + nu/8, -1/4 - nu/12, -1/8 + 3*nu/8,
        -1/4 + nu/12, -1/8 - nu/8, nu/6, 1/8 - 3*nu/8
    ])
    KE = E/(1 - nu**2) * np.array([
        [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
        [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
        [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
        [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
        [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
        [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
        [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
        [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]]
    ])
    return KE


# -----------------------------------------------------------
# Finite element solver
# -----------------------------------------------------------
def FE_solve(nelx, nely, xPhys, penal, KE, free, f,
             Emin=1e-9, Emax=1.0, edofMat=None):

    nel = nelx * nely
    ndof = f.size

    if edofMat is None:
        edofMat = make_edofMat(nelx, nely)

    K = np.zeros((ndof, ndof))

    # SIMP stiffness
    xpen = Emin + xPhys.flatten()**penal * (Emax - Emin)

    for el in range(nel):
        edofs = edofMat[el]
        K[np.ix_(edofs, edofs)] += xpen[el] * KE

    # Solve
    U = np.zeros(ndof)
    Kff = K[np.ix_(free, free)]
    ff = f[free]
    U[free] = np.linalg.solve(Kff, ff)

    return U


# -----------------------------------------------------------
# Compliance + sensitivity
# -----------------------------------------------------------
def compliance_and_sensitivity(U, KE, penal, xPhys,
                               Emin, Emax, H, Hs, edofMat):

    nel = xPhys.size
    u_e = U[edofMat].reshape(nel, 8)
    ce = (u_e @ KE * u_e).sum(1)

    obj = ((Emin + xPhys.flatten()**penal * (Emax - Emin)) * ce).sum()

    dc = (-penal * xPhys.flatten()**(penal - 1) * (Emax - Emin)) * ce
    dc = np.asarray(H * (dc[:, None] / Hs))[:, 0]

    return obj, dc, ce


# -----------------------------------------------------------
# OC update
# -----------------------------------------------------------
def oc(nelx, nely, x, volfrac, dc, dv, g):
    l1, l2 = 0.0, 1e9
    move = 0.2

    for _ in range(80):
        lmid = 0.5 * (l1 + l2)
        x_new = np.maximum(1e-3,
            np.maximum(x - move,
            np.minimum(1.0,
            np.minimum(x + move,
            x * np.sqrt(-dc / dv / lmid)))))

        if g + (dv * (x_new - x)).sum() > 0:
            l1 = lmid
        else:
            l2 = lmid

        if (l2 - l1) / (l1 + l2 + 1e-9) < 1e-3:
            break

    return x_new, g


# -----------------------------------------------------------
# Build edofMat (connectivity)
# -----------------------------------------------------------
def make_edofMat(nelx, nely):
    ndof = 2 * (nelx+1) * (nely+1)
    edofMat = np.zeros((nelx*nely, 8), dtype=int)

    for elx in range(nelx):
        for ely in range(nely):
            el = ely + nely*elx
            n1 = (nely+1)*elx + ely
            n2 = (nely+1)*(elx+1) + ely

            edofMat[el, :] = np.array([
                2*n1,     2*n1+1,
                2*n2,     2*n2+1,
                2*(n2+1), 2*(n2+1)+1,
                2*(n1+1), 2*(n1+1)+1
            ])
    return edofMat
