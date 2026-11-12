"""
GCSF — Geometric/Graph Connection Spectral Features (Python port).

Dimension-general (any d) positional encoding for directed spatial graphs,
ported from the MATLAB reference implementation. Uses only numpy + scipy for
computation and (optionally) torch / torch_geometric for output.

Three levels of embedding are produced:
    node_pe  : per-node local spectral + SE(d)-invariant geometric features
    sync_pe  : globally synchronised encoding via a (unitary) connection Laplacian
    graph_pe : permutation-invariant graph-level readout

SE(d) invariance is obtained exactly as in the MATLAB code: equivariant
geometric quantities are computed per patch and contracted to invariant
scalars (Gram matrices / norms of spectral-geometric frame vectors, and the
eigenvalues of a local covariance/gyration tensor). Everything is defined for
any dimension d (only inner products, no cross products).

Notes on the port:
  * MATLAB is 1-indexed; here everything is 0-indexed internally.
  * eigs(...,'smallestreal') / eigs(...,'largestreal') -> scipy.sparse.linalg.eigsh
    with sigma/which, falling back to dense numpy.linalg.eig(h) on failure,
    mirroring the try/catch fallbacks in the original.
  * Complex Hermitian patches use eigh (dense) or eigsh on the Hermitian
    operator; both return real eigenvalues.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh, ArpackNoConvergence

EPS = np.finfo(float).eps


# ======================================================================
# Parameters
# ======================================================================
@dataclass
class GCSFParams:
    # Core
    k: int = 8
    r: int = 2
    T: Sequence[float] = (0.1, 1.0, 10.0)
    alpha: float = 1.0
    self_tuned: bool = True
    sigma_k: int = 8

    # Diffusion
    use_laplacian: bool = True
    drop_trivial: bool = True

    # Sync
    compute_sync: bool = False
    sync_dim: int = 1
    sync_confidence: bool = True
    conf_tau: float = 0.5

    # Graph embedding
    graph_include_counts: bool = True
    graph_include_sync: bool = False
    graph_valid_only: bool = True

    # Magnetic
    use_magnetic: bool = False
    magnetic_q: float = 0.25
    magnetic_feat: str = "reim"  # 'reim' or 'abs'

    # [N1] canonical frame degeneracy tolerance
    deg_tol: float = 1e-4

    # [N3] adaptive local q
    adaptive_q: bool = False

    # [N4] spectral gap truncation
    spectral_gap_k: bool = False
    gap_tol: float = 0.10

    # [N5] local flow imbalance feature
    include_flow_imbalance: bool = False

    # [N6] SE(d)-invariant geometric features
    geom_features: bool = True
    geom_gram: bool = True
    geom_cov: bool = True
    geom_kframe: int = 4

    # [N7] upgraded readout
    readout_cov: bool = True
    readout_spectral: bool = True
    readout_nquant: int = 8

    def __post_init__(self):
        if self.graph_include_sync:
            self.compute_sync = True
        self.T = np.asarray(self.T, dtype=float).ravel()


@dataclass
class GCSFResult:
    node_pe: np.ndarray
    sync_pe: Optional[np.ndarray]
    graph_pe: np.ndarray
    coords: np.ndarray
    edge_index: np.ndarray  # (2, E) directed, 0-indexed


# ======================================================================
# Main entry point
# ======================================================================
def gcsf_embed(
    coords: np.ndarray,
    edges: np.ndarray,
    params: Optional[GCSFParams] = None,
    index_base: Union[int, str] = 0,
) -> GCSFResult:
    """Compute GCSF embeddings.

    Parameters
    ----------
    coords     : (n, d) float array of node positions (any dimension d).
    edges      : (E, 2) int array of directed edges. For an undirected graph,
                 list both directions. A PyG ``edge_index`` of shape (2, E)
                 should be passed as ``edge_index.T``.
    params     : GCSFParams.
    index_base : 0 (default, PyG/NumPy convention), 1, or "auto". "auto" only
                 accepts edge lists whose base is unambiguous (index 0 present
                 -> 0-based; index n present -> 1-based) and raises otherwise.

    Returns
    -------
    GCSFResult with node_pe, sync_pe (or None), graph_pe.
    """
    if params is None:
        params = GCSFParams()

    coords = np.asarray(coords, dtype=float)
    n, d = coords.shape

    src, dst = _resolve_edge_index(edges, n, index_base)

    A_dir = sp.csr_matrix(
        (np.ones(len(src)), (src, dst)), shape=(n, n)
    )
    A_dir = _spones(A_dir)
    A_und = _spones(A_dir + A_dir.T)
    n_unique_edges = int(sp.triu(A_und).nnz)

    if params.self_tuned:
        sigma = _estimate_sigma_self_tuned(coords, A_und, params.sigma_k)
        W_dir = _build_W_self_tuned_directed(coords, src, dst, sigma, n)
    else:
        W_dir = _build_W_alpha_directed(coords, src, dst, params.alpha, n)

    if params.use_magnetic:
        W_sym = 0.5 * (W_dir + W_dir.T)
        A_nbr = A_und
    else:
        W_sym = W_dir.maximum(W_dir.T)
        A_nbr = _spones(W_sym)

    A_nbr = A_nbr.tocsr()
    nbrs: List[np.ndarray] = [A_nbr.indices[A_nbr.indptr[u]:A_nbr.indptr[u + 1]]
                              for u in range(n)]

    # --- feature dimensionality bookkeeping ---
    k = params.k
    T = params.T
    M = T.size

    Dspec = 2 * k * M if (params.use_magnetic and params.magnetic_feat.lower() == "reim") else k * M
    Dfi = int(params.include_flow_imbalance)

    kf = max(0, min(params.geom_kframe, k))
    Dgram = 0
    Dcov = 0
    if params.geom_features:
        if params.geom_gram:
            Dgram = kf * (kf + 1) // 2 + kf
        if params.geom_cov:
            Dcov = d
    Dgeom = Dgram + Dcov

    Dnode = Dspec + Dfi + Dgeom
    node_pe = np.zeros((n, Dnode))

    A_dir_csr = A_dir.tocsr()
    W_dir_csr = W_dir.tocsr()
    W_sym_csr = W_sym.tocsr()

    # cache for sync
    cache_nodes: List[Optional[np.ndarray]] = [None] * n
    cache_V: List[Optional[np.ndarray]] = [None] * n
    cache_mu: List[Optional[np.ndarray]] = [None] * n

    mu_store = np.full((n, k), np.nan)

    for i in range(n):
        sub = _rhop_nodes(i, nbrs, params.r)
        if sub.size <= 2:
            continue

        fi_i = None
        if params.include_flow_imbalance:
            Ap_dir_full = _submatrix_dense(A_dir_csr, sub)
            fi_i = _local_flow_imbalance(Ap_dir_full)

        if params.use_magnetic:
            Wp_dir = _submatrix_dense(W_dir_csr, sub)
            Ws = 0.5 * (Wp_dir + Wp_dir.T)
            if np.count_nonzero(Ws) == 0:
                continue
            Ap_dir = _submatrix_dense(A_dir_csr, sub)

            if params.adaptive_q:
                fi_patch = _local_flow_imbalance(Ap_dir)
                q_local = params.magnetic_q * fi_patch
            else:
                q_local = params.magnetic_q

            Theta = 2.0 * np.pi * q_local * (Ap_dir - Ap_dir.T)
            Hq = Ws * np.exp(1j * Theta)

            di = Ws.sum(axis=1)
            di[di == 0] = EPS
            Dm12 = 1.0 / np.sqrt(di)
            S = (Dm12[:, None] * Hq) * Dm12[None, :]
            L = np.eye(S.shape[0]) - S

            k_req = min(k + int(params.drop_trivial), L.shape[0])
            Vall, muall = _bottomk_eigs_hermitian_psd(L, k_req)

            if params.drop_trivial and k_req >= 2:
                V = Vall[:, 1:k_req]
                mu = muall[1:k_req]
            else:
                V = Vall
                mu = muall

            V = _canonical_frame_complex(V, mu, params.deg_tol)

            if params.spectral_gap_k:
                k_eff = _spectral_gap_truncate(mu, k, params.gap_tol)
                V[:, k_eff:] = 0
                mu = mu.copy()
                mu[k_eff:] = 0

            V = _pad_cols(V, k)
            mu = _pad_vec(mu, k)

        else:
            Wi = _submatrix_dense(W_sym_csr, sub)
            if np.count_nonzero(Wi) == 0:
                continue

            di = Wi.sum(axis=1)
            di[di == 0] = EPS
            Dm12 = 1.0 / np.sqrt(di)
            S = (Dm12[:, None] * Wi) * Dm12[None, :]

            if params.use_laplacian:
                L = np.eye(S.shape[0]) - S
                k_req = min(k + int(params.drop_trivial), L.shape[0])
                Vall, muall = _bottomk_eigs_psd(L, k_req)

                if params.drop_trivial and k_req >= 2:
                    V = Vall[:, 1:k_req]
                    mu = muall[1:k_req]
                else:
                    V = Vall
                    mu = muall

                V = _canonical_frame_real(V, mu, params.deg_tol)

                if params.spectral_gap_k:
                    k_eff = _spectral_gap_truncate(mu, k, params.gap_tol)
                    V[:, k_eff:] = 0
                    mu = mu.copy()
                    mu[k_eff:] = 0

                V = _pad_cols(V, k)
                mu = _pad_vec(mu, k)
            else:
                V0, lam0 = _topk_eigs_sym(S, min(k, S.shape[0]))
                mu0 = 1.0 - lam0
                V0 = _canonical_frame_real(V0, mu0, params.deg_tol)
                V = _pad_cols(V0, k)
                mu = _pad_vec(mu0, k)

        center = np.where(sub == i)[0]
        if center.size == 0:
            continue
        center = center[0]
        v_center = V[center, :]

        # --- multi-time diffusion coordinate (spectral block) ---
        phi_blocks = np.zeros((1, k * M), dtype=V.dtype)
        for m in range(M):
            scale = np.exp(-T[m] * mu.real)
            phi_blocks[0, m * k:(m + 1) * k] = v_center * scale

        if params.use_magnetic:
            if params.magnetic_feat.lower() == "reim":
                phi_spec = np.concatenate([phi_blocks.real, phi_blocks.imag], axis=1).ravel()
            else:
                phi_spec = np.abs(phi_blocks).ravel()
        else:
            phi_spec = phi_blocks.real.ravel()

        # --- [N6] geometric features ---
        phi_geom = np.array([])
        if params.geom_features:
            P = coords[sub, :] - coords[i, :]
            wloc = di.astype(float).copy()
            wloc = wloc / max(wloc.sum(), EPS)
            phi_geom = _geom_invariants(
                P, V, mu, wloc, kf, params.geom_gram, params.geom_cov, d
            )

        # --- assemble ---
        parts = [phi_spec]
        if params.include_flow_imbalance:
            parts.append(np.array([fi_i]))
        if phi_geom.size > 0:
            parts.append(phi_geom)
        phi_i = np.concatenate(parts)

        node_pe[i, :] = phi_i.real

        cache_nodes[i] = sub
        cache_V[i] = V
        cache_mu[i] = mu.ravel()
        mu_store[i, :] = mu.real.ravel()

    # ----------------------------
    # Optional sync
    # ----------------------------
    sync_pe = None
    if params.compute_sync:
        sync_pe = _compute_sync(
            A_und, W_sym_csr, cache_nodes, cache_V, k, params, n
        )

    # ----------------------------
    # Graph readout
    # ----------------------------
    graph_pe = _graph_readout(node_pe, sync_pe, mu_store, params, n, n_unique_edges)

    return GCSFResult(
        node_pe=node_pe,
        sync_pe=sync_pe,
        graph_pe=graph_pe,
        coords=coords,
        edge_index=np.vstack([src, dst]),
    )


# ======================================================================
# Sync driver
# ======================================================================
def _compute_sync(A_und, W_sym_csr, cache_nodes, cache_V, k, params, n):
    tri = sp.triu(A_und, k=1).tocoo()
    srcU = tri.row
    dstU = tri.col
    m_u = srcU.size

    Qcell: List[Optional[np.ndarray]] = [None] * m_u
    okEdge = np.zeros(m_u, dtype=bool)
    wMult = np.ones(m_u)

    for e in range(m_u):
        ii = int(srcU[e])
        jj = int(dstU[e])
        Q, ok, res = _transport_Qij(ii, jj, cache_nodes, cache_V)
        if ok:
            Qcell[e] = Q
            okEdge[e] = True
            if params.sync_confidence:
                tau = max(params.conf_tau, EPS)
                wMult[e] = np.exp(-((res / tau) ** 2))

    sync_dim = max(1, params.sync_dim)
    H, used = _build_connection_laplacian(
        srcU, dstU, W_sym_csr, Qcell, okEdge, wMult, k, n
    )

    if used < 1:
        warnings.warn("compute_sync: no usable edges. Returning zeros.")
        sync_pe_c = np.zeros((n, k * sync_dim), dtype=complex)
    else:
        Vec = _smallest_eigs_hermitian(H, sync_dim)
        sync_pe_c = np.zeros((n, k * sync_dim), dtype=complex)
        for dd in range(sync_dim):
            vd = Vec[:, dd]
            Bd = vd.reshape(n, k)  # MATLAB reshape([k,n]).' -> row-major (n,k)
            sync_pe_c[:, dd * k:(dd + 1) * k] = Bd

    if params.use_magnetic and params.magnetic_feat.lower() == "reim":
        sync_pe = np.concatenate([sync_pe_c.real, sync_pe_c.imag], axis=1)
    else:
        sync_pe = sync_pe_c.real
    return sync_pe


# ======================================================================
# [N6] SE(d)-invariant geometric features
# ======================================================================
def _geom_invariants(P, V, mu, wloc, kf, use_gram, use_cov, d):
    feats = []
    Vr = np.real(V)

    if use_gram and kf >= 1:
        Gf = Vr[:, :kf].T @ P              # (kf, d) equivariant rows
        Gram = Gf @ Gf.T                   # (kf, kf) invariant
        iu = np.triu_indices(kf)
        gram_vec = Gram[iu]
        norms = np.sqrt(np.maximum(np.diag(Gram), 0.0))
        feats.append(gram_vec)
        feats.append(norms)

    if use_cov:
        Pw = P * wloc[:, None]
        C = Pw.T @ P                        # (d, d)
        C = 0.5 * (C + C.T)
        ev = np.sort(np.real(np.linalg.eigvals(C)))[::-1]
        if ev.size < d:
            ev = np.concatenate([ev, np.zeros(d - ev.size)])
        feats.append(ev[:d])

    if not feats:
        return np.array([])
    return np.concatenate(feats)


# ======================================================================
# [N1] Canonical frames
# ======================================================================
def _canonical_frame_real(V, mu, tol):
    if V.size == 0:
        return V
    V = V.copy()
    mu = np.real(mu).ravel()
    k = V.shape[1]
    used = np.zeros(k, dtype=bool)

    for j in range(k):
        if used[j]:
            continue
        block = np.where(np.abs(mu - mu[j]) < tol)[0]
        block = block[~used[block]]
        if block.size == 1:
            b = block[0]
            ii = np.argmax(np.abs(V[:, b]))
            if V[ii, b] < 0:
                V[:, b] = -V[:, b]
        else:
            Q, _ = np.linalg.qr(V[:, block])
            for bb in range(Q.shape[1]):
                ii = np.argmax(np.abs(Q[:, bb]))
                if Q[ii, bb] < 0:
                    Q[:, bb] = -Q[:, bb]
            V[:, block] = Q
        used[block] = True
    return V


def _canonical_frame_complex(V, mu, tol):
    if V.size == 0:
        return V
    V = V.copy()
    mu = np.real(mu).ravel()
    k = V.shape[1]
    used = np.zeros(k, dtype=bool)

    for j in range(k):
        if used[j]:
            continue
        block = np.where(np.abs(mu - mu[j]) < tol)[0]
        block = block[~used[block]]
        if block.size > 1:
            Q, _ = np.linalg.qr(V[:, block])
            V[:, block] = Q
        used[block] = True

    for c in range(k):
        idx = np.argmax(np.abs(V[:, c]))
        ph = np.angle(V[idx, c])
        V[:, c] = V[:, c] * np.exp(-1j * ph)
    return V


# ======================================================================
# [N3] Local directed flow imbalance
# ======================================================================
def _local_flow_imbalance(Ap_dir):
    skew = Ap_dir - Ap_dir.T
    sym = Ap_dir + Ap_dir.T
    denom = np.linalg.norm(sym, "fro")
    if denom < EPS:
        return 0.0
    return np.linalg.norm(skew, "fro") / denom


# ======================================================================
# [N4] Spectral-gap-based effective k
# ======================================================================
def _spectral_gap_truncate(mu, k_max, gap_tol):
    mu = np.real(mu).ravel()
    k_max = min(k_max, mu.size)
    if k_max <= 1:
        return k_max
    mu_mean = np.mean(mu[:k_max])
    if mu_mean < EPS:
        return k_max
    gaps = np.diff(mu[:k_max])
    threshold = gap_tol * mu_mean
    idx = np.where(gaps > threshold)[0]
    if idx.size == 0:
        return k_max
    return idx[0] + 1  # +1 because gap idx points to the eigenvalue below the gap


# ======================================================================
# [N7] Graph readout
# ======================================================================
def _graph_readout(node_pe, sync_pe, mu_store, params, n_nodes, n_unique_edges):
    Xn = node_pe
    if params.graph_valid_only:
        valid = np.any(np.abs(Xn) > 0, axis=1)
        Xn = Xn[valid, :]
    if Xn.shape[0] == 0:
        Xn = np.zeros((1, node_pe.shape[1]))

    g_moments = np.concatenate([
        Xn.mean(axis=0),
        Xn.std(axis=0, ddof=0),
        Xn.max(axis=0),
    ])

    g_cov = np.array([])
    if params.readout_cov:
        Xc = Xn - Xn.mean(axis=0)
        C = (Xc.T @ Xc) / max(Xn.shape[0] - 1, 1)
        iu = np.triu_indices(C.shape[0])
        g_cov = C[iu]

    g_spec = np.array([])
    if params.readout_spectral and mu_store.size > 0:
        mu_all = mu_store.ravel()
        mu_all = mu_all[~np.isnan(mu_all)]
        nq = max(2, params.readout_nquant)
        if mu_all.size == 0:
            g_spec = np.zeros(nq + 2)
        else:
            qs = np.linspace(0.0, 1.0, nq)
            g_spec = np.quantile(mu_all, qs)
            g_spec = np.concatenate([g_spec, [mu_all.mean(), mu_all.std(ddof=0)]])

    if params.graph_include_sync and sync_pe is not None and sync_pe.size > 0:
        Xs = sync_pe
        if params.graph_valid_only:
            validS = np.any(np.abs(Xs) > 0, axis=1)
            Xs = Xs[validS, :]
        if Xs.shape[0] == 0:
            g_sync = np.zeros(3 * sync_pe.shape[1])
        else:
            g_sync = np.concatenate([
                Xs.mean(axis=0), Xs.std(axis=0, ddof=0), Xs.max(axis=0)
            ])
    else:
        g_sync = np.array([])

    if params.graph_include_counts:
        dens = n_unique_edges / max(n_nodes * (n_nodes - 1) / 2, 1)
        g_counts = np.array([np.log1p(n_nodes), np.log1p(n_unique_edges), dens])
    else:
        g_counts = np.array([])

    return np.concatenate([g_counts, g_moments, g_cov, g_spec, g_sync])


# ======================================================================
# Sigma estimation & weight builders (dimension-general)
# ======================================================================
def _estimate_sigma_self_tuned(coords, A_und, k_sigma):
    n = coords.shape[0]
    A_und = A_und.tocsr()
    sigma = np.zeros(n)
    for i in range(n):
        nb = A_und.indices[A_und.indptr[i]:A_und.indptr[i + 1]]
        if nb.size == 0:
            sigma[i] = 1.0
            continue
        dist = np.sqrt(np.sum((coords[nb, :] - coords[i, :]) ** 2, axis=1))
        dist.sort()
        idx = min(k_sigma, dist.size)
        sigma[i] = max(dist[max(idx - 1, 0)], EPS)
    return np.maximum(sigma, EPS)


def _build_W_self_tuned_directed(coords, src, dst, sigma, n):
    rij2 = np.sum((coords[src, :] - coords[dst, :]) ** 2, axis=1)
    denom = np.maximum(sigma[src] * sigma[dst], EPS)
    vals = np.exp(-rij2 / denom)
    return sp.csr_matrix((vals, (src, dst)), shape=(n, n))


def _build_W_alpha_directed(coords, src, dst, alpha, n):
    a2 = max(alpha * alpha, EPS)
    rij2 = np.sum((coords[src, :] - coords[dst, :]) ** 2, axis=1)
    vals = np.exp(-rij2 / a2)
    return sp.csr_matrix((vals, (src, dst)), shape=(n, n))


# ======================================================================
# Neighbourhoods
# ======================================================================
def _rhop_nodes(i, nbrs, r):
    n = len(nbrs)
    visited = np.zeros(n, dtype=bool)
    visited[i] = True
    frontier = np.array([i])
    for _ in range(r):
        if frontier.size == 0:
            break
        mask = np.zeros(n, dtype=bool)
        for u in frontier:
            nu = nbrs[u]
            if nu.size > 0:
                mask[nu] = True
        mask = mask & ~visited
        visited |= mask
        frontier = np.where(mask)[0]
    return np.where(visited)[0]


# ======================================================================
# Eigensolvers (with dense fallbacks mirroring the MATLAB try/catch)
# ======================================================================
def _bottomk_eigs_psd(L, k):
    n = L.shape[0]
    k_eff = min(k, n)
    if k_eff < n:
        try:
            mu0, V0 = eigsh(L, k=k_eff, which="SA")
        except (ArpackNoConvergence, Exception):
            mu0, V0 = np.linalg.eigh(L)
    else:
        mu0, V0 = np.linalg.eigh(L)
    mu0 = np.real(mu0)
    V0 = np.real(V0)
    order = np.argsort(mu0)
    return V0[:, order], mu0[order]


def _bottomk_eigs_hermitian_psd(L, k):
    n = L.shape[0]
    k_eff = min(k, n)
    if k_eff < n:
        try:
            mu0, V0 = eigsh(L, k=k_eff, which="SA")
        except (ArpackNoConvergence, Exception):
            mu0, V0 = np.linalg.eigh(L)
    else:
        mu0, V0 = np.linalg.eigh(L)
    mu0 = np.real(mu0)
    order = np.argsort(mu0)
    return V0[:, order], mu0[order]


def _smallest_eigs_hermitian(H, s):
    Nk = H.shape[0]
    s = min(s, Nk)
    try:
        # eigsh requires k < N; fall back to dense otherwise
        if s < Nk:
            _, Vec = eigsh(H, k=s, which="SA")
        else:
            raise ValueError
    except Exception:
        Hd = H.toarray() if sp.issparse(H) else H
        w, Vfull = np.linalg.eigh(Hd)
        order = np.argsort(np.real(w))
        Vec = Vfull[:, order[:s]]
    return Vec


def _topk_eigs_sym(S, k):
    n = S.shape[0]
    k_eff = min(k, n)
    if k_eff < n:
        try:
            lam0, V0 = eigsh(S, k=k_eff, which="LA")
        except (ArpackNoConvergence, Exception):
            lam0, V0 = np.linalg.eigh(S)
    else:
        lam0, V0 = np.linalg.eigh(S)
    lam0 = np.real(lam0)
    V0 = np.real(V0)
    order = np.argsort(lam0)[::-1]
    return V0[:, order], lam0[order]


def _pad_cols(V, k):
    n, k0 = V.shape
    Vp = np.zeros((n, k), dtype=V.dtype)
    kk = min(k0, k)
    Vp[:, :kk] = V[:, :kk]
    return Vp


def _pad_vec(v, k):
    v = np.asarray(v).ravel()
    vp = np.zeros(k, dtype=v.dtype)
    kk = min(v.size, k)
    vp[:kk] = v[:kk]
    return vp


# ======================================================================
# Sync internals — UNITARY / orthogonal Procrustes
# ======================================================================
def _transport_Qij(i, j, cache_nodes, cache_V):
    ok = False
    Q = None
    res = np.inf
    ni = cache_nodes[i]
    nj = cache_nodes[j]
    if ni is None or nj is None:
        return Q, ok, res
    Vi_full = cache_V[i]
    Vj_full = cache_V[j]
    if Vi_full is None or Vj_full is None:
        return Q, ok, res

    common, ia, ib = np.intersect1d(ni, nj, return_indices=True)
    k = Vi_full.shape[1]
    if common.size < k:
        return Q, ok, res

    Vi = Vi_full[ia, :]
    Vj = Vj_full[ib, :]

    X = Vj.conj().T @ Vi              # Hermitian-correct cross-correlation
    U, _, Wh = np.linalg.svd(X)       # numpy returns Wh = W^H
    Q = Wh.conj().T @ U.conj().T      # == Wsv * U'  in MATLAB (Wsv=Wh', U as-is)
    # In MATLAB: [U,~,Wsv]=svd(X); Q = Wsv*U'.  numpy: U,S,Wh with Wh=Wsv'.
    # So Wsv = Wh.conj().T, U' = U.conj().T -> Q = Wh.conj().T @ U.conj().T.

    R = Vj - Vi @ Q
    res = np.linalg.norm(R, "fro") / np.sqrt(max(R.shape[0], 1))
    ok = True
    return Q, ok, res


def _build_connection_laplacian(src, dst, Wsym, Qcell, okEdge, wMult, k, n):
    Nk = n * k
    Wsym = Wsym.tocsr()
    I_list = []
    J_list = []
    V_list = []
    used_edges = 0

    is_complex_Q = any(
        okEdge[e] and Qcell[e] is not None and np.iscomplexobj(Qcell[e])
        for e in range(len(src))
    )

    rr, cc = np.meshgrid(np.arange(k), np.arange(k), indexing="ij")
    rr = rr.ravel()
    cc = cc.ravel()

    for e in range(len(src)):
        if not okEdge[e]:
            continue
        i = int(src[e])
        j = int(dst[e])
        w = Wsym[i, j]
        if w <= 0:
            continue
        Q = Qcell[e]
        if Q is None:
            continue
        c = max(wMult[e], 0.0)
        w = w * c
        if w <= 0:
            continue
        used_edges += 1

        ii = i * k + np.arange(k)
        jj = j * k + np.arange(k)

        # diagonal (degree) blocks: w * I
        I_list.append(ii)
        J_list.append(ii)
        V_list.append(w * np.ones(k))
        I_list.append(jj)
        J_list.append(jj)
        V_list.append(w * np.ones(k))

        # off-diagonal (i,j): -w * Q
        Hij = (-w * Q).ravel(order="F")  # match MATLAB column-major Hij(:)
        I_list.append(ii[rr])
        J_list.append(jj[cc])
        V_list.append(Hij)

        # off-diagonal (j,i): -w * Q'
        Hji = (-w * Q.conj().T).ravel(order="F")
        I_list.append(jj[rr])
        J_list.append(ii[cc])
        V_list.append(Hji)

    if used_edges == 0:
        return sp.csr_matrix((Nk, Nk)), 0

    I = np.concatenate(I_list)
    J = np.concatenate(J_list)
    Vv = np.concatenate(V_list)

    if is_complex_Q:
        H = sp.csr_matrix((Vv, (I, J)), shape=(Nk, Nk))
        H = (H + H.conj().T) / 2
    else:
        H = sp.csr_matrix((np.real(Vv), (I, J)), shape=(Nk, Nk))
        H = (H + H.T) / 2
    return H, used_edges


# ======================================================================
# Edge-index validation
# ======================================================================
def _resolve_edge_index(edges, n, index_base=0):
    """Return 0-based (src, dst) arrays, validating every index.

    Never guesses silently and never clips: an edge list whose indexing cannot
    be determined, or that references a node outside 0..n-1, raises ValueError.
    """
    E = np.asarray(edges)
    if E.size == 0:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty.copy()
    if E.ndim != 2 or E.shape[1] != 2:
        raise ValueError(
            f"edges must have shape (E, 2); got {E.shape}. "
            "For a PyG edge_index of shape (2, E), pass edge_index.T."
        )
    if not np.issubdtype(E.dtype, np.integer):
        if not np.all(np.equal(np.mod(E, 1), 0)):
            raise ValueError("edge indices must be integers.")
    E = E.astype(np.int64)

    if isinstance(index_base, str):
        if index_base.lower() != "auto":
            raise ValueError(f"index_base must be 0, 1, or 'auto'; got {index_base!r}.")
        lo, hi = int(E.min()), int(E.max())
        if lo == 0 and hi == n:
            raise ValueError(
                f"edge indices contain both 0 and n={n}, so they are neither "
                "0-based nor 1-based."
            )
        if lo == 0:
            base = 0
        elif hi == n:
            base = 1
        else:
            raise ValueError(
                f"cannot tell whether edge indices are 0- or 1-based (neither 0 "
                f"nor n={n} appears). Pass index_base=0 or index_base=1."
            )
    elif index_base in (0, 1):
        base = int(index_base)
    else:
        raise ValueError(f"index_base must be 0, 1, or 'auto'; got {index_base!r}.")

    src = E[:, 0] - base
    dst = E[:, 1] - base
    bad = (src < 0) | (src >= n) | (dst < 0) | (dst >= n)
    if bad.any():
        raise ValueError(
            f"{int(bad.sum())} edge(s) reference nodes outside 0..{n - 1} "
            f"after applying index_base={base}."
        )
    return src, dst


# ======================================================================
# Small sparse helpers
# ======================================================================
def _spones(A):
    A = A.tocoo()
    return sp.csr_matrix((np.ones(A.nnz), (A.row, A.col)), shape=A.shape)


def _submatrix_dense(A_csr, idx):
    return np.asarray(A_csr[np.ix_(idx, idx)].todense())


# ======================================================================
# I/O helpers and torch_geometric export
# ======================================================================
def load_graph_csv(node_file, edge_file):
    """Load node/edge CSVs matching the MATLAB readmatrix layout.

    node_file: first column is an id, remaining columns are coordinates.
    edge_file: two columns (src, dst), returned as stored; the index base is
               resolved by gcsf_embed(..., index_base=...).
    """
    node_data = np.loadtxt(node_file, delimiter=",", ndmin=2)
    coords = node_data[:, 1:]
    edges = np.loadtxt(edge_file, delimiter=",", ndmin=2).astype(np.int64)
    return coords, edges


def to_pyg_data(result: GCSFResult):
    """Convert a GCSFResult into a torch_geometric Data object.

    node_pe -> data.x
    graph_pe -> data.graph_pe (graph-level attribute, shape (1, D))
    sync_pe -> data.sync_pe (if present)
    coords  -> data.pos
    edges   -> data.edge_index
    """
    import torch
    from torch_geometric.data import Data

    data = Data(
        x=torch.as_tensor(result.node_pe, dtype=torch.float32),
        pos=torch.as_tensor(result.coords, dtype=torch.float32),
        edge_index=torch.as_tensor(result.edge_index, dtype=torch.long),
    )
    data.graph_pe = torch.as_tensor(result.graph_pe, dtype=torch.float32).unsqueeze(0)
    if result.sync_pe is not None:
        data.sync_pe = torch.as_tensor(result.sync_pe, dtype=torch.float32)
    return data


# ======================================================================
# Batch runner
# ======================================================================
def batch_gcsf_embed(input_folder, output_folder=None, params=None,
                     to_pyg=False, write_node_csv=False, index_base="auto"):
    """Process all *_nodes.csv / *_edges.csv pairs in a folder.

    Returns a dict {graph_id: GCSFResult} (or PyG Data if to_pyg=True).
    If output_folder is given, graph/sync/node CSVs are written like the
    MATLAB batch runner. index_base is passed to gcsf_embed for every graph;
    set it to 0 or 1 explicitly if any edge file is ambiguous.
    """
    import glob
    import os

    if params is None:
        params = GCSFParams()
    if output_folder and not os.path.isdir(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    node_files = sorted(glob.glob(os.path.join(input_folder, "*_nodes.csv")))
    if not node_files:
        warnings.warn(f"No *_nodes.csv files found in: {input_folder}")
        return {}

    results = {}
    n_done = n_skip = n_fail = 0
    for idx, npath in enumerate(node_files, 1):
        base = os.path.basename(npath)
        graph_id = base[: -len("_nodes.csv")]
        epath = os.path.join(input_folder, f"{graph_id}_edges.csv")
        if not os.path.exists(epath):
            warnings.warn(f"Skipping {graph_id} (missing edge file)")
            n_skip += 1
            continue
        print(f"[{idx}/{len(node_files)}] GCSF embedding: {graph_id}")
        try:
            coords, edges = load_graph_csv(npath, epath)
            res = gcsf_embed(coords, edges, params, index_base=index_base)
            if output_folder:
                if write_node_csv:
                    np.savetxt(os.path.join(output_folder, f"{graph_id}_nodePE.csv"),
                               res.node_pe, delimiter=",")
                if params.compute_sync and res.sync_pe is not None:
                    np.savetxt(os.path.join(output_folder, f"{graph_id}_syncPE.csv"),
                               res.sync_pe, delimiter=",")
                np.savetxt(os.path.join(output_folder, f"{graph_id}_graphPE.csv"),
                           res.graph_pe[None, :], delimiter=",")
            results[graph_id] = to_pyg_data(res) if to_pyg else res
            n_done += 1
        except Exception as ex:  # noqa: BLE001
            warnings.warn(f"FAILED on {graph_id}: {ex}")
            n_fail += 1
    print(f"Batch complete. Done={n_done}, Skipped={n_skip}, "
          f"Failed={n_fail} (Total={len(node_files)})")
    return results
