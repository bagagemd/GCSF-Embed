# GCSF-Embed

**Gauge-consistent, SE(d)-invariant spectral positional encodings for directed spatial graphs.**

GCSF-Embed (**G**eometric/Graph **C**onnection **S**pectral **F**eatures) computes positional
encodings for graphs whose nodes carry coordinates in `R^d` (any dimension) and whose edges may be
**directed**. It fuses per-patch magnetic-Laplacian spectra with global connection-Laplacian
synchronization, and augments both with SE(d)-invariant geometric features derived from equivariant
contractions. The output representation is stable under rigid motions (rotation + translation) in any
dimension, while retaining local structural and geometric information.

---

## Methodology and math

### 1. Weighted directed graph

From node coordinates `x_i ∈ R^d` and a directed edge set, build a distance-based weight matrix.
Weights depend only on Euclidean distances, so they are already rotation/translation invariant:

```
$$W_ij = exp( -||x_i - x_j||^2 / (sigma_i * sigma_j) )      (self-tuned bandwidth)$$
```

with per-node bandwidth `sigma_i` set from the k-th nearest neighbor (self-tuning). A fixed global
bandwidth `alpha` is also available.

### 2. Per-node local spectra (`nodePE`)

For each node `i`, extract its `r`-hop patch and build a normalized Laplacian on that patch.

**Undirected mode** — symmetric normalized Laplacian:

```
L = I - D^{-1/2} W D^{-1/2}
```

with real eigenvectors.

**Magnetic mode** — a Hermitian *magnetic Laplacian* that encodes direction as phase:

```
Theta_ij = 2*pi*q * (A_ij - A_ji)         (antisymmetric phase from direction)
H_q       = W_sym ⊙ exp(i * Theta)         (Hermitian)
L         = I - D^{-1/2} H_q D^{-1/2}
```

The magnetic charge `q` controls how strongly direction is encoded. The bottom-`k` eigenvectors are
taken, and the center node's row is turned into a multi-scale **heat-kernel diffusion coordinate**
across diffusion times `T = {t_1, ..., t_M}`:

```
phi(i)_{m,c} = V_center,c * exp( -t_m * mu_c )
```

For magnetic mode the complex coordinate is split into real/imaginary parts (`reim` features).

### 3. SE(d)-invariant geometric features `[N6]`

This is the key geometric contribution. On each patch, using relative positions
`p_j - p_i` (translation-invariant, rotation-**equivariant**):

- **Spectral–geometric frame vectors** fuse spectral structure with geometry:

  ```
  g_c = sum_j V(j,c) * (p_j - p_i)  ∈ R^d      (rotation-equivariant)
  ```

- **Contract to invariants.** Inner products and norms of equivariant vectors are rotation-invariant
  *scalars* that still encode lengths and angles:

  ```
  Gram_{c,c'} = <g_c, g_c'>          (invariant; encodes angles + lengths)
  ||g_c||                             (invariant)
  eig( sum_j w_j (p_j - p_i)(p_j - p_i)^T )   (local covariance spectrum; anisotropy/shape)
  ```

Under a rigid motion `p -> R p + t` (with `R ∈ SO(d)`), every `g_c -> R g_c`, so all inner products
and all covariance eigenvalues are unchanged. This holds in **any dimension** — only inner products
are used, no cross products. (Verified numerically to machine precision for d = 2, 3, 4, 5, 7.)

### 4. Global synchronization (`syncPE`)

Patch eigenframes are only defined up to a rotation (real) or unitary phase (complex). To make them
globally consistent, GCSF-Embed builds a **connection Laplacian** (vector-diffusion-maps style):

- For each edge, solve a **Procrustes** problem on the overlapping nodes to get the best transport map
  `Q_ij` aligning the two patch frames. For complex (magnetic) frames this is a **unitary** map
  (`Q Q^H = I`), obtained from the SVD of the conjugate cross-correlation:

  ```
  X = V_j^H V_i,   X = U S W^H,   Q_ij = W U^H
  ```

- Assemble the Hermitian connection Laplacian with blocks `-w_ij Q_ij` (and `Q_ij^H` on the
  transpose), then take its smallest eigenvectors to obtain globally synchronized coordinates.

Using the conjugate transpose throughout guarantees `H` is exactly Hermitian and positive
semidefinite for complex frames, and reduces to the standard orthogonal construction for real frames.

### 5. Graph readout (`graphPE`)

A permutation-invariant, SE(d)-invariant graph vector, pooling the invariant node features via:

- first-order moments (mean, std, max),
- **second-order** channel-covariance pooling (captures channel interactions),
- **spectral-distribution** pooling (quantiles of the patch eigenvalues), and
- log-scaled size descriptors (log nodes, log edges, density).

---

## Complexity

For bounded-degree graphs and constant dimension `d`:

- **Per node:** `O(s^2 k)` patch eigensolve + `O(s k d + s d^2 + d^3)` geometry, where `s` is the
  patch size (`s = O(dbar^r)` for average degree `dbar` and radius `r`).
- **Synchronization:** `O(m k^3)` transport + a sparse Hermitian eigensolve on an `nk x nk` operator.

This is **functionally linear in `n`** for bounded-degree graphs. Dense eigen-decomposition fallbacks
(for tiny patches or non-convergence) are `O(s^3)` / `O((nk)^3)` in the worst case and exist only as
safety nets.

---

## Pipeline / repository breakdown

```
Input:  <graph>_nodes.csv   (col 1 = id, cols 2..d+1 = coordinates in R^d)
        <graph>_edges.csv   (src, dst; 0- or 1-based, auto-detected)

  gcsf_embed(nodeFile, edgeFile, outNodeCSV, params)
    │
    ├─ load coords (any dimension) + build directed/undirected adjacency
    ├─ build distance weights  (self-tuned or fixed bandwidth)
    │
    ├─ per-node loop ───────────────────────────────► nodePE
    │     ├─ r-hop patch
    │     ├─ magnetic OR undirected normalized Laplacian
    │     ├─ bottom-k eigenpairs  ──► canonical frame [N1/N2]
    │     ├─ heat-kernel diffusion coordinate over T
    │     ├─ [N6] SE(d)-invariant geometric features
    │     └─ [N5] flow-imbalance scalar (optional)
    │
    ├─ synchronization (optional) ──────────────────► syncPE
    │     ├─ per-edge unitary Procrustes transport [N13]
    │     └─ connection Laplacian → smallest eigenvectors
    │
    └─ graph readout [N7] ──────────────────────────► graphPE

  batch_gcsf_embed(inputFolder, outputFolder, params)
      runs gcsf_embed over every *_nodes.csv / *_edges.csv pair
```

### Outputs

- `nodePE` — per-node local encoding (spectral + geometric + optional flow scalar)
- `syncPE` — globally synchronized encoding (optional)
- `graphPE` — single graph-level vector

### Key parameters

| Parameter               | Meaning                                            | Default          |
|-------------------------|----------------------------------------------------|------------------|
| `k`                     | eigenvectors per patch                             | 8                |
| `r`                     | patch (hop) radius                                 | 2                |
| `T`                     | diffusion times                                    | `[0.1 1 10]`     |
| `self_tuned`            | self-tuning bandwidth                              | `true`           |
| `use_magnetic`          | magnetic (directed) Laplacian                      | `false`          |
| `magnetic_q`            | magnetic charge                                    | 0.25             |
| `adaptive_q`            | scale `q` by local flow imbalance `[N3]`           | `false`          |
| `geom_features`         | SE(d)-invariant geometry `[N6]`                    | `true`           |
| `geom_kframe`           | # spectral–geometric frames                        | 4                |
| `compute_sync`          | build synchronized encoding                        | `false`          |
| `sync_dim`              | # synchronized eigenvectors                        | 1                |
| `readout_cov`           | covariance pooling in readout `[N7]`               | `true`           |
| `readout_spectral`      | spectral-distribution pooling `[N7]`               | `true`           |

---

## Usage

```matlab
% single graph, directed with adaptive magnetic charge and synchronization
params = struct();
params.use_magnetic  = true;
params.adaptive_q    = true;
params.compute_sync  = true;
[nodePE, syncPE, graphPE] = gcsf_embed('g_nodes.csv', 'g_edges.csv', '', params);

% batch over a folder
batch_gcsf_embed('data/in', 'data/out', params);
```
