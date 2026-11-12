# GCSF-Embed

**Local spectral–geometric positional encodings for spatial graphs, with optional support for directed edges.**

GCSF (**G**eometric **C**onnection **S**pectral **F**eatures) computes training-free encodings for graphs whose
nodes carry coordinates in $\mathbb{R}^d$ (any dimension) and whose edges may be directed. Around every node it
builds a local patch, computes multiscale diffusion features from the patch Laplacian, and couples them with the
patch geometry through quantities that do not change under translations, rotations, or reflections of the
coordinates. A magnetic Laplacian handles directed edges, and an optional connection-Laplacian stage
synchronizes the local spectral frames across the graph.

This repository accompanies the paper *Local Spectral–Geometric Representations of Spatial Graphs*
(IEEE BigData 2026). It contains a MATLAB reference implementation and a Python port.

---

## Installation

**Python** (`Main/gcsf_embed.py`): requires `numpy` and `scipy`. `torch` and `torch_geometric` are needed only
for the optional `to_pyg_data` export.

```bash
pip install numpy scipy            # core
pip install torch torch_geometric  # optional, for PyG export
```

**MATLAB** (`Main/gcsf_embed.m`, `Main/batch_gcsf_embed.m`): R2019a or later (uses `readmatrix`/`writematrix`).
Add `Main/` to the MATLAB path.

---

## Usage

### Python

```python
import sys; sys.path.append("Main")
import numpy as np
from gcsf_embed import gcsf_embed, GCSFParams

# coords: (n, d) node positions; edges: (E, 2) directed edge list, 0-based.
# For an undirected graph, list both directions of every edge.
res = gcsf_embed(coords, edges, GCSFParams(k=8, r=2), index_base=0)

res.node_pe   # (n, D_node) per-node encoding
res.graph_pe  # (D_graph,)  graph-level descriptor
res.sync_pe   # None unless compute_sync=True
```

From a PyTorch Geometric `Data` object, pass the transposed `edge_index`:

```python
res = gcsf_embed(coords, data.edge_index.numpy().T, params, index_base=0)
```

Batch processing over a folder of CSV files:

```python
from gcsf_embed import batch_gcsf_embed
results = batch_gcsf_embed("data/in", "data/out", params, index_base="auto")
```

### MATLAB

```matlab
% single graph, directed, with adaptive magnetic charge and synchronization
params = struct();
params.use_magnetic = true;
params.adaptive_q   = true;
params.compute_sync = true;
[nodePE, syncPE, graphPE] = gcsf_embed('g_nodes.csv', 'g_edges.csv', '', params);

% batch over a folder
batch_gcsf_embed('data/in', 'data/out', params);
```

### Input format and edge indexing

```
<graph>_nodes.csv   column 1 = node id, columns 2..d+1 = coordinates in R^d
<graph>_edges.csv   two columns: src, dst
```

Edge indices are validated, never guessed silently or clipped. Set the base explicitly:

| `index_base` | Meaning |
|---|---|
| `0` | 0-based indices (default in Python; the NumPy/PyG convention) |
| `1` | 1-based indices |
| `"auto"` | Accepted only when unambiguous: an index of `0` means 0-based, an index of `n` means 1-based. Otherwise an error asks for an explicit base. Default in MATLAB and in the batch runners. |

An explicit base matters when node 0 has no edges, for example a molecule whose SMILES starts with a separate
ion such as `[Na+].[O-]C(C)=O`. Any index outside the valid range raises an error.

---

## Method

### 1. Spatial edge weights

Each edge $i \to j$ receives a weight from the distance between its endpoints,

$$
W_{ij} = \exp\!\left(-\frac{\lVert \mathbf{x}_i - \mathbf{x}_j \rVert_2^2}{\sigma_i \sigma_j}\right),
$$

where $\sigma_i$ is the distance from node $i$ to its $k_\sigma$-th nearest graph neighbor, or its farthest neighbor if it has fewer (self-tuning). A fixed
bandwidth, $\sigma_i\sigma_j = \alpha^2$, is also available. Because the weights depend only on distances, they
are unaffected by translations, rotations, and reflections.

### 2. Local spectra and multiscale diffusion features (`nodePE`)

For each node $i$, the $r$-hop patch $\mathcal{N}_r(i)$ is extracted and a normalized Laplacian is built on it.

**Undirected mode.** With symmetric weights $W^{(u)}_{ij} = \max(W_{ij}, W_{ji})$ restricted to the patch,

$$
L_i = I - D_i^{-1/2} W_i D_i^{-1/2}.
$$

**Magnetic mode.** Direction is encoded as complex phase. With $F_i = A_i - A_i^\top$ and symmetric amplitude
$W^{(s)} = \tfrac{1}{2}(W + W^\top)$,

$$
H_i = W_i^{(s)} \odot e^{\,\mathrm{i}\,2\pi q_i F_i}, \qquad
L_i^{(q)} = I - D_i^{-1/2} H_i D_i^{-1/2}.
$$

The charge $q_i$ is either fixed or scaled by the patch's flow imbalance,
$\eta_i = \lVert A_i - A_i^\top \rVert_F / \lVert A_i + A_i^\top \rVert_F$. On an undirected graph
$F_i = 0$, every phase vanishes, and the magnetic Laplacian reduces to the undirected one.

The $k$ lowest eigenpairs $(\mathbf{v}_{i,\ell}, \mu_{i,\ell})$ are kept (optionally dropping the trivial mode),
and the center node's entries are evaluated at diffusion times $\mathcal{T} = \{t_1, \ldots, t_M\}$:

$$
\phi_{i,\ell}(t_m) = v_{i,\ell}(c_i)\, \exp(-t_m \mu_{i,\ell}),
$$

where $c_i$ is the position of node $i$ in its patch. In magnetic mode the complex values are split into real
and imaginary parts (`magnetic_feat = 'reim'`) or replaced by their magnitudes (`'abs'`).

### 3. Invariant spectral–geometric features

With patch coordinates centered at node $i$, $P_i = X_{\mathcal{N}_r(i)} - \mathbf{1}\mathbf{x}_i^\top$, the
first $k_f$ spectral components are projected onto the geometry:

$$
B_i = \bigl(V_i^{(k_f)}\bigr)^\top P_i \in \mathbb{R}^{k_f \times d}.
$$

Row $\ell$ records how the patch geometry projects onto spectral mode $\ell$. Under any orthogonal
transformation $R \in O(d)$, $B_i \mapsto B_i R$, so the Gram matrix $G_i = B_i B_i^\top$ is unchanged. The
features keep the upper triangle of $G_i$ and the row norms $\sqrt{(G_i)_{\ell\ell}}$, plus the ordered
eigenvalues of the weighted second-moment tensor

$$
C_i = P_i^\top \mathop{\text{diag}}(\mathbf{w}_i)\, P_i,
$$

with $\mathbf{w}_i$ the normalized weighted degrees of the patch. Only inner products and eigenvalues are used, so
the construction works in any dimension $d$ and is invariant under the full Euclidean group $E(d)$, reflections
included.

### 4. Global synchronization (`syncPE`, optional)

For adjacent nodes whose patches share at least $k$ nodes, the orthogonal (real) or unitary (magnetic) Procrustes
transport between their frames is

$$
Q_{ij} = W U^{*}, \qquad \text{where } \bigl(V_j^{\cap}\bigr)^{*} V_i^{\cap} = U \Sigma W^{*}.
$$

The transports, optionally down-weighted by their alignment residual, are assembled into a Hermitian connection
Laplacian with blocks $-\tilde{w}_{ij} Q_{ij}$ and $-\tilde{w}_{ij} Q_{ij}^{*}$, in the style of vector diffusion
maps. Its lowest eigenvectors give the synchronized encoding.

### 5. Graph readout (`graphPE`)

Node features are pooled with statistics that do not depend on node order:

- coordinatewise mean, standard deviation, and maximum;
- the upper triangle of the feature covariance (second-order pooling);
- quantiles, mean, and standard deviation of all retained patch eigenvalues;
- size descriptors: $\log(1+n)$, $\log(1+|E|)$, and edge density;
- the same moments of `syncPE`, when included.

---

## Invariances and known limitations

- **Coordinates:** node features are unchanged by translations and by orthogonal transformations (rotations and
  reflections). A consequence is that mirror-image configurations, such as enantiomers, receive identical
  features.
- **Node order:** relabeling nodes only permutes the rows of `nodePE`; `graphPE` is unchanged.
- **Eigenvector sign and phase:** fixed by a deterministic convention (largest-magnitude entry made positive or
  real).
- **Repeated eigenvalues:** the convention does **not** make the basis unique within a repeated eigenvalue. Patches
  with exactly or nearly repeated eigenvalues, common in symmetric substructures, can therefore give features
  that depend on the eigensolver's arbitrary choice of basis. A fully basis-invariant variant would build features
  from eigenspace projectors instead of individual eigenvectors.
- **Directed edges:** the magnetic construction only differs from the undirected one when the adjacency matrix is
  asymmetric. An edge list that contains both directions of every edge is treated as undirected.

---

## Complexity

For fixed $r$ and $k$, every patch is processed independently. With largest patch size $m$, the local
eigendecompositions cost at most $O(n m^3)$ with a dense solver, and the geometric contractions add a lower-order
cost. The node-level construction is therefore linear in $n$ when patch size is bounded (bounded degree and small
$r$), although $m$ can grow quickly with $r$ on dense graphs. Synchronization adds one Procrustes problem per
eligible edge, $O(m k^2 + k^3)$ each, and one sparse eigenproblem of dimension $nk$.

---

## Outputs

| Output | Description |
|---|---|
| `nodePE` / `node_pe` | per-node encoding: spectral block, optional flow-imbalance scalar, geometric block |
| `syncPE` / `sync_pe` | synchronized encoding (only when `compute_sync` is enabled) |
| `graphPE` / `graph_pe` | single graph-level vector |

## Parameters

| Parameter | Meaning | Default |
|---|---|---|
| `k` | eigenvectors per patch | 8 |
| `r` | patch radius (hops) | 2 |
| `T` | diffusion times | `[0.1 1 10]` |
| `self_tuned` | self-tuning bandwidth | `true` |
| `sigma_k` | neighbor rank used for $\sigma_i$ | 8 |
| `alpha` | fixed bandwidth (when `self_tuned` is false) | 1.0 |
| `drop_trivial` | drop the lowest eigenpair | `true` |
| `deg_tol` | tolerance for treating eigenvalues as repeated | `1e-4` |
| `use_magnetic` | magnetic (directed) Laplacian | `false` |
| `magnetic_q` | magnetic charge $q$ | 0.25 |
| `adaptive_q` | scale $q$ by local flow imbalance | `false` |
| `magnetic_feat` | `'reim'` or `'abs'` | `'reim'` |
| `include_flow_imbalance` | append $\eta_i$ to node features | `false` |
| `spectral_gap_k` | suppress modes above the first large spectral gap | `false` |
| `geom_features` | invariant geometric block | `true` |
| `geom_kframe` | number of spectral components $k_f$ used in the geometric block | 4 |
| `compute_sync` | compute the synchronized encoding | `false` |
| `sync_dim` | number of synchronized eigenvectors | 1 |
| `readout_cov` | covariance pooling in the readout | `true` |
| `readout_spectral` | eigenvalue-quantile pooling in the readout | `true` |
| `index_base` | edge index base: `0`, `1`, or `"auto"` | Python: `0`; MATLAB and batch: `"auto"` |

---

## Citation

```bibtex
@inproceedings{bagagem2026gcsf,
  title     = {Local Spectral--Geometric Representations of Spatial Graphs},
  author    = {Bagagem, Derek and Serra, Edoardo},
  booktitle = {Proceedings of the IEEE International Conference on Big Data (BigData)},
  year      = {2026}
}
```
