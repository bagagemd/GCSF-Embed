function [nodePE, syncPE, graphPE] = gcsf_embed(nodeFile, edgeFile, outNodeCSV, params)
% GCSF_EMBED  Geometric/Graph Connection Spectral Features.
%
% Dimension-general (any d) positional encoding for directed spatial graphs.
% Produces three levels of embedding:
%   nodePE  : per-node local spectral + SE(d)-invariant geometric features
%   syncPE  : globally synchronised encoding via a (unitary) connection Laplacian
%   graphPE : permutation-invariant graph-level readout
%
% ---------------------------------------------------------------------------
% SE(d) INVARIANCE DESIGN
% ---------------------------------------------------------------------------
% The output representation is INVARIANT to rigid motions (rotation +
% translation) in R^d, for ANY d, WITHOUT collapsing to a distance-only
% description. This is achieved by computing EQUIVARIANT geometric quantities
% on each patch and contracting them to invariant SCALARS:
%
%   * relative position vectors (p_j - p_i) are translation-invariant and
%     rotation-equivariant by construction;
%   * spectral-geometric frame vectors g_c = sum_j V(j,c) (p_j - p_i) in R^d
%     are rotation-equivariant and fuse spectral structure with geometry;
%   * their Gram matrix <g_c, g_c'> and norms ||g_c|| are rotation-INVARIANT
%     scalars that STILL encode lengths and angles (local geometry);
%   * the eigenvalues of the local covariance/gyration tensor C in R^{d x d}
%     are rotation-invariant and encode local anisotropy (shape).
%
% Inner products / norms / covariance spectra are defined in every dimension,
% so nothing is hardcoded to d = 3 (no cross products). 
%
% ---------------------------------------------------------------------------
% COMPLEXITY (constant d, bounded degree)
% ---------------------------------------------------------------------------
% Per node: O(s^2 k)         patch eigensolve   (s = patch size)
%         + O(s k d + s d^2 + d^3)   geometric channels   
% Sync    : O(m k^3) transport + sparse Hermitian eigensolve on (nk x nk).
% => functionally linear in n for bounded-degree graphs; s = O(dbar^r).
% Dense eig fallbacks (tiny patches / eigs non-convergence) are O(s^3) /
% O((nk)^3) worst case and are retained only as safety nets.

    if nargin < 4, params = struct(); end
    if nargin < 3, outNodeCSV = ""; end

    % ----------------------------
    % Core defaults
    % ----------------------------
    if ~isfield(params,'k'),            params.k = 8; end
    if ~isfield(params,'r'),            params.r = 2; end
    if ~isfield(params,'T'),            params.T = [0.1 1 10]; end
    if ~isfield(params,'alpha'),        params.alpha = 1.0; end
    if ~isfield(params,'self_tuned'),   params.self_tuned = true; end
    if ~isfield(params,'sigma_k'),      params.sigma_k = 8; end
    if ~isfield(params,'index_base'),   params.index_base = 'auto'; end  % 0, 1, or 'auto'

    % Diffusion defaults
    if ~isfield(params,'use_laplacian'), params.use_laplacian = true; end
    if ~isfield(params,'drop_trivial'),  params.drop_trivial  = true; end

    % Sync defaults
    if ~isfield(params,'compute_sync'),    params.compute_sync = false; end
    if ~isfield(params,'sync_dim'),        params.sync_dim = 1; end
    if ~isfield(params,'outSyncCSV'),      params.outSyncCSV = ""; end
    if ~isfield(params,'sync_confidence'), params.sync_confidence = true; end
    if ~isfield(params,'conf_tau'),        params.conf_tau = 0.5; end

    % Graph embedding defaults
    if ~isfield(params,'outGraphCSV'),          params.outGraphCSV = ""; end
    if ~isfield(params,'graph_include_counts'), params.graph_include_counts = true; end
    if ~isfield(params,'graph_include_sync'),   params.graph_include_sync = false; end
    if ~isfield(params,'graph_valid_only'),     params.graph_valid_only = true; end

    % Magnetic defaults
    if ~isfield(params,'use_magnetic'),   params.use_magnetic = false; end
    if ~isfield(params,'magnetic_q'),     params.magnetic_q = 0.25; end
    if ~isfield(params,'magnetic_feat'),  params.magnetic_feat = 'reim'; end

    % Canonical frame degeneracy tolerance
    if ~isfield(params,'deg_tol'),        params.deg_tol = 1e-4; end

    % Adaptive local q
    if ~isfield(params,'adaptive_q'),     params.adaptive_q = false; end

    % Spectral gap truncation
    if ~isfield(params,'spectral_gap_k'), params.spectral_gap_k = false; end
    if ~isfield(params,'gap_tol'),        params.gap_tol = 0.10; end

    % Local flow imbalance feature
    if ~isfield(params,'include_flow_imbalance'), params.include_flow_imbalance = false; end

    % SE(d)-invariant equivariant-contracted geometric features
    if ~isfield(params,'geom_features'), params.geom_features = true;  end
    if ~isfield(params,'geom_gram'),     params.geom_gram     = true;  end % <g_c,g_c'> invariants
    if ~isfield(params,'geom_cov'),      params.geom_cov      = true;  end % local covariance spectrum
    if ~isfield(params,'geom_kframe'),   params.geom_kframe   = 4;     end % # spectral-geometric frames used

    % Upgraded readout (second-order + spectral-distribution pooling)
    if ~isfield(params,'readout_cov'),       params.readout_cov = true; end   % channel covariance
    if ~isfield(params,'readout_spectral'),  params.readout_spectral = true; end % eigenvalue quantiles
    if ~isfield(params,'readout_nquant'),    params.readout_nquant = 8; end    % # quantiles

    if params.graph_include_sync
        params.compute_sync = true;
    end

    % ----------------------------
    % Load data (dimension-general)
    % ----------------------------
    nodeData = readmatrix(nodeFile);
    coords   = nodeData(:, 2:end);          % <-- any dimension d
    n        = size(coords, 1);
    d        = size(coords, 2);

    E = readmatrix(edgeFile);
    [src_raw, dst_raw] = resolve_edge_index(E, n, params.index_base);

    A_dir = sparse(src_raw, dst_raw, 1, n, n);
    A_dir = spones(A_dir);
    A_und = spones(A_dir + A_dir');
    nUniqueEdges = nnz(triu(A_und));
    A_for_sigma  = A_und;

    if params.self_tuned
        sigma = estimate_sigma_self_tuned(coords, A_for_sigma, params.sigma_k);
        W_dir = build_W_self_tuned_directed(coords, src_raw, dst_raw, sigma, n);
    else
        W_dir = build_W_alpha_directed(coords, src_raw, dst_raw, params.alpha, n);
    end

    if params.use_magnetic
        W_sym = 0.5 * (W_dir + W_dir');
        A_nbr = A_und;
    else
        W_sym = max(W_dir, W_dir');
        A_nbr = spones(W_sym);
    end

    nbrs = cell(n,1);
    for u = 1:n
        nbrs{u} = find(A_nbr(u,:));
    end

    % ----------------------------
    % Feature dimensionality bookkeeping
    % ----------------------------
    k = params.k;
    T = params.T(:)'; M = numel(T);

    if params.use_magnetic && strcmpi(params.magnetic_feat,'reim')
        Dspec = 2*k*M;
    else
        Dspec = k*M;
    end

    Dfi = double(params.include_flow_imbalance);

    % geometric feature width
    kf = max(0, min(params.geom_kframe, k));
    Dgram = 0; Dcov = 0;
    if params.geom_features
        if params.geom_gram
            Dgram = kf*(kf+1)/2 + kf;   % upper-tri Gram (incl diag) + norms
        end
        if params.geom_cov
            Dcov = d;                   % d covariance eigenvalues
        end
    end
    Dgeom = Dgram + Dcov;

    Dnode = Dspec + Dfi + Dgeom;
    nodePE = zeros(n, Dnode);

    % Cache for sync (stores complex V when magnetic)
    localCache = struct('nodes', cell(n,1), 'V', cell(n,1), 'mu', cell(n,1));

    % Store per-node eigenvalues for spectral-distribution readout
    muStore = nan(n, k);

    % ----------------------------
    % Node-wise local spectra + geometry
    % ----------------------------
    for i = 1:n
        sub = rhop_nodes(i, nbrs, params.r);
        if numel(sub) <= 2
            continue;
        end

        % Local flow imbalance on this patch
        if params.include_flow_imbalance
            Ap_dir_full = full(A_dir(sub, sub));
            fi_i = local_flow_imbalance(Ap_dir_full);
        end

        if params.use_magnetic
            % ---- Magnetic Laplacian on patch ---- adaptive q ----
            Wp_dir = full(W_dir(sub, sub));
            Ws     = 0.5 * (Wp_dir + Wp_dir');
            if nnz(Ws) == 0, continue; end

            Ap_dir = full(A_dir(sub, sub));

            if params.adaptive_q
                fi_patch = local_flow_imbalance(Ap_dir);
                q_local  = params.magnetic_q * fi_patch;
            else
                q_local  = params.magnetic_q;
            end

            Theta = 2*pi*q_local * (Ap_dir - Ap_dir');
            Hq    = Ws .* exp(1i * Theta);

            di   = sum(Ws, 2);
            di(di==0) = eps;
            Dm12 = diag(1 ./ sqrt(di));
            S    = Dm12 * Hq * Dm12;
            L    = eye(size(S,1)) - S;

            k_req = min(k + double(params.drop_trivial), size(L,1));
            [Vall, muall] = bottomk_eigs_hermitian_psd(L, k_req);

            if params.drop_trivial && k_req >= 2
                V  = Vall(:, 2:k_req);
                mu = muall(2:k_req);
            else
                V  = Vall; mu = muall;
            end

            % canonical complex frame + per-column U(1) phase fix
            V = canonical_frame_complex(V, mu, params.deg_tol);

            % Spectral gap truncation
            if params.spectral_gap_k
                k_eff = spectral_gap_truncate(mu, k, params.gap_tol);
                V(:, k_eff+1:end) = 0;
                mu(k_eff+1:end)   = 0;
            end

            V  = pad_cols(V, k);
            mu = pad_vec(mu, k);

        else
            % ---- Undirected diffusion on patch ----
            Wi = full(W_sym(sub, sub));
            if nnz(Wi) == 0, continue; end

            di = sum(Wi, 2);
            di(di==0) = eps;
            Dm12 = diag(1 ./ sqrt(di));
            S    = Dm12 * Wi * Dm12;

            if params.use_laplacian
                L     = eye(size(S,1)) - S;
                k_req = min(k + double(params.drop_trivial), size(L,1));
                [Vall, muall] = bottomk_eigs_psd(L, k_req);

                if params.drop_trivial && k_req >= 2
                    V  = Vall(:, 2:k_req);
                    mu = muall(2:k_req);
                else
                    V  = Vall; mu = muall;
                end

                V = canonical_frame_real(V, mu, params.deg_tol);

                if params.spectral_gap_k
                    k_eff = spectral_gap_truncate(mu, k, params.gap_tol);
                    V(:, k_eff+1:end) = 0;
                    mu(k_eff+1:end)   = 0;
                end

                V  = pad_cols(V, k);
                mu = pad_vec(mu, k);

            else
                [V0, lam0] = topk_eigs_sym(S, min(k, size(S,1)));
                mu0 = 1 - lam0(:);
                V0  = canonical_frame_real(V0, mu0, params.deg_tol);
                V   = pad_cols(V0, k);
                mu  = pad_vec(mu0, k);
            end
        end

        center = find(sub == i, 1);
        if isempty(center), continue; end
        v_center = V(center, :);

        % ---- Multi-time diffusion coordinate (spectral block) ----
        phi_blocks = zeros(1, k*M);
        for m = 1:M
            scale = exp(-T(m) * mu(:)).';
            phi_blocks(1, (m-1)*k + (1:k)) = v_center .* scale;
        end

        if params.use_magnetic
            if strcmpi(params.magnetic_feat, 'reim')
                phi_spec = [real(phi_blocks), imag(phi_blocks)];
            else
                phi_spec = abs(phi_blocks);
            end
        else
            phi_spec = real(phi_blocks);
        end

        % ---- SE(d)-invariant geometric features ----
        phi_geom = [];
        if params.geom_features
            % relative positions of patch nodes to the centre (translation-inv,
            % rotation-equivariant). coords are ANY dimension d.
            P = coords(sub, :) - coords(i, :);          % (s x d)

            % local weights for the covariance/gyration tensor
            wloc = di(:);                               % degree-like weights on patch
            wloc = wloc / max(sum(wloc), eps);

            phi_geom = geom_invariants(P, V, mu, wloc, kf, ...
                                       params.geom_gram, params.geom_cov, d);
        end

        % ---- assemble node feature ----
        phi_i = phi_spec;
        if params.include_flow_imbalance
            phi_i = [phi_i, fi_i]; %#ok<AGROW>
        end
        if ~isempty(phi_geom)
            phi_i = [phi_i, phi_geom]; %#ok<AGROW>
        end

        nodePE(i, :) = phi_i;

        localCache(i).nodes = sub;
        localCache(i).V     = V;        % complex if magnetic
        localCache(i).mu    = mu(:);
        muStore(i, :)       = real(mu(:).');
    end

    if strlength(outNodeCSV) > 0
        writematrix(real(nodePE), outNodeCSV);
    end

    % ----------------------------
    % Optional global synchronisation (UNITARY when complex)
    % ----------------------------
    syncPE = [];
    if params.compute_sync
        [srcU, dstU] = find(triu(A_und, 1));
        m_u   = numel(srcU);
        Qcell = cell(m_u, 1);
        okEdge = false(m_u, 1);
        wMult  = ones(m_u, 1);

        for e = 1:m_u
            ii = srcU(e); jj = dstU(e);
            [Q, ok, res] = transport_Qij(ii, jj, localCache);   % unitary if complex
            if ok
                Qcell{e} = Q;
                okEdge(e) = true;
                if params.sync_confidence
                    tau = max(params.conf_tau, eps);
                    wMult(e) = exp(-(res / tau)^2);
                end
            end
        end

        sync_dim = max(1, params.sync_dim);
        [H, used] = build_connection_laplacian(srcU, dstU, W_sym, Qcell, okEdge, wMult, k);

        if used < 1
            warning('compute_sync: no usable edges. Returning zeros.');
            syncPEc = zeros(n, k*sync_dim);
        else
            Vec = smallest_eigs_hermitian(H, sync_dim);
            syncPEc = zeros(n, k*sync_dim);
            for dd = 1:sync_dim
                vd = Vec(:, dd);
                Bd = reshape(vd, [k, n]).';
                syncPEc(:, (dd-1)*k + (1:k)) = Bd;
            end
        end

        if params.use_magnetic && strcmpi(params.magnetic_feat, 'reim')
            syncPE = [real(syncPEc), imag(syncPEc)];
        else
            syncPE = real(syncPEc);
        end

        if strlength(params.outSyncCSV) > 0
            writematrix(real(syncPE), params.outSyncCSV);
        end
    end

    % ----------------------------
    % Graph-level embedding (upgraded readout) 
    % ----------------------------
    graphPE = graph_readout(nodePE, syncPE, muStore, params, n, nUniqueEdges);
    if strlength(params.outGraphCSV) > 0
        writematrix(real(graphPE), params.outGraphCSV);
    end
end


% ======================================================================
% SE(d)-invariant geometric features from equivariant contractions
%
% Inputs:
%   P    : (s x d) relative positions of patch nodes to centre (equivariant)
%   V    : (s x k) patch eigenvectors (real or complex)
%   mu   : (k x 1) eigenvalues
%   wloc : (s x 1) nonneg weights summing to 1 (for covariance tensor)
%   kf   : number of spectral-geometric frame vectors to use (<= k)
%   useGram, useCov : toggles
%   d    : ambient dimension
%
% Construction:
%   Frame vectors  g_c = sum_j Re(V(j,c)) * P(j,:)   in R^d   (equivariant)
%   Invariants:
%     Gram  G_{cc'} = <g_c, g_c'>   (upper triangle incl diagonal)
%     Norms ||g_c||
%     Covariance C = sum_j wloc_j P(j,:)' P(j,:)  (d x d), eig(C) invariant
%
% Rotating P -> P R rotates every g_c -> R' g_c, leaving all inner products
% and every eigenvalue of C unchanged. Translation was removed by centring.
% Valid for any d; uses only inner products (no cross products).
% ======================================================================
function f = geom_invariants(P, V, mu, wloc, kf, useGram, useCov, d)
    f = [];

    % Use real part of eigenvectors as scalar weights on equivariant vectors.
    % (For complex/magnetic frames this takes the real gauge-fixed component;
    %  the imaginary part is already represented in the spectral block.)
    Vr = real(V);

    if useGram && kf >= 1
        % Frame vectors G = Vr(:,1:kf)' * P   -> (kf x d), each row is g_c'
        Gf = Vr(:, 1:kf).' * P;            % (kf x d), equivariant rows
        Gram = Gf * Gf.';                  % (kf x kf) invariant
        % upper triangle including diagonal
        ut = triu(true(kf));
        gram_vec = Gram(ut).';             % 1 x kf(kf+1)/2
        norms = sqrt(max(diag(Gram), 0)).';% 1 x kf
        f = [f, gram_vec, norms];
    end

    if useCov
        % weighted local covariance / gyration tensor (d x d)
        Pw = P .* wloc;                    % (s x d)
        C  = Pw.' * P;                     % (d x d), invariant spectrum
        C  = 0.5 * (C + C.');              % symmetrise for numerical safety
        ev = sort(real(eig(C)), 'descend');% d eigenvalues
        ev = ev(:).';
        if numel(ev) < d, ev = [ev, zeros(1, d-numel(ev))]; end
        f = [f, ev(1:d)];
    end
end


% ======================================================================
% Canonical frame for REAL degenerate eigenspaces
% ======================================================================
function V = canonical_frame_real(V, mu, tol)
    if isempty(V), return; end
    mu   = real(mu(:));
    k    = size(V, 2);
    used = false(1, k);

    for j = 1:k
        if used(j), continue; end
        block = find(abs(mu - mu(j)) < tol);
        block = block(~used(block));
        if numel(block) == 1
            [~, ii] = max(abs(V(:, block)));
            if V(ii, block) < 0, V(:, block) = -V(:, block); end
        else
            [Q, ~, ~] = qr(V(:, block), 'econ');
            for b = 1:size(Q, 2)
                [~, ii] = max(abs(Q(:, b)));
                if Q(ii, b) < 0, Q(:, b) = -Q(:, b); end
            end
            V(:, block) = Q;
        end
        used(block) = true;
    end
end


% ======================================================================
% Canonical frame for COMPLEX (Hermitian) degenerate eigenspaces
% + per-column U(1) global-phase canonicalisation
% ======================================================================
function V = canonical_frame_complex(V, mu, tol)
    if isempty(V), return; end
    mu   = real(mu(:));
    k    = size(V, 2);
    used = false(1, k);

    for j = 1:k
        if used(j), continue; end
        block = find(abs(mu - mu(j)) < tol);
        block = block(~used(block));
        if numel(block) > 1
            [Q, ~] = qr(V(:, block), 'econ');
            V(:, block) = Q;
        end
        used(block) = true;
    end

    for c = 1:k
        [~, idx] = max(abs(V(:, c)));
        ph = angle(V(idx, c));
        V(:, c) = V(:, c) * exp(-1i * ph);
    end
end


% ======================================================================
% Local directed flow imbalance scalar
% ======================================================================
function fi = local_flow_imbalance(Ap_dir)
    skew = Ap_dir - Ap_dir';
    sym  = Ap_dir + Ap_dir';
    denom = norm(sym, 'fro');
    if denom < eps
        fi = 0.0;
    else
        fi = norm(skew, 'fro') / denom;
    end
end


% ======================================================================
% Spectral-gap-based effective k selection
% ======================================================================
function k_eff = spectral_gap_truncate(mu, k_max, gap_tol)
    mu = real(mu(:));
    k_max = min(k_max, numel(mu));
    if k_max <= 1
        k_eff = k_max;
        return;
    end
    mu_mean = mean(mu(1:k_max));
    if mu_mean < eps
        k_eff = k_max;
        return;
    end
    gaps = diff(mu(1:k_max));
    threshold = gap_tol * mu_mean;
    idx = find(gaps > threshold, 1, 'first');
    if isempty(idx)
        k_eff = k_max;
    else
        k_eff = idx;
    end
end


% ======================================================================
% Graph readout: moments + second-order (covariance) pooling
%      + spectral-distribution pooling + explicit size descriptors
% ======================================================================
function graphPE = graph_readout(nodePE, syncPE, muStore, params, nNodes, nUniqueEdges)
    Xn = nodePE;
    if params.graph_valid_only
        valid = any(abs(Xn) > 0, 2);
        Xn = Xn(valid, :);
    end
    if isempty(Xn), Xn = zeros(1, size(nodePE,2)); end

    % --- first-order moments ---
    g_moments = [mean(Xn,1), std(Xn,0,1), max(Xn,[],1)];

    % --- second-order (channel covariance) pooling ---
    g_cov = [];
    if params.readout_cov
        Xc = Xn - mean(Xn,1);
        C  = (Xc.' * Xc) / max(size(Xn,1)-1, 1);   % (D x D) channel covariance
        ut = triu(true(size(C,1)));
        g_cov = C(ut).';                            % upper triangle (incl diag)
    end

    % --- spectral-distribution pooling (patch eigenvalue quantiles) ---
    g_spec = [];
    if params.readout_spectral && ~isempty(muStore)
        mu_all = muStore(:);
        mu_all = mu_all(~isnan(mu_all));
        nq = max(2, params.readout_nquant);
        if isempty(mu_all)
            g_spec = zeros(1, nq+2);
        else
            qs = linspace(0, 1, nq);
            g_spec = quantile(mu_all, qs);
            g_spec = [g_spec, mean(mu_all), std(mu_all)];
        end
    end

    % --- sync readout (unchanged form, still invariant) ---
    if params.graph_include_sync && ~isempty(syncPE)
        Xs = syncPE;
        if params.graph_valid_only
            validS = any(abs(Xs) > 0, 2);
            Xs = Xs(validS, :);
        end
        if isempty(Xs)
            g_sync = zeros(1, 3*size(syncPE,2));
        else
            g_sync = [mean(Xs,1), std(Xs,0,1), max(Xs,[],1)];
        end
    else
        g_sync = [];
    end

    % --- explicit size descriptors (log-scaled, not raw counts) ---
    if params.graph_include_counts
        dens = nUniqueEdges / max(nNodes*(nNodes-1)/2, 1);
        g_counts = [log1p(nNodes), log1p(nUniqueEdges), dens];
    else
        g_counts = [];
    end

    graphPE = [g_counts, g_moments, g_cov, g_spec, g_sync];
end


% ======================================================================
% Sigma estimation and weight builders (dimension-general)
% ======================================================================
function sigma = estimate_sigma_self_tuned(coords, A_und, k_sigma)
    n = size(coords,1);
    sigma = zeros(n,1);
    for i = 1:n
        nb = find(A_und(i,:));
        if isempty(nb), sigma(i) = 1.0; continue; end
        d = sqrt(sum((coords(nb,:) - coords(i,:)).^2, 2));
        d = sort(d);
        idx = min(k_sigma, numel(d));
        sigma(i) = max(d(max(idx,1)), eps);
    end
    sigma = max(sigma, eps);
end

function W = build_W_self_tuned_directed(coords, src, dst, sigma, n)
    m = numel(src); vals = zeros(m,1);
    for e = 1:m
        i = src(e); j = dst(e);
        rij2 = sum((coords(i,:) - coords(j,:)).^2);
        denom = max(sigma(i)*sigma(j), eps);
        vals(e) = exp(-rij2 / denom);
    end
    W = sparse(src, dst, vals, n, n);
end

function W = build_W_alpha_directed(coords, src, dst, alpha, n)
    m = numel(src); vals = zeros(m,1);
    a2 = max(alpha*alpha, eps);
    for e = 1:m
        i = src(e); j = dst(e);
        rij2 = sum((coords(i,:) - coords(j,:)).^2);
        vals(e) = exp(-rij2 / a2);
    end
    W = sparse(src, dst, vals, n, n);
end


% ======================================================================
% Edge-index validation
% ======================================================================
function [src, dst] = resolve_edge_index(E, n, base)
% Convert an edge list to 1-based MATLAB indices, validating every index.
%   base = 0, 1, or 'auto'. 'auto' accepts the file only when its indexing
%   is unambiguous (index 0 present -> 0-based; index n present -> 1-based)
%   and errors otherwise. Indices are never guessed silently or clipped.
    if isempty(E)
        src = zeros(0,1); dst = zeros(0,1);
        return;
    end
    if size(E,2) < 2
        error('gcsf_embed:edges', 'Edge file must have two columns (src, dst).');
    end
    E = E(:, 1:2);
    if any(E(:) ~= round(E(:)))
        error('gcsf_embed:edges', 'Edge indices must be integers.');
    end

    if ischar(base) || isstring(base)
        if ~strcmpi(base, 'auto')
            error('gcsf_embed:edges', 'params.index_base must be 0, 1, or ''auto''.');
        end
        lo = min(E(:)); hi = max(E(:));
        if lo == 0 && hi == n
            error('gcsf_embed:edges', ...
                'Edge indices contain both 0 and n = %d, so they are neither 0- nor 1-based.', n);
        elseif lo == 0
            base = 0;
        elseif hi == n
            base = 1;
        else
            error('gcsf_embed:edges', ...
                ['Cannot tell whether edge indices are 0- or 1-based (neither 0 nor ' ...
                 'n = %d appears). Set params.index_base to 0 or 1.'], n);
        end
    elseif ~(isequal(base, 0) || isequal(base, 1))
        error('gcsf_embed:edges', 'params.index_base must be 0, 1, or ''auto''.');
    end

    src = E(:,1) + (1 - base);
    dst = E(:,2) + (1 - base);
    bad = src < 1 | src > n | dst < 1 | dst > n;
    if any(bad)
        error('gcsf_embed:edges', ...
            '%d edge(s) reference nodes outside 1..%d after applying index_base = %d.', ...
            nnz(bad), n, base);
    end
end


% ======================================================================
% Neighbourhoods
% ======================================================================
function sub = rhop_nodes(i, nbrs, r)
    n = numel(nbrs);
    visited = false(1, n); visited(i) = true;
    frontier = i;
    for step = 1:r
        if isempty(frontier), break; end
        mask = false(1, n);
        for u = frontier
            nu = nbrs{u};
            if ~isempty(nu), mask(nu) = true; end
        end
        mask = mask & ~visited;
        visited = visited | mask;
        frontier = find(mask);
    end
    sub = find(visited);
end


% ======================================================================
% Eigensolvers
% ======================================================================
function [V, mu] = bottomk_eigs_psd(L, k)
    n = size(L,1); k_eff = min(k, n);
    opts.issym = true; opts.isreal = true;
    try
        if k_eff < n
            [V0, D0] = eigs(L, k_eff, 'smallestreal', opts);
            mu0 = diag(D0);
        else
            [V0, D0] = eig(full(L));
            mu0 = diag(D0);
        end
    catch
        [V0, D0] = eig(full(L));
        mu0 = diag(D0);
    end
    mu0 = real(mu0); V0 = real(V0);
    [mu0, idx] = sort(mu0, 'ascend');
    V0 = V0(:, idx);
    V = V0; mu = mu0(:);
end

function [V, mu] = bottomk_eigs_hermitian_psd(L, k)
    n = size(L,1); k_eff = min(k, n);
    opts.issym = true; opts.isreal = false;
    try
        if k_eff < n
            [V0, D0] = eigs(L, k_eff, 'smallestreal', opts);
            mu0 = diag(D0);
        else
            [V0, D0] = eig(full(L));
            mu0 = diag(D0);
        end
    catch
        [V0, D0] = eig(full(L));
        mu0 = diag(D0);
    end
    mu0 = real(mu0);
    [mu0, idx] = sort(mu0, 'ascend');
    V0 = V0(:, idx);
    V = V0; mu = mu0(:);
end

function Vec = smallest_eigs_hermitian(H, s)
    Nk = size(H,1); s = min(s, Nk);
    opts.issym = true; opts.isreal = false;
    try
        [Vec, ~] = eigs(H, s, 'smallestreal', opts);
    catch
        [Vfull, Dfull] = eig(full(H));
        [~, idx] = sort(real(diag(Dfull)), 'ascend');
        Vec = Vfull(:, idx(1:s));
    end
end

function [V, lam] = topk_eigs_sym(S, k)
    n = size(S,1); k_eff = min(k, n);
    opts.issym = true; opts.isreal = true;
    try
        if k_eff < n
            [V0, D0] = eigs(S, k_eff, 'largestreal', opts);
            lam0 = diag(D0);
        else
            [V0, D0] = eig(S);
            lam0 = diag(D0);
        end
    catch
        [V0, D0] = eig(full(S));
        lam0 = diag(D0);
    end
    lam0 = real(lam0); V0 = real(V0);
    [lam0, idx] = sort(lam0, 'descend');
    V0 = V0(:, idx);
    V = V0; lam = lam0(:);
end

function Vp = pad_cols(V, k)
    [n, k0] = size(V);
    Vp = zeros(n, k);
    if ~isreal(V), Vp = complex(Vp); end
    Vp(:, 1:min(k0,k)) = V(:, 1:min(k0,k));
end

function vp = pad_vec(v, k)
    v = v(:); vp = zeros(k,1);
    vp(1:min(numel(v),k)) = v(1:min(numel(v),k));
end


% ======================================================================
% Sync internals  ---  UNITARY Procrustes for complex frames
% ======================================================================
function [Q, ok, res] = transport_Qij(i, j, cache)
% Best-aligning map Q between overlapping patch frames.
%   Real case:    Q in O(k)  (orthogonal Procrustes)
%   Complex case: Q in U(k)  (UNITARY Procrustes)
% Both come from the SVD of the cross-correlation using the correct
% (conjugate) transpose; for real inputs this reduces to the old behaviour.
    ok = false; Q = []; res = inf;
    ci = cache(i); cj = cache(j);
    if isempty(ci.nodes) || isempty(cj.nodes), return; end
    if isempty(ci.V)     || isempty(cj.V),     return; end
    [~, ia, ib] = intersect(ci.nodes, cj.nodes);
    k = size(ci.V, 2);
    if numel(ia) < k, return; end
    Vi = ci.V(ia, :); Vj = cj.V(ib, :);

    X = Vj' * Vi;             % ' is conjugate transpose -> Hermitian-correct
    [U, ~, Wsv] = svd(X, 'econ');
    Q = Wsv * U';             % unitary (or orthogonal if real): Q Q^H = I

    R = Vj - Vi * Q;
    res = norm(R, 'fro') / sqrt(max(size(R,1), 1));
    ok = true;
end

function [H, usedEdges] = build_connection_laplacian(src, dst, Wsym, Qcell, okEdge, wMult, k)
% Hermitian connection Laplacian. Off-diagonal blocks use Q and its
% conjugate transpose Q' so that H is Hermitian for complex (unitary) Q
% and symmetric for real (orthogonal) Q.
    n = size(Wsym,1); Nk = n * k;
    I = []; J = []; V = []; usedEdges = 0;
    if nargin < 6 || isempty(wMult), wMult = ones(numel(src),1); end

    isComplexQ = false;
    for e = 1:numel(src)
        if okEdge(e) && ~isempty(Qcell{e}) && ~isreal(Qcell{e})
            isComplexQ = true; break;
        end
    end

    for e = 1:numel(src)
        if ~okEdge(e), continue; end
        i = src(e); j = dst(e);
        w = full(Wsym(i,j));
        if w <= 0, continue; end
        Q = Qcell{e};
        if isempty(Q), continue; end
        c = max(wMult(e), 0); w = w * c;
        if w <= 0, continue; end
        usedEdges = usedEdges + 1;

        ii = (i-1)*k + (1:k); jj = (j-1)*k + (1:k);

        % diagonal (degree) blocks: w * I
        I = [I; ii(:); jj(:)];
        J = [J; ii(:); jj(:)];
        V = [V; w*ones(k,1); w*ones(k,1)];

        [rr, cc] = ndgrid(1:k, 1:k);

        % off-diagonal block (i,j): -w * Q
        Hij = -w * Q;
        I = [I; ii(rr(:))]; J = [J; jj(cc(:))]; V = [V; Hij(:)];

        % off-diagonal block (j,i): -w * Q'  (conjugate transpose)
        Hji = -w * Q';
        I = [I; jj(rr(:))]; J = [J; ii(cc(:))]; V = [V; Hji(:)];
    end

    if isComplexQ
        H = sparse(I, J, V, Nk, Nk);
        H = (H + H') / 2;          % Hermitian symmetrisation (conjugate)
    else
        H = sparse(I, J, real(V), Nk, Nk);
        H = (H + H.') / 2;         % real symmetrisation
    end
end
