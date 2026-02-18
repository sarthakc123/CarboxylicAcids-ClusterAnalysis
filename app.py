from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import hashlib

import numpy as np
import pandas as pd
import streamlit as st
from plotly import graph_objects as go
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.Draw import rdMolDraw2D
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# --- optional: plotly events fallback (older Streamlit) ---
try:
    from streamlit_plotly_events import plotly_events  # type: ignore
except Exception:
    plotly_events = None

# --- optional: UMAP ---
try:
    import umap.umap_ as umap  # type: ignore
except Exception:
    umap = None

DATA_PATH = Path(__file__).parent / "All Carboxylic Acids.xlsx"


# -----------------------------
# Fast cache invalidation (mtime/size) + optional strict SHA
# -----------------------------
@st.cache_data(show_spinner=False)
def sha1_for_file(path_str: str, mtime_ns: int, size: int) -> str:
    p = Path(path_str)
    h = hashlib.sha1()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fingerprint(path: Path, strict_hash: bool) -> Tuple[int, int, str]:
    stat = path.stat()
    mtime_ns, size = int(stat.st_mtime_ns), int(stat.st_size)
    if not strict_hash:
        return (mtime_ns, size, "")
    return (mtime_ns, size, sha1_for_file(str(path), mtime_ns, size))


def stable_key(*parts) -> str:
    s = repr(parts).encode("utf-8")
    return hashlib.sha1(s).hexdigest()


# -----------------------------
# Utilities
# -----------------------------
def _norm_str(x) -> str:
    try:
        return str(x).strip().lower()
    except Exception:
        return ""


def _row_contains_any(vals_lower: pd.Series, needles: List[str]) -> bool:
    for n in needles:
        if vals_lower.str.contains(n, regex=False, na=False).any():
            return True
    return False


def _score_header_row(row: pd.Series) -> Tuple[int, int]:
    vals = row.astype(str).str.lower()
    groups = [
        ["canonical smiles", "smiles"],
        ["ca index name", "compound name", "molecule name", "name"],
        ["cas", "cas number", "cas no"],
    ]
    group_hits = 0
    total_hits = 0
    for g in groups:
        hit = _row_contains_any(vals, g)
        group_hits += int(hit)
        for n in g:
            total_hits += int(vals.str.contains(n, regex=False, na=False).any())
    return (group_hits, total_hits)


def _find_header_row_scored(raw: pd.DataFrame) -> int:
    scores = raw.apply(_score_header_row, axis=1)
    best_idx = None
    best_score = (-1, -1)
    for i, s in enumerate(scores.values):
        s = tuple(s)
        if s > best_score:
            best_score = s  # type: ignore
            best_idx = i
    if best_idx is None or best_score[0] < 1:
        raise RuntimeError("Could not find a plausible header row (no SMILES-like header detected).")
    return int(best_idx)


def _find_col(df: pd.DataFrame, needles: List[str]) -> Optional[str]:
    needles_l = [n.lower() for n in needles]
    for c in df.columns:
        cl = _norm_str(c)
        if any(n in cl for n in needles_l):
            return c
    return None


def smiles_to_mol(smiles: str):
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    try:
        return Chem.MolFromSmiles(smiles.strip())
    except Exception:
        return None


def canonicalize_smiles(smiles: str) -> Optional[str]:
    mol = smiles_to_mol(smiles)
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def mol_to_png_bytes(mol) -> bytes:
    if mol is None:
        return b""
    drawer = rdMolDraw2D.MolDraw2DCairo(320, 320)
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def parse_customdata(cd) -> Optional[Tuple[str, int]]:
    if cd is None:
        return None
    if isinstance(cd, (list, tuple)) and len(cd) > 0:
        cd = cd[0]
    if not isinstance(cd, str) or ":" not in cd:
        return None
    kind, val = cd.split(":", 1)
    kind = kind.strip()
    try:
        return kind, int(val)
    except Exception:
        return None


# -----------------------------
# Data loading & descriptors
# -----------------------------
@st.cache_data(show_spinner="Loading Excel data…")
def load_raw_dataframe(path: Path, fp: Tuple[int, int, str]) -> pd.DataFrame:
    raw = pd.read_excel(path, header=None)
    header_row_idx = _find_header_row_scored(raw)
    df = pd.read_excel(path, header=header_row_idx)

    name_col = _find_col(df, ["ca index name", "compound name", "molecule name", "name"])
    smiles_col = _find_col(df, ["canonical smiles", "smiles"])
    if smiles_col is None:
        raise RuntimeError("Could not locate a SMILES column (looked for 'Canonical SMILES'/'SMILES').")

    if name_col is None:
        other_cols = [c for c in df.columns if c != smiles_col]
        name_col = other_cols[0] if other_cols else None

    if name_col is None:
        df["Name"] = [f"Molecule {i+1}" for i in range(len(df))]
        name_col = "Name"

    df = df.rename(columns={name_col: "Name", smiles_col: "SMILES"})
    df = df[df["SMILES"].notna()].copy()
    df["SMILES"] = df["SMILES"].astype(str).str.strip()
    df["Name"] = df["Name"].astype(str).fillna("").replace("nan", "")

    canon = df["SMILES"].apply(canonicalize_smiles)
    df = df.loc[canon.notna()].copy()
    df["SMILES"] = canon.loc[canon.notna()].values
    df = df.drop_duplicates(subset=["SMILES"]).reset_index(drop=True)

    empty_name = df["Name"].str.strip().eq("") | df["Name"].str.lower().eq("nan")
    if empty_name.any():
        df.loc[empty_name, "Name"] = [f"Molecule {i+1}" for i in df.index[empty_name]]

    return df[["Name", "SMILES"]].copy()


@st.cache_data(show_spinner="Computing RDKit descriptors…")
def compute_descriptor_dataframe(base_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    descriptor_list = Descriptors._descList
    descriptor_names: List[str] = [d[0] for d in descriptor_list]
    descriptor_fns = [d[1] for d in descriptor_list]

    records = []
    for r in base_df.itertuples(index=False):
        mol = smiles_to_mol(r.SMILES)
        if mol is None:
            continue

        vals = []
        for fn in descriptor_fns:
            try:
                v = fn(mol)
            except Exception:
                v = np.nan
            vals.append(v)

        rec = {"Name": r.Name, "SMILES": r.SMILES}
        rec.update(dict(zip(descriptor_names, vals)))
        records.append(rec)

    df = pd.DataFrame.from_records(records)
    if df.empty:
        raise RuntimeError("No valid RDKit molecules could be created from the SMILES column.")

    desc_cols = [c for c in df.columns if c not in {"Name", "SMILES"}]
    all_nan = [c for c in desc_cols if df[c].isna().all()]
    if all_nan:
        df = df.drop(columns=all_nan)
        desc_cols = [c for c in desc_cols if c not in all_nan]

    return df.reset_index(drop=True), desc_cols


# -----------------------------
# Correlation pruning
# -----------------------------
def prune_correlated_features(
    df: pd.DataFrame,
    descriptor_cols: List[str],
    threshold: float = 0.95,
) -> Tuple[List[str], pd.DataFrame]:
    cols = [c for c in descriptor_cols if c in df.columns]
    if len(cols) < 2:
        return cols, pd.DataFrame(columns=["dropped", "kept", "abs_corr"])

    X = df[cols].astype(float).replace([np.inf, -np.inf], np.nan)
    missing_frac = X.isna().mean(axis=0)

    X_imp = X.fillna(X.median(numeric_only=True))
    variances = X_imp.var(axis=0)
    corr = X_imp.corr().abs()

    order = sorted(cols, key=lambda c: (missing_frac[c], -variances[c]))

    kept: List[str] = []
    dropped_rows = []

    for c in order:
        drop = False
        for k in kept:
            v = corr.loc[c, k]
            if pd.notna(v) and float(v) >= threshold:
                drop = True
                dropped_rows.append({"dropped": c, "kept": k, "abs_corr": float(v)})
                break
        if not drop:
            kept.append(c)

    report_df = (
        pd.DataFrame(dropped_rows).sort_values("abs_corr", ascending=False)
        if dropped_rows
        else pd.DataFrame(columns=["dropped", "kept", "abs_corr"])
    )
    return kept, report_df


# -----------------------------
# Feature space (cached)
# -----------------------------
def prepare_feature_matrix(df: pd.DataFrame, cols: List[str]) -> Tuple[np.ndarray, SimpleImputer, StandardScaler]:
    if not cols:
        raise ValueError("No descriptor columns found to build the feature matrix.")
    X = df[cols].astype(float).to_numpy()
    X[~np.isfinite(X)] = np.nan

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_imp = imputer.fit_transform(X)
    X_scaled = scaler.fit_transform(X_imp)
    return X_scaled, imputer, scaler


def fit_pca_embedding(X_scaled: np.ndarray, n_components: int = 2, random_state: int = 0) -> Tuple[np.ndarray, PCA]:
    pca = PCA(n_components=n_components, random_state=random_state)
    coords = pca.fit_transform(X_scaled)
    return coords, pca


@st.cache_data(show_spinner="Building feature space…")
def build_feature_space(
    mol_desc_df: pd.DataFrame,
    descriptor_cols: Tuple[str, ...],
    drop_corr: bool,
    corr_threshold: float,
):
    desc_list = list(descriptor_cols)
    corr_report = pd.DataFrame(columns=["dropped", "kept", "abs_corr"])
    used_cols = desc_list

    if drop_corr:
        used_cols, corr_report = prune_correlated_features(mol_desc_df, desc_list, threshold=corr_threshold)

    X_scaled, imputer, scaler = prepare_feature_matrix(mol_desc_df, used_cols)

    # Always fit PCA(2) so loadings are available even if we visualize with UMAP
    pca_coords, pca_model = fit_pca_embedding(X_scaled, n_components=2, random_state=0)

    df = mol_desc_df.copy().reset_index(drop=True)
    df["pca_x"] = pca_coords[:, 0]
    df["pca_y"] = pca_coords[:, 1]
    return df, X_scaled, used_cols, corr_report, imputer, scaler, pca_model


# -----------------------------
# UMAP embedding (cached) — visualization only
# -----------------------------
@st.cache_data(
    show_spinner="Computing UMAP embedding…",
    hash_funcs={np.ndarray: lambda x: (x.shape, str(x.dtype))},
)
def get_umap_coords(
    feature_key: str,
    X_scaled: np.ndarray,
    n_neighbors: int,
    min_dist: float,
    metric: str,
    random_state: int,
) -> np.ndarray:
    if umap is None:
        raise RuntimeError("UMAP is not installed. Add `umap-learn` to requirements.")
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=int(n_neighbors),
        min_dist=float(min_dist),
        metric=str(metric),
        random_state=int(random_state),
    )
    coords = reducer.fit_transform(X_scaled)
    return coords


# -----------------------------
# Cached NN model (fast hashing)
# -----------------------------
@st.cache_resource(show_spinner=False, hash_funcs={np.ndarray: lambda x: (x.shape, str(x.dtype))})
def get_nn_model(feature_key: str, X_scaled: np.ndarray) -> NearestNeighbors:
    model = NearestNeighbors(metric="euclidean")
    model.fit(X_scaled)
    return model


# -----------------------------
# KMeans cached in session_state (fast UI)
# -----------------------------
def get_kmeans_cached(X_scaled: np.ndarray, k: int, km_key: str) -> Dict[str, object]:
    store_key = f"kmeans::{km_key}::k={k}"
    if store_key in st.session_state:
        return st.session_state[store_key]

    km = KMeans(n_clusters=k, random_state=0, n_init=20)
    labels = km.fit_predict(X_scaled)

    res = {
        "labels": labels.astype(int),
        "centers_scaled": km.cluster_centers_,
        "inertia": float(km.inertia_),
    }
    st.session_state[store_key] = res
    return res


def evaluate_quality(X_scaled: np.ndarray, labels: np.ndarray, inertia: float) -> Tuple[float, float]:
    if len(np.unique(labels)) < 2:
        return float(inertia), float("nan")
    n = len(X_scaled)
    sample_size = min(2000, n)
    try:
        sil = float(silhouette_score(X_scaled, labels, sample_size=sample_size, random_state=0))
    except Exception:
        sil = float("nan")
    return float(inertia), sil


def compute_cluster_exemplars(
    X_scaled: np.ndarray,
    labels: np.ndarray,
    centers_scaled: np.ndarray,
    top_n: int = 5,
) -> Dict[int, List[int]]:
    exemplars = {}
    for c in range(centers_scaled.shape[0]):
        idxs = np.where(labels == c)[0]
        if len(idxs) == 0:
            exemplars[c] = []
            continue
        d = np.linalg.norm(X_scaled[idxs] - centers_scaled[c], axis=1)
        best = idxs[np.argsort(d)[:top_n]]
        exemplars[c] = best.tolist()
    return exemplars


# -----------------------------
# PCA loadings table (Explain axes)
# -----------------------------
def pca_loading_table(pca_model: PCA, feature_names: List[str], top_n: int = 10) -> pd.DataFrame:
    """
    Return a table with top positive and negative loadings for PC1 and PC2.
    Note: PCA components_ are unit vectors; loadings here are component weights.
    """
    if not hasattr(pca_model, "components_") or pca_model.components_.shape[0] < 2:
        return pd.DataFrame()

    rows = []
    for pc_i, pc_name in [(0, "PC1"), (1, "PC2")]:
        w = pca_model.components_[pc_i]
        w = np.asarray(w)

        pos_idx = np.argsort(w)[-top_n:][::-1]
        neg_idx = np.argsort(w)[:top_n]

        for idx in pos_idx:
            rows.append({"PC": pc_name, "Direction": "+", "Descriptor": feature_names[idx], "Loading": float(w[idx])})
        for idx in neg_idx:
            rows.append({"PC": pc_name, "Direction": "-", "Descriptor": feature_names[idx], "Loading": float(w[idx])})

    out = pd.DataFrame(rows)
    out["abs_loading"] = out["Loading"].abs()
    out = out.sort_values(["PC", "abs_loading"], ascending=[True, False]).drop(columns=["abs_loading"])
    return out.reset_index(drop=True)


# -----------------------------
# Streamlit UI
# -----------------------------
def main():
    st.set_page_config(page_title="Carboxylic Acids – RDKit Clustering", layout="wide")
    st.title("Carboxylic Acids – RDKit Descriptor Clustering")

    if not DATA_PATH.exists():
        st.error(f"Excel file not found at `{DATA_PATH}`. Place `All Carboxylic Acids.xlsx` next to `app.py`.")
        return

    st.sidebar.header("Controls")

    strict_hash = st.sidebar.checkbox("Strict cache invalidation (slower)", value=False)
    fp = fingerprint(DATA_PATH, strict_hash)

    base_df = load_raw_dataframe(DATA_PATH, fp)
    mol_desc_df, descriptor_cols = compute_descriptor_dataframe(base_df)

    n = len(mol_desc_df)
    if n < 3:
        st.error("Need at least 3 valid molecules to cluster/visualize.")
        return

    st.sidebar.write(f"**Molecules (valid + deduped):** {len(base_df)}")
    st.sidebar.write(f"**With RDKit descriptors:** {n}")
    st.sidebar.write(f"**RDKit descriptors available:** {len(descriptor_cols)}")

    # Feature selection
    st.sidebar.subheader("Feature selection")
    drop_corr = st.sidebar.checkbox("Drop highly correlated descriptors", value=True)
    corr_threshold = st.sidebar.slider("Correlation threshold (|r| ≥ drop)", 0.80, 0.99, 0.95, 0.01)
    show_corr_report = st.sidebar.checkbox("Show correlation drop report", value=False)

    df, X_scaled, used_cols, corr_report, imputer, scaler, pca_model = build_feature_space(
        mol_desc_df, tuple(descriptor_cols), drop_corr, corr_threshold
    )

    st.sidebar.write(f"**Descriptors used:** {len(used_cols)}")
    if drop_corr:
        st.sidebar.write(f"**Dropped (correlated):** {len(descriptor_cols) - len(used_cols)}")
        if show_corr_report and not corr_report.empty:
            with st.sidebar.expander("Dropped features (top correlations)", expanded=False):
                st.dataframe(corr_report.head(50), use_container_width=True)

    # --- Plot embedding toggle (PCA vs UMAP) ---
    st.sidebar.subheader("Visualization")
    emb_mode = st.sidebar.radio("2D embedding for plot", ["PCA (fast + interpretable)", "UMAP (better neighborhoods)"])

    umap_neighbors = 15
    umap_min_dist = 0.10
    umap_metric = "euclidean"

    if emb_mode.startswith("UMAP"):
        if umap is None:
            st.sidebar.error("UMAP not installed. Add `umap-learn` to requirements to enable.")
            emb_mode = "PCA (fast + interpretable)"
        else:
            with st.sidebar.expander("UMAP parameters", expanded=False):
                umap_neighbors = st.slider("n_neighbors", 5, 50, 15, 1)
                umap_min_dist = st.slider("min_dist", 0.0, 0.99, 0.10, 0.01)
                umap_metric = st.selectbox("metric", ["euclidean", "manhattan", "cosine"], index=0)

    # K + detail controls
    k_max = min(20, len(df) - 1)
    k_default = min(8, k_max)
    k = st.sidebar.slider("Number of clusters (K)", 2, k_max, k_default, 1)
    nn_count = st.sidebar.slider("Nearest neighbors to show", 3, 10, 5, 1)
    show_centroids = st.sidebar.checkbox("Show centroids", value=True)
    exemplars_n = st.sidebar.slider("Exemplars per centroid", 3, 12, 6, 1)

    # Plot performance controls
    st.sidebar.subheader("Plot performance")
    default_fast_plot = len(df) > 5000
    fast_plot = st.sidebar.checkbox("Fast plot (WebGL)", value=default_fast_plot)
    show_names_hover = st.sidebar.checkbox("Show names on hover (slower)", value=(len(df) <= 3000))

    max_render_default = 8000 if len(df) > 8000 else len(df)
    max_points = st.sidebar.slider(
        "Max points to render",
        1000,
        max(1000, min(50000, len(df))),
        max_render_default,
        1000,
    )
    point_size = st.sidebar.slider("Point size", 4, 14, 7, 1)

    # Force fresh plot key on K changes
    if "k_nonce" not in st.session_state:
        st.session_state.k_nonce = 0
        st.session_state.prev_k = k
    if k != st.session_state.prev_k:
        st.session_state.k_nonce += 1
        st.session_state.prev_k = k
        st.session_state.pop("last_selected", None)
    plot_key = f"plot_{k}_{st.session_state.k_nonce}"

    # Cache keys
    feature_key = stable_key(fp, drop_corr, corr_threshold, tuple(used_cols))

    # KMeans cached
    km = get_kmeans_cached(X_scaled, k, feature_key)
    labels = km["labels"]  # type: ignore
    centers_scaled = km["centers_scaled"]  # type: ignore
    inertia_val = km["inertia"]  # type: ignore

    df = df.copy()
    df["cluster"] = labels

    inertia, sil = evaluate_quality(X_scaled, labels, inertia_val)  # type: ignore

    evr = getattr(pca_model, "explained_variance_ratio_", None)
    evr_txt = ""
    if evr is not None and len(evr) >= 2:
        evr_txt = f" | PCA explained var: PC1={evr[0]:.2%}, PC2={evr[1]:.2%}"

    st.subheader(f"Embedding + Clusters (K = {k})")
    st.caption(
        f"Inertia: **{inertia:.2f}** | Silhouette (sampled): **{sil:.3f}**"
        f" | Unique clusters: **{int(pd.Series(labels).nunique())}**{evr_txt}"
    )

    # NN model cached
    nn_model = get_nn_model(feature_key, X_scaled)

    exemplars = compute_cluster_exemplars(X_scaled, labels, centers_scaled, top_n=exemplars_n)  # type: ignore

    # Download
    out_csv = df[["Name", "SMILES", "cluster", "pca_x", "pca_y"]].to_csv(index=False).encode("utf-8")
    st.sidebar.download_button(
        "Download clusters + PCA coords (CSV)",
        data=out_csv,
        file_name=f"carboxylic_acids_k{k}.csv",
        mime="text/csv",
        use_container_width=True,
    )

    # ---- Choose visualization coordinates (PCA vs UMAP) ----
    if emb_mode.startswith("UMAP"):
        coords = get_umap_coords(
            feature_key=feature_key,
            X_scaled=X_scaled,
            n_neighbors=umap_neighbors,
            min_dist=umap_min_dist,
            metric=umap_metric,
            random_state=0,
        )
        df["vis_x"] = coords[:, 0]
        df["vis_y"] = coords[:, 1]
        xlab, ylab = "UMAP1", "UMAP2"
        vis_note = "UMAP axes are not directly interpretable; use PCA loadings below to explain variance drivers."
    else:
        df["vis_x"] = df["pca_x"]
        df["vis_y"] = df["pca_y"]
        xlab, ylab = "PC1", "PC2"
        vis_note = "PCA axes are interpretable; see loadings below."

    # ---- Plot df (downsample visualization only) ----
    if len(df) <= max_points:
        plot_df = df
    else:
        plot_df = df.sample(max_points, random_state=0).sort_index()

    Scatter = go.Scattergl if fast_plot else go.Scatter

    # ---- Centroids in visual space (mean of cluster points) ----
    # avoids needing PCA.transform / UMAP.transform and stays consistent with current labels
    cent_xy = (
        df.groupby("cluster")[["vis_x", "vis_y"]]
        .mean()
        .reindex(list(range(k)))
        .to_numpy()
    )

    # ---- Build figure ----
    fig = go.Figure()

    if show_names_hover:
        fig.add_trace(
            Scatter(
                x=plot_df["vis_x"],
                y=plot_df["vis_y"],
                mode="markers",
                name="Molecules",
                marker=dict(
                    color=plot_df["cluster"].astype(int).tolist(),
                    colorscale="Viridis",
                    showscale=True,
                    size=point_size,
                    opacity=0.85,
                    cmin=-0.5,
                    cmax=k - 0.5,
                    line=dict(width=0.5, color="rgba(0,0,0,0.25)"),
                    colorbar=dict(title="Cluster", tickmode="array", tickvals=list(range(k))),
                ),
                text=plot_df["Name"],
                customdata=[f"mol:{i}" for i in plot_df.index.to_list()],
                hovertemplate="Name: %{text}<br>Cluster: %{marker.color}<extra></extra>",
            )
        )
    else:
        fig.add_trace(
            Scatter(
                x=plot_df["vis_x"],
                y=plot_df["vis_y"],
                mode="markers",
                name="Molecules",
                marker=dict(
                    color=plot_df["cluster"].astype(int).tolist(),
                    colorscale="Viridis",
                    showscale=True,
                    size=point_size,
                    opacity=0.85,
                    cmin=-0.5,
                    cmax=k - 0.5,
                    line=dict(width=0.5, color="rgba(0,0,0,0.25)"),
                    colorbar=dict(title="Cluster", tickmode="array", tickvals=list(range(k))),
                ),
                customdata=[f"mol:{i}" for i in plot_df.index.to_list()],
                hovertemplate="Idx: %{customdata}<br>Cluster: %{marker.color}<extra></extra>",
            )
        )

    if show_centroids:
        fig.add_trace(
            go.Scatter(
                x=cent_xy[:, 0],
                y=cent_xy[:, 1],
                mode="markers",
                name="Centroids",
                marker=dict(symbol="x", size=max(12, point_size + 6), line=dict(width=2)),
                text=[f"Centroid {c} (n={int((labels == c).sum())})" for c in range(k)],
                customdata=[f"centroid:{c}" for c in range(k)],
                hovertemplate="%{text}<extra></extra>",
            )
        )

    fig.update_layout(
        xaxis_title=xlab,
        yaxis_title=ylab,
        height=650,
        clickmode="event+select",
        dragmode="lasso",
        legend=dict(orientation="h"),
        margin=dict(l=10, r=10, t=10, b=10),
        template="plotly_white",
    )

    col_plot, col_detail = st.columns([2.2, 1.0])

    with col_plot:
        selected_customdata = None

        # native selection (new Streamlit)
        try:
            plot_state = st.plotly_chart(fig, use_container_width=True, key=plot_key, on_select="rerun")
            if plot_state is not None and hasattr(plot_state, "selection"):
                pts = getattr(plot_state.selection, "points", None)
                if pts:
                    selected_customdata = pts[0].get("customdata")
        except TypeError:
            # fallback (older Streamlit)
            if plotly_events is None:
                st.warning(
                    "Selection fallback requires `streamlit-plotly-events`, but it isn't installed. "
                    "Install it or upgrade Streamlit for click selection."
                )
            else:
                events = plotly_events(fig, click_event=True, select_event=True, hover_event=False, key=plot_key)
                if events:
                    selected_customdata = events[0].get("customdata")

        parsed = parse_customdata(selected_customdata)
        if parsed:
            st.session_state.last_selected = parsed

        st.caption(vis_note)

        # Cluster Explorer under plot
        st.markdown("### Cluster Explorer")
        cluster_sizes = pd.Series(labels).value_counts().sort_index()
        bar = go.Figure(data=[go.Bar(x=cluster_sizes.index.astype(int), y=cluster_sizes.values)])
        bar.update_layout(
            height=220,
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis_title="Cluster",
            yaxis_title="Count",
            template="plotly_white",
        )
        st.plotly_chart(bar, use_container_width=True)

        cluster_pick = st.selectbox(
            "Pick a cluster",
            options=list(range(k)),
            index=0,
            key=f"cluster_pick_{k}_{st.session_state.k_nonce}",
        )
        size = int((labels == cluster_pick).sum())
        st.write(f"**Cluster {cluster_pick}** | size: **{size}**")

        ex_idx = exemplars.get(cluster_pick, [])
        if ex_idx:
            ex_df = df.loc[ex_idx, ["Name", "SMILES", "cluster"]].rename(columns={"cluster": "Cluster"})
            st.write("**Exemplars (closest to centroid, within cluster)**")
            st.dataframe(ex_df, use_container_width=True)

            mol = smiles_to_mol(df.loc[ex_idx[0], "SMILES"])
            png = mol_to_png_bytes(mol)
            if png:
                st.write("**Top exemplar structure**")
                st.image(png)
        else:
            st.info("No exemplars found for this cluster.")

        # ---- Explain the axes: PCA loadings table ----
        with st.expander("Explain axes: PCA loadings (top drivers for PC1/PC2)", expanded=False):
            load_df = pca_loading_table(pca_model, used_cols, top_n=10)
            if load_df.empty:
                st.info("PCA loadings unavailable.")
            else:
                st.write(
                    "These are the descriptors with the largest positive/negative weights in the PCA components. "
                    "They explain which descriptors are driving PC1/PC2 variance (and often the long 'arm')."
                )
                st.dataframe(load_df, use_container_width=True)

    with col_detail:
        st.markdown("### Click Selection")
        sel = st.session_state.get("last_selected")
        if not sel:
            st.info("Click a molecule point (or centroid X) in the plot to view details here.")
            return

        kind, idx = sel

        if kind == "mol":
            row = df.loc[idx]
            st.write("**Selected molecule**")
            st.dataframe(
                pd.DataFrame([{"Name": row["Name"], "Cluster": int(row["cluster"]), "SMILES": row["SMILES"]}]),
                use_container_width=True,
            )

            key_descs = ["MolWt", "MolLogP", "TPSA", "NumHAcceptors", "NumHDonors", "NumRotatableBonds"]
            key_descs = [d for d in key_descs if d in df.columns]
            if key_descs:
                st.write("**Key RDKit descriptors**")
                st.dataframe(row[key_descs].to_frame().T, use_container_width=True)

            nn_n = min(nn_count + 1, len(df))
            distances, indices = nn_model.kneighbors(X_scaled[[idx]], n_neighbors=nn_n)
            neighbor_idx = indices[0][1:nn_n]

            neighbors = df.iloc[neighbor_idx][["Name", "cluster", "SMILES"]].reset_index(drop=True)
            neighbors = neighbors.rename(columns={"cluster": "Cluster"})
            st.write("**Nearest neighbors (descriptor space)**")
            st.dataframe(neighbors, use_container_width=True)

            mol = smiles_to_mol(row["SMILES"])
            png = mol_to_png_bytes(mol)
            if png:
                st.write("**Structure**")
                st.image(png)

        elif kind == "centroid":
            c = idx
            size = int((labels == c).sum())
            st.write(f"**Selected centroid: Cluster {c}** (n={size})")

            st.write("**Centroid (visual space)**")
            st.dataframe(
                pd.DataFrame([{"cluster": c, xlab: float(cent_xy[c, 0]), ylab: float(cent_xy[c, 1])}]),
                use_container_width=True,
            )

            ex_idx = exemplars.get(c, [])
            if ex_idx:
                st.write("**Closest molecules to centroid (within cluster)**")
                ex_df = df.loc[ex_idx, ["Name", "SMILES", "cluster"]].rename(columns={"cluster": "Cluster"})
                st.dataframe(ex_df, use_container_width=True)

                mol = smiles_to_mol(df.loc[ex_idx[0], "SMILES"])
                png = mol_to_png_bytes(mol)
                if png:
                    st.write("**Top centroid exemplar structure**")
                    st.image(png)
            else:
                st.info("No exemplars found for this centroid.")
        else:
            st.warning(f"Unknown selection kind: {kind}")


if __name__ == "__main__":
    main()
