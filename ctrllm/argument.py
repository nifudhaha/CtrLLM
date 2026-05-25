import os
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union
from sentence_transformers import SentenceTransformer
import hdbscan

# WIBA for argument detection
try:
    from wiba import WIBA
    WIBA_AVAILABLE = True
except ImportError:
    WIBA_AVAILABLE = False

# UMAP for dimensionality reduction
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False


# WIBA hard limit per API call
WIBA_MAX_CHARS = 9_000   


class ArgumentAnalyzer:

    def __init__(self,
                 lang: str = "en",
                 embedding_model: str = "all-mpnet-base-v2",
                 min_cluster_size: int = 2,
                 min_samples: int = 1,
                 cluster_selection_epsilon: float = 0.0,
                 use_umap: bool = True,
                 umap_n_neighbors: int = 15,
                 umap_min_dist: float = 0.1,
                 umap_metric: str = 'cosine',
                 wiba_token: Optional[str] = None):

        if not WIBA_AVAILABLE:
            raise ImportError(
                "WIBA package not installed. "
                "Install with: pip install wiba"
            )

        if use_umap and not UMAP_AVAILABLE:
            raise ImportError(
                "UMAP package not installed. "
                "Install with: pip install umap-learn"
            )

        self.lang = lang

        # Setup WIBA
        token = wiba_token or os.getenv("WIBA_API_TOKEN")
        if not token:
            raise ValueError(
                "WIBA_API_TOKEN not provided. "
                "Set it with: export WIBA_API_TOKEN='your-token' or pass wiba_token parameter"
            )
        self.wiba_client = WIBA(api_token=token)

        # Setup embedding model
        self.embedding_model = SentenceTransformer(embedding_model)

        # HDBSCAN parameters
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.cluster_selection_epsilon = cluster_selection_epsilon

        # UMAP parameters
        self.use_umap = use_umap
        self.umap_n_neighbors = umap_n_neighbors
        self.umap_min_dist = umap_min_dist
        self.umap_metric = umap_metric

        print(f"ArgumentAnalyzer initialized with WIBA + HDBSCAN" + (" + UMAP" if use_umap else ""))
        print(f"  Language: {lang}")
        print(f"  Embedding model: {embedding_model}")
        print(f"  HDBSCAN min_cluster_size: {min_cluster_size}")
        if use_umap:
            print(f"  UMAP enabled: n_neighbors={umap_n_neighbors}, min_dist={umap_min_dist}")

        # State variables
        self.arguments = []
        self.argument_embeddings = None
        self.umap_embeddings = None
        self.umap_reducer = None
        self.clusters = None
        self.cluster_labels = None
        self.clusterer = None
        self._current_text = None
        self._wiba_results_df = None  # Store full WIBA results for downstream use


    def discover_arguments(self,
                           data: Union[str, pd.DataFrame],
                           text_column: Optional[str] = None,
                           window_size: int = 3,
                           step_size: int = 1) -> pd.DataFrame:

        if isinstance(data, str):
            return self._discover_arguments_from_text(data, window_size, step_size)

        elif isinstance(data, pd.DataFrame):
            if text_column is None:
                raise ValueError("text_column must be specified when data is DataFrame")

            all_results = []
            for idx, row in data.iterrows():
                text = row[text_column]
                result_df = self._discover_arguments_from_text(text, window_size, step_size)
                result_df['source_index'] = idx
                all_results.append(result_df)

            return pd.concat(all_results, ignore_index=True)

        else:
            raise ValueError("data must be either str or pandas DataFrame")


    @staticmethod
    def _chunk_text(text: str, max_chars: int = WIBA_MAX_CHARS) -> List[str]:

        import re
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        chunks, current, current_len = [], [], 0

        for sent in sentences:
            sent_len = len(sent) + 1          # +1 for the space we'll rejoin with
            if current and current_len + sent_len > max_chars:
                chunks.append(' '.join(current))
                current, current_len = [], 0
            current.append(sent)
            current_len += sent_len

        if current:
            chunks.append(' '.join(current))

        return chunks

    def _discover_arguments_from_text(self,
                                       text: str,
                                       window_size: int,
                                       step_size: int) -> pd.DataFrame:

        empty_df = pd.DataFrame(columns=[
            "text_segment", "is_argument", "argument_confidence",
            "claims", "premises",
            "topic_fine", "topic_broad",
            "stance_fine", "stance_broad",
            "argument_type", "argument_scheme",
        ])

        # -- Split into WIBA-safe chunks if needed --
        if len(text) > WIBA_MAX_CHARS:
            chunks = self._chunk_text(text, WIBA_MAX_CHARS)
            print(f"Text is {len(text):,} chars — splitting into {len(chunks)} chunks "
                  f"(≤{WIBA_MAX_CHARS:,} chars each) for WIBA...")
        else:
            chunks = [text]

        all_chunk_dfs = []
        for i, chunk in enumerate(chunks):
            if len(chunks) > 1:
                print(f"  Chunk {i+1}/{len(chunks)} ({len(chunk):,} chars) — "
                      f"calling WIBA discover_arguments() "
                      f"(window_size={window_size}, step_size={step_size})...")
            else:
                print(f"Calling WIBA discover_arguments() "
                      f"(window_size={window_size}, step_size={step_size})...")

            try:
                raw_df = self.wiba_client.discover_arguments(
                    chunk,
                    window_size=window_size,
                    step_size=step_size,
                )
            except Exception as e:
                print(f"  Warning: WIBA discover_arguments() failed on chunk {i+1}: {e}")
                continue   # skip this chunk, try the rest

            if raw_df is not None and len(raw_df) > 0:
                all_chunk_dfs.append(raw_df)

        if not all_chunk_dfs:
            print("Warning: all WIBA chunks failed or returned empty results.")
            return empty_df

        # Concatenate chunks
        raw_df = pd.concat(all_chunk_dfs, ignore_index=True)

        if raw_df is None or len(raw_df) == 0:
            return empty_df

        # ── Inspect actual columns returned by WIBA ──────────────────────
        print(f"  WIBA raw columns: {list(raw_df.columns)}")


        col_map = {
            # is_argument variants
            "argument_prediction":   "is_argument",   # older WIBA versions
            "is_arg":                "is_argument",
            # argument_confidence variants
            "confidence":            "argument_confidence",
            "confidence_score":      "argument_confidence",
            "argument_confidence_score": "argument_confidence",
        }
        raw_df = raw_df.rename(columns={k: v for k, v in col_map.items()
                                         if k in raw_df.columns})

        # If is_argument is a string ("Argument"/"NoArgument"), convert to bool
        if "is_argument" in raw_df.columns:
            if raw_df["is_argument"].dtype == object:
                raw_df["is_argument"] = raw_df["is_argument"].isin(
                    ["Argument", "argument", "True", "true", True]
                )
        else:
            # Column still missing after rename — add it as all-False so downstream
            # code doesn't crash; user will see 0 arguments and can investigate.
            print("  Warning: could not find an 'is_argument' column in WIBA output. "
                  "All segments will be treated as non-arguments.")
            raw_df["is_argument"] = False

        if "argument_confidence" not in raw_df.columns:
            print("  Warning: could not find an 'argument_confidence' column. Defaulting to 0.0.")
            raw_df["argument_confidence"] = 0.0

        if "text_segment" not in raw_df.columns:
            # Some versions use 'text' or 'segment'
            for alt in ("text", "segment", "sentence"):
                if alt in raw_df.columns:
                    raw_df = raw_df.rename(columns={alt: "text_segment"})
                    break
            else:
                print("  Warning: could not find a text column in WIBA output.")
                raw_df["text_segment"] = ""

        n_args = int(raw_df["is_argument"].sum())
        print(f"✓ WIBA returned {len(raw_df)} segments ({n_args} arguments)")

        return raw_df


    def extract_arguments(self,
                          results_df: pd.DataFrame,
                          confidence_threshold: float = 0.5) -> List[str]:
        self._wiba_results_df = results_df

        if results_df is None or len(results_df) == 0:
            self.arguments = []
            return self.arguments

        # Filter: is_argument=True AND confidence above threshold
        mask = (
            results_df["is_argument"].astype(bool) &
            (results_df["argument_confidence"] >= confidence_threshold)
        )
        filtered = results_df[mask]

        if len(filtered) == 0:
            print(f"No arguments found above confidence threshold ({confidence_threshold})")
            self.arguments = []
            return self.arguments

        print(f"Extracted {len(filtered)} arguments (confidence ≥ {confidence_threshold})")
        self.arguments = filtered["text_segment"].tolist()
        return self.arguments


    def process_text(self, text: str):
        """
        Store text for compatibility with old API.

        Args:
            text: Input text to analyze
        """
        self._current_text = text
        print(f"Text loaded: {len(text)} characters")



    def compute_argument_embeddings(self) -> np.ndarray:

        if not self.arguments:
            raise ValueError("No arguments detected. Call extract_arguments() first.")

        self.argument_embeddings = self.embedding_model.encode(self.arguments)
        return self.argument_embeddings


    def apply_umap(self, n_components: Optional[int] = None) -> np.ndarray:

        if self.argument_embeddings is None:
            self.compute_argument_embeddings()

        n_args = len(self.arguments)

        if n_args < 15:
            print(f"Skipping UMAP: only {n_args} arguments (need ≥15)")
            self.umap_embeddings = self.argument_embeddings
            return self.umap_embeddings

        if n_components is None:
            n_components = 10

        max_components = min(self.argument_embeddings.shape[1], n_args - 1)
        n_components = min(n_components, max_components)
        n_neighbors = min(self.umap_n_neighbors, n_args - 1)

        print(f"UMAP transformation: {self.argument_embeddings.shape} → ({n_args}, {n_components})")

        self.umap_reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=self.umap_min_dist,
            metric=self.umap_metric,
            random_state=42
        )

        self.umap_embeddings = self.umap_reducer.fit_transform(self.argument_embeddings)
        print(f"UMAP complete: variance explained ≈ {self._estimate_variance_explained():.1%}")

        return self.umap_embeddings

    def _estimate_variance_explained(self) -> float:
        """Estimate variance explained by UMAP (approximation)."""
        if self.umap_embeddings is None or self.argument_embeddings is None:
            return 0.0
        original_var = np.var(self.argument_embeddings)
        reduced_var = np.var(self.umap_embeddings)
        return min(1.0, reduced_var / (original_var + 1e-10))


    def cluster_arguments_hdbscan(self) -> np.ndarray:

        if self.argument_embeddings is None:
            self.compute_argument_embeddings()

        if len(self.arguments) < 2:
            self.cluster_labels = np.array([0] * len(self.arguments))
            self.clusterer = None
            return self.cluster_labels

        if self.use_umap and len(self.arguments) >= 15:
            embeddings_for_clustering = self.apply_umap()
            print(f"Clustering with UMAP embeddings: {embeddings_for_clustering.shape}")
        else:
            embeddings_for_clustering = self.argument_embeddings
            print(f"Clustering with original embeddings: {embeddings_for_clustering.shape}")

        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric='euclidean' if embeddings_for_clustering is self.argument_embeddings else 'euclidean',
            cluster_selection_epsilon=self.cluster_selection_epsilon,
            cluster_selection_method='eom'
        )

        self.cluster_labels = self.clusterer.fit_predict(embeddings_for_clustering)

        # Handle noise points (label = -1) by assigning to nearest cluster
        if -1 in self.cluster_labels:
            noise_indices = np.where(self.cluster_labels == -1)[0]
            valid_indices = np.where(self.cluster_labels != -1)[0]

            if len(valid_indices) > 0:
                for idx in noise_indices:
                    distances = np.linalg.norm(
                        embeddings_for_clustering[valid_indices] - embeddings_for_clustering[idx],
                        axis=1
                    )
                    nearest_valid_idx = valid_indices[np.argmin(distances)]
                    self.cluster_labels[idx] = self.cluster_labels[nearest_valid_idx]
            else:
                self.cluster_labels = np.zeros(len(self.cluster_labels), dtype=int)

        self.clusters = {}
        for label in set(self.cluster_labels):
            indices = np.where(self.cluster_labels == label)[0]
            self.clusters[label] = {
                'arguments': [self.arguments[i] for i in indices],
                'embeddings': self.argument_embeddings[indices],
                'size': len(indices)
            }

        print(f"HDBSCAN found {len(self.clusters)} clusters")
        return self.cluster_labels

    def get_cluster_probabilities(self) -> Optional[np.ndarray]:
        """Get soft cluster membership probabilities from HDBSCAN."""
        if self.clusterer is None:
            return None
        return self.clusterer.probabilities_

    def get_cluster_persistences(self) -> Optional[np.ndarray]:
        """Get cluster persistence values (stability) from HDBSCAN."""
        if self.clusterer is None:
            return None
        if hasattr(self.clusterer, 'cluster_persistence_'):
            return self.clusterer.cluster_persistence_
        return None

    def main_vs_fringe_perspective(self) -> Dict[str, float]:

        if self.clusters is None:
            raise ValueError("No clusters found. Call cluster_arguments_hdbscan() first.")

        if len(self.clusters) == 0:
            return {
                'main_ratio': 0.0,
                'main_cluster_size': 0,
                'total_arguments': 0,
                'num_clusters': 0,
                'fringe_clusters': 0
            }

        cluster_sizes = [cluster['size'] for cluster in self.clusters.values()]
        total_args = sum(cluster_sizes)
        max_cluster_size = max(cluster_sizes)
        main_ratio = max_cluster_size / total_args if total_args > 0 else 0.0
        fringe_clusters = sum(1 for size in cluster_sizes if size < max_cluster_size)

        return {
            'main_ratio': float(main_ratio),
            'main_cluster_size': int(max_cluster_size),
            'total_arguments': int(total_args),
            'num_clusters': len(self.clusters),
            'fringe_clusters': int(fringe_clusters),
            'fringe_ratio': float(1 - main_ratio)
        }

    def argument_diversity(self) -> Dict[str, float]:
        if self.clusters is None:
            raise ValueError("No clusters found. Call cluster_arguments_hdbscan() first.")

        n_args = len(self.arguments)
        k_clusters = len(self.clusters)

        if n_args == 0:
            return {
                'arg_diversity': 0.0,
                'num_arguments': 0,
                'num_clusters': 0,
                'cluster_entropy': 0.0
            }

        diversity = k_clusters / np.log(1 + n_args)
        cluster_sizes = [cluster['size'] for cluster in self.clusters.values()]
        cluster_probs = np.array(cluster_sizes) / sum(cluster_sizes)
        cluster_entropy = -np.sum(cluster_probs * np.log2(cluster_probs + 1e-10))

        return {
            'arg_diversity': float(diversity),
            'num_arguments': int(n_args),
            'num_clusters': int(k_clusters),
            'cluster_entropy': float(cluster_entropy),
            'normalized_diversity': float(diversity / np.log2(n_args + 1)) if n_args > 0 else 0.0
        }

    def argument_distinctness(self) -> Dict[str, float]:

        if self.clusters is None:
            raise ValueError("No clusters found. Call cluster_arguments_hdbscan() first.")

        if len(self.clusters) < 2:
            return {
                'narrative_distinctness': 0.0,
                'mean_distance': 0.0,
                'min_distance': 0.0,
                'max_distance': 0.0,
                'num_clusters': len(self.clusters)
            }

        centroids = []
        for cluster in self.clusters.values():
            centroid = np.mean(cluster['embeddings'], axis=0)
            centroids.append(centroid)

        centroids = np.array(centroids)

        distances = []
        for i in range(len(centroids)):
            for j in range(i + 1, len(centroids)):
                cos_sim = np.dot(centroids[i], centroids[j]) / (
                    np.linalg.norm(centroids[i]) * np.linalg.norm(centroids[j])
                )
                distances.append(1 - cos_sim)

        if not distances:
            return {
                'narrative_distinctness': 0.0,
                'mean_distance': 0.0,
                'min_distance': 0.0,
                'max_distance': 0.0,
                'num_clusters': len(self.clusters)
            }

        mean_dist = np.mean(distances)
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        nd = np.sqrt(mean_dist * min_dist)

        return {
            'narrative_distinctness': float(nd),
            'mean_distance': float(mean_dist),
            'min_distance': float(min_dist),
            'max_distance': float(max_dist),
            'num_clusters': len(self.clusters),
            'std_distance': float(np.std(distances))
        }

    def deliberation_intensity(self) -> Dict[str, float]:

        if self.clusters is None:
            raise ValueError("No clusters found. Call cluster_arguments_hdbscan() first.")

        diversity = self.argument_diversity()
        distinctness = self.argument_distinctness()

        ddiv = diversity.get('normalized_diversity', 0.0)
        dnd = distinctness.get('narrative_distinctness', 0.0)
        delib_intensity = (ddiv + dnd) / 2.0

        return {
            'deliberation_intensity': float(delib_intensity),
            'diversity_component': float(ddiv),
            'distinctness_component': float(dnd)
        }



    def get_all_metrics(self,
                        text: Optional[str] = None,
                        results_df: Optional[pd.DataFrame] = None,
                        window_size: int = 3,
                        step_size: int = 1,
                        confidence_threshold: float = 0.5,
                        **kwargs) -> Dict:

        # Resolve text source
        if text is None and results_df is None:
            if self._current_text is not None:
                text = self._current_text
            else:
                raise ValueError(
                    "Must provide either text, results_df, or call process_text() first"
                )

        # Step 1: Discover arguments via WIBA
        if results_df is None:
            results_df = self.discover_arguments(
                text, window_size=window_size, step_size=step_size
            )

        # Step 2: Extract arguments (WIBA already resolved overlaps)
        self.extract_arguments(results_df, confidence_threshold=confidence_threshold)

        num_windows = len(results_df)

        if len(self.arguments) == 0:
            return {
                'num_arguments': 0,
                'num_windows': num_windows,
                'argumentativeness': 0.0,
                'backend': 'WIBA + HDBSCAN',
                'arguments': [],           # always present so callers never get KeyError
                'argument_details': [],
                'main_vs_fringe': {'main_ratio': 0.0, 'num_clusters': 0},
                'argument_diversity': {'arg_diversity': 0.0, 'num_clusters': 0},
                'argument_distinctness': {'narrative_distinctness': 0.0, 'num_clusters': 0},
                'deliberation_intensity': {'deliberation_intensity': 0.0}
            }

        # Step 3: Embed arguments
        self.compute_argument_embeddings()

        # Step 4: Cluster with HDBSCAN
        self.cluster_arguments_hdbscan()

        # Step 5: Compute metrics
        main_fringe = self.main_vs_fringe_perspective()
        diversity = self.argument_diversity()
        distinctness = self.argument_distinctness()
        delib_intensity = self.deliberation_intensity()

        argumentativeness = len(self.arguments) / num_windows if num_windows > 0 else 0.0

        probabilities = self.get_cluster_probabilities()
        avg_probability = float(np.mean(probabilities)) if probabilities is not None else None

        umap_info = None
        if self.umap_embeddings is not None:
            umap_info = {
                'applied': True,
                'original_dim': self.argument_embeddings.shape[1],
                'reduced_dim': self.umap_embeddings.shape[1],
                'variance_explained': self._estimate_variance_explained()
            }
        elif self.use_umap and len(self.arguments) < 15:
            umap_info = {
                'applied': False,
                'reason': f'Too few arguments ({len(self.arguments)} < 15)'
            }

        # Enrich argument list with WIBA comprehensive fields if available
        argument_details = []
        if self._wiba_results_df is not None:
            wiba_args = self._wiba_results_df[
                self._wiba_results_df["is_argument"].astype(bool) &
                (self._wiba_results_df["argument_confidence"] >= confidence_threshold)
            ]
            for _, row in wiba_args.iterrows():
                detail = {"text": row["text_segment"]}
                for col in ["claims", "premises", "topic_fine", "topic_broad",
                            "stance_fine", "stance_broad", "argument_type", "argument_scheme"]:
                    if col in row:
                        detail[col] = row[col]
                argument_details.append(detail)

        return {
            'num_arguments': len(self.arguments),
            'num_windows': num_windows,
            'argumentativeness': float(argumentativeness),
            'backend': 'WIBA + HDBSCAN' + (' + UMAP' if self.umap_embeddings is not None else ''),
            'arguments': self.arguments,
            'argument_details': argument_details,  # Full WIBA comprehensive fields per argument
            'umap': umap_info,
            'cluster_quality': {
                'avg_membership_probability': avg_probability,
                'num_stable_clusters': len(self.clusters)
            },
            'main_vs_fringe': main_fringe,
            'argument_diversity': diversity,
            'argument_distinctness': distinctness,
            'deliberation_intensity': delib_intensity
        }
