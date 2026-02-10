import itertools
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd


class VectorizedTriangleArbitrage:
    """向量化三角套利检测器（包含增量更新）"""

    def __init__(self, fee_rate: float = 0.001, data_path: str = "data/filtered_pairs_coins.json"):
        self.fee_multiplier = 1 - fee_rate

        tradable_coins, tradable_pairs = self._load_filtered_pairs(data_path)
        self._precompute_triangles_vectorized(tradable_coins, tradable_pairs)

        # 提取三角形币种与索引
        self.coins = sorted(np.unique(self.triangles))
        self.coin_to_idx = {coin: i for i, coin in enumerate(self.coins)}
        self.n_coins = len(self.coins)
        self.triangles_indices = np.vectorize(self.coin_to_idx.get)(self.triangles).astype(np.int32)

        # 建立 exchange symbol 到 base, quote 索引的映射
        self.df_mapping = self._build_pair_mapping(tradable_pairs)
        self.symbol_to_idx = {symbol: idx for idx, symbol in enumerate(self.df_mapping.index)}
        self.base_indices = self.df_mapping["base_index"].values
        self.quote_indices = self.df_mapping["quote_index"].values

        # 汇率矩阵（对数空间）
        self.rate_matrix = np.full((self.n_coins, self.n_coins), -np.inf, dtype=np.float64)
        np.fill_diagonal(self.rate_matrix, 0.0)

        # 增量映射：edge -> triangle indices
        self._build_edge_to_triangles()

        # 性能统计
        self.stats = {
            "vectorized_checks": 0,
            "opportunities_found": 0,
            "total_processing_time": 0.0,
        }

    @staticmethod
    def _load_filtered_pairs(path: str) -> Tuple[List[str], List[str]]:
        payload = json.loads(Path(path).read_text())
        return payload.get("coins", []), payload.get("pairs", [])

    def _build_pair_mapping(self, tradable_pairs: List[str]) -> pd.DataFrame:
        self.watchlist_pair_to_coin_idx = []
        for pair in tradable_pairs:
            if pair not in self.all_possible_trading_pairs:
                continue
            base, quote = pair.split("/")
            base_idx, quote_idx = self.coin_to_idx[base], self.coin_to_idx[quote]
            self.watchlist_pair_to_coin_idx.append(
                {
                    "pair": pair.replace("/", ""),
                    "base_index": base_idx,
                    "quote_index": quote_idx,
                }
            )
        return pd.DataFrame(self.watchlist_pair_to_coin_idx).set_index("pair")

    def _precompute_triangles_vectorized(self, tradable_coins: List[str], tradable_pairs: List[str]) -> None:
        cycles_undirected = list(itertools.combinations(tradable_coins, 3))
        cycles_undirected = np.array(cycles_undirected)

        i, j, k = cycles_undirected[:, 0], cycles_undirected[:, 1], cycles_undirected[:, 2]

        ij = np.isin(i + "/" + j, tradable_pairs) | np.isin(j + "/" + i, tradable_pairs)
        jk = np.isin(j + "/" + k, tradable_pairs) | np.isin(k + "/" + j, tradable_pairs)
        ki = np.isin(k + "/" + i, tradable_pairs) | np.isin(i + "/" + k, tradable_pairs)

        triangles_undirected = cycles_undirected[ij & jk & ki]
        triangles_reversed = triangles_undirected[:, ::-1]
        triangles_directed = np.vstack([triangles_undirected, triangles_reversed])

        self.triangles = triangles_directed
        self.n_triangles = self.triangles.shape[0]
        print(f"预计算了{self.n_triangles}个三角形套利路径。")

        i, j, k = self.triangles[:, 0], self.triangles[:, 1], self.triangles[:, 2]
        self.all_possible_trading_pairs = set(i + "/" + j) | set(j + "/" + i) | set(j + "/" + k) | set(k + "/" + j) | set(k + "/" + i) | set(i + "/" + k)

    def _build_edge_to_triangles(self) -> None:
        self.edge_to_triangles: Dict[Tuple[int, int], List[int]] = {}
        for idx, (i, j, k) in enumerate(self.triangles_indices):
            for a, b in ((i, j), (j, k), (k, i)):
                edge = (a, b) if a < b else (b, a)
                if edge not in self.edge_to_triangles:
                    self.edge_to_triangles[edge] = []
                self.edge_to_triangles[edge].append(idx)

    def update_price_vectorized_v2(self, symbol_indices: List[int], ij_rates: np.ndarray, ji_rates: np.ndarray) -> None:
        bases = self.base_indices[symbol_indices]
        quotes = self.quote_indices[symbol_indices]
        self.rate_matrix[bases, quotes] = np.log(ij_rates)
        self.rate_matrix[quotes, bases] = np.log(ji_rates)

    def find_arbitrage_vectorized(self, min_profit: float = 0.001) -> List[Dict]:
        start_time = time.perf_counter()
        if self.n_triangles == 0:
            return []

        i_indices = self.triangles_indices[:, 0]
        j_indices = self.triangles_indices[:, 1]
        k_indices = self.triangles_indices[:, 2]

        rates_ij = self.rate_matrix[i_indices, j_indices]
        rates_jk = self.rate_matrix[j_indices, k_indices]
        rates_ki = self.rate_matrix[k_indices, i_indices]

        total_log_returns = rates_ij + rates_jk + rates_ki
        threshold = np.log(1 + min_profit)
        profitable_mask = total_log_returns > threshold
        profitable_indices = np.where(profitable_mask)[0]

        opportunities = []
        for idx in profitable_indices:
            log_return = total_log_returns[idx]
            actual_return = np.exp(log_return) - 1
            opportunities.append(
                {
                    "profit": actual_return * 100,
                    "log_return": log_return,
                    "triangle_index": idx,
                    "path": "-".join(self.triangles[idx]),
                }
            )

        opportunities.sort(key=lambda x: x["profit"], reverse=True)

        processing_time = (time.perf_counter() - start_time) * 1000
        self.stats["vectorized_checks"] += 1
        self.stats["opportunities_found"] += len(opportunities)
        self.stats["total_processing_time"] += processing_time

        return opportunities

    def find_arbitrage_incremental(self, updated_edges: List[Tuple[int, int]], min_profit: float = 0.001) -> List[Dict]:
        if not updated_edges:
            return []

        affected = set()
        for i, j in updated_edges:
            edge = (i, j) if i < j else (j, i)
            if edge in self.edge_to_triangles:
                affected.update(self.edge_to_triangles[edge])

        if not affected:
            return []

        affected_indices = np.array(list(affected), dtype=np.int32)
        triangle_subset = self.triangles_indices[affected_indices]

        i_indices = triangle_subset[:, 0]
        j_indices = triangle_subset[:, 1]
        k_indices = triangle_subset[:, 2]

        rates_ij = self.rate_matrix[i_indices, j_indices]
        rates_jk = self.rate_matrix[j_indices, k_indices]
        rates_ki = self.rate_matrix[k_indices, i_indices]

        total_log_returns = rates_ij + rates_jk + rates_ki
        threshold = np.log(1 + min_profit)
        profitable_mask = total_log_returns > threshold

        opportunities = []
        for local_idx, global_idx in enumerate(affected_indices):
            if profitable_mask[local_idx]:
                log_return = total_log_returns[local_idx]
                actual_return = np.exp(log_return) - 1
                opportunities.append(
                    {
                        "profit": actual_return * 100,
                        "log_return": log_return,
                        "triangle_index": global_idx,
                        "path": "-".join(self.triangles[global_idx]),
                    }
                )

        opportunities.sort(key=lambda x: x["profit"], reverse=True)
        return opportunities
