"""
Rule Bandit Module - LinUCB/UCT based rule selection.
Uses purely mathematical reward updates from sequence-level cost reduction.
"""
import base64
import hashlib
import importlib.util
import json
import os
import re
import numpy as np
from typing import Dict, List, Optional, Tuple, Type
from pathlib import Path

# Context layout: SQL fingerprint hash + plan bottleneck operators + advice groups
FINGERPRINT_HASH_DIM = 32
PLAN_BOTTLENECK_DIM = 24
ADVICE_DIM = 8
DEFAULT_CONTEXT_DIM = FINGERPRINT_HASH_DIM + PLAN_BOTTLENECK_DIM + ADVICE_DIM

# Plan report: operator name inside full-width brackets 【…】
_BRACKET_OP_RE = re.compile(r"【([^】]+)】")
_PCT_AFTER_BRACKET_RE = re.compile(r"代价占比:\s*([\d.]+)\s*%")

_sql_fingerprint_generator: Optional[object] = None
_SQLFingerprintGeneratorCls: Optional[Type] = None


def _get_sql_fingerprint_generator():
    """
    Lazy-load SQLFingerprintGenerator from sql_fingerprint.py only (avoids importing
    global_memory/__init__.py and heavy ChromaDB deps).
    """
    global _sql_fingerprint_generator, _SQLFingerprintGeneratorCls
    if _sql_fingerprint_generator is None:
        if _SQLFingerprintGeneratorCls is None:
            path = Path(__file__).resolve().parent / "global_memory" / "sql_fingerprint.py"
            spec = importlib.util.spec_from_file_location(
                "query_rewriter_sql_fingerprint", path
            )
            if spec is None or spec.loader is None:
                raise ImportError(f"Cannot load SQL fingerprint from {path}")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            _SQLFingerprintGeneratorCls = mod.SQLFingerprintGenerator
        _sql_fingerprint_generator = _SQLFingerprintGeneratorCls()
    return _sql_fingerprint_generator


class RuleBandit:
    """
    Contextual Bandit for SQL rewrite rule selection using LinUCB algorithm.
    
    Core responsibilities:
    1. Calculate UCT scores for rule ranking (LinUCB formula)
    2. Convert sequence-level cost reduction into rule rewards
    3. Update online learning statistics (A, b matrices)
    4. Provide context extraction for state representation
    
    Reward allocation is computed from observed cost reduction (pure math).
    """
    
    def __init__(
        self,
        storage_path: Optional[str] = None,
        context_dim: int = DEFAULT_CONTEXT_DIM,
        alpha: float = 1.0,
    ):
        """
        Initialize RuleBandit
        
        Args:
            storage_path: Path to store bandit statistics
            context_dim: Dimension of context feature vector (must match extract_context output)
            alpha: Exploration parameter for UCB (higher = more exploration)
        """
        if storage_path is None:
            project_root = Path(__file__).parent.parent.parent
            storage_path = str(project_root / "data" / "global_memory" / "rule_bandit_stats.json")
        
        self.storage_path = storage_path
        self.context_dim = context_dim
        self.alpha = alpha
        
        # Per-rule statistics for LinUCB
        # "A": base64(float64 row-major d×d) or legacy nested list; "b": base64(float64 d) or legacy list
        self.rule_stats: Dict[str, Dict] = {}
        
        self._load_stats()
        print(f"✅ RuleBandit initialized (storage: {self.storage_path}, dim={context_dim})")
        print(f"   Tracked rules: {len([k for k in self.rule_stats.keys() if not k.startswith('_')])}")
    
    def _load_stats(self):
        """Load statistics from persistent storage"""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    self.rule_stats = json.load(f)
                count = len([k for k in self.rule_stats.keys() if not k.startswith('_')])
                print(f"📊 Loaded {count} rule statistics")
                self._drop_stats_wrong_dim()
            except Exception as e:
                print(f"⚠️ Failed to load bandit stats: {e}")
                self.rule_stats = {}
    
    def _save_stats(self):
        """Save statistics to persistent storage"""
        try:
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
            self._normalize_stats_for_save()
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(self.rule_stats, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️ Failed to save bandit stats: {e}")

    @staticmethod
    def _encode_f64_buffer(arr: np.ndarray) -> str:
        """Row-major float64 bytes → standard base64 ASCII (one line, no whitespace)."""
        return base64.standard_b64encode(
            np.asarray(arr, dtype=np.float64).tobytes(order="C")
        ).decode("ascii")

    def _encode_A_matrix(self, A: np.ndarray) -> str:
        return self._encode_f64_buffer(A.reshape(self.context_dim, self.context_dim))

    def _encode_b_vector(self, b: np.ndarray) -> str:
        return self._encode_f64_buffer(np.asarray(b, dtype=np.float64).reshape(-1))

    def _A_to_numpy(self, st: Dict) -> np.ndarray:
        a = st.get("A")
        if isinstance(a, str):
            raw = base64.standard_b64decode(a.encode("ascii"))
            need = self.context_dim * self.context_dim * 8
            if len(raw) != need:
                raise ValueError(f"A payload size {len(raw)} != expected {need}")
            return np.frombuffer(raw, dtype=np.float64).reshape(
                self.context_dim, self.context_dim
            ).copy()
        return np.asarray(a, dtype=np.float64)

    def _b_to_numpy(self, st: Dict) -> np.ndarray:
        b = st.get("b")
        if isinstance(b, str):
            raw = base64.standard_b64decode(b.encode("ascii"))
            need = self.context_dim * 8
            if len(raw) != need:
                raise ValueError(f"b payload size {len(raw)} != expected {need}")
            return np.frombuffer(raw, dtype=np.float64).copy()
        return np.asarray(b, dtype=np.float64).reshape(-1)

    def _normalize_stats_for_save(self) -> None:
        """Convert legacy list A/b to compact base64 before writing JSON."""
        for rid, st in self.rule_stats.items():
            if rid.startswith("_"):
                continue
            A = st.get("A")
            if isinstance(A, list):
                st["A"] = self._encode_A_matrix(np.asarray(A, dtype=np.float64))
            b = st.get("b")
            if isinstance(b, list):
                st["b"] = self._encode_b_vector(np.asarray(b, dtype=np.float64))

    def _get_or_init_rule(self, rule_id: str) -> Dict:
        """Get or initialize statistics for a rule"""
        if rule_id not in self.rule_stats:
            # LinUCB initialization: A = I (identity), b = 0
            self.rule_stats[rule_id] = {
                "A": self._encode_A_matrix(np.eye(self.context_dim)),
                "b": self._encode_b_vector(np.zeros(self.context_dim, dtype=np.float64)),
                "count": 0,
                "total_reward": 0.0,
                "avg_reward": 0.0,
            }
        return self.rule_stats[rule_id]

    def _drop_stats_wrong_dim(self) -> None:
        """Remove persisted rules whose A/b size does not match current context_dim."""
        stale = []
        for rid, st in self.rule_stats.items():
            if rid.startswith("_"):
                continue
            A = st.get("A")
            b = st.get("b")
            ok_a = False
            ok_b = False
            try:
                if isinstance(A, str):
                    raw = base64.standard_b64decode(A.encode("ascii"))
                    ok_a = len(raw) == self.context_dim * self.context_dim * 8
                elif isinstance(A, list):
                    ok_a = len(A) == self.context_dim and all(
                        len(row) == self.context_dim for row in A
                    )
            except Exception:
                ok_a = False
            try:
                if isinstance(b, str):
                    rawb = base64.standard_b64decode(b.encode("ascii"))
                    ok_b = len(rawb) == self.context_dim * 8
                elif isinstance(b, list):
                    ok_b = len(b) == self.context_dim
            except Exception:
                ok_b = False
            if not ok_a or not ok_b:
                stale.append(rid)
                continue
        for rid in stale:
            del self.rule_stats[rid]
        if stale:
            print(
                f"⚠️ Dropped {len(stale)} rule stats (dimension mismatch; expected {self.context_dim})"
            )
            self._save_stats()

    @staticmethod
    def _fingerprint_hash_features(sql: str) -> np.ndarray:
        """32-dim vector from SHA256 of project SQL fingerprint template."""
        tpl = _get_sql_fingerprint_generator().get_template(sql or "")
        if not tpl:
            return np.zeros(FINGERPRINT_HASH_DIM, dtype=np.float32)
        h = hashlib.sha256(tpl.encode("utf-8")).digest()
        return np.frombuffer(h, dtype=np.uint8).astype(np.float32) / 255.0

    @staticmethod
    def _plan_bottleneck_features(plan_text: str) -> np.ndarray:
        """
        Encode plan bottlenecks from 【算子名】 in the analyzer text only.
        If a 代价占比 line appears shortly after a bracket, weight by that share; else weight 1.0.
        Names are hashed into a fixed number of buckets (no curated operator list).
        """
        vec = np.zeros(PLAN_BOTTLENECK_DIM, dtype=np.float32)
        text = str(plan_text) if plan_text is not None else ""
        if not text.strip():
            return vec

        bracket_matches = list(_BRACKET_OP_RE.finditer(text))
        if not bracket_matches:
            return vec

        d = PLAN_BOTTLENECK_DIM
        for m in bracket_matches:
            label = m.group(1).strip()
            if not label:
                continue
            window = text[m.end() : m.end() + 500]
            pct_m = _PCT_AFTER_BRACKET_RE.search(window)
            if pct_m:
                try:
                    pct = float(pct_m.group(1))
                except ValueError:
                    pct = 100.0
                w = min(max(pct / 100.0, 0.0), 1.0)
            else:
                w = 1.0

            key = label.lower()
            digest = hashlib.sha256(key.encode("utf-8")).digest()
            i0 = int.from_bytes(digest[0:2], "big") % d
            i1 = int.from_bytes(digest[2:4], "big") % d
            half = 0.5 * w
            vec[i0] += half
            vec[i1] += half

        vec = np.minimum(vec, 1.0)
        return vec

    def extract_context(self, sql: str, explain_info: str, advice_groups: List[str]) -> np.ndarray:
        """
        Extract context feature vector from SQL fingerprint, plan bottlenecks, and advice groups.

        Features (DEFAULT_CONTEXT_DIM = 64):
        - 32 dims: SHA256 bytes of SQLFingerprintGenerator template (project SQL fingerprint)
        - 24 dims: 【】中的算子名（稳定哈希分桶），邻近有代价占比则按占比加权
        - 8 dims: advice group indicators

        Returns:
            L2-normalized feature vector of length context_dim
        """
        fp = self._fingerprint_hash_features(sql)
        plan = self._plan_bottleneck_features(explain_info)

        all_groups = [
            "子查询优化",
            "连接优化",
            "谓词简化",
            "常量折叠",
            "聚合优化",
            "投影优化",
            "排序优化",
            "集合优化",
        ]
        advice = np.array(
            [1.0 if g in advice_groups else 0.0 for g in all_groups],
            dtype=np.float32,
        )

        features = np.concatenate([fp, plan, advice], axis=0)
        if features.shape[0] != self.context_dim:
            raise ValueError(
                f"Context length {features.shape[0]} != context_dim {self.context_dim}"
            )

        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm

        return features.astype(np.float32)
    
    def _predict_reward(self, rule_id: str, context: np.ndarray) -> Tuple[float, np.ndarray]:
        stats = self._get_or_init_rule(rule_id)
        A = self._A_to_numpy(stats)
        b = self._b_to_numpy(stats)
        try:
            A_inv = np.linalg.inv(A)
            theta = A_inv @ b
        except np.linalg.LinAlgError:
            A_inv = np.linalg.pinv(A)
            theta = A_inv @ b
        predicted_reward = float(theta @ context)
        return predicted_reward, A_inv

    def calculate_ucb_score(self, rule_id: str, context: np.ndarray) -> float:
        """
        Calculate UCB (Upper Confidence Bound) score using LinUCB formula.
        
        Formula: score = θ^T · x + α · √(x^T · A^(-1) · x)
        
        where:
        - θ = A^(-1) · b (estimated reward parameter)
        - x = context vector
        - α = exploration parameter
        - First term = predicted reward (exploitation)
        - Second term = uncertainty bonus (exploration)
        
        Args:
            rule_id: Rule identifier
            context: Context feature vector
            
        Returns:
            UCB score (higher = better candidate for selection)
        """
        stats = self._get_or_init_rule(rule_id)
        count = stats["count"]
        predicted_reward, A_inv = self._predict_reward(rule_id, context)
        
        # Uncertainty bonus (exploration term)
        uncertainty = np.sqrt(context @ A_inv @ context)
        exploration_bonus = self.alpha * uncertainty
        
        # Small bonus for rarely tried rules
        exploration_bonus += 0.1 / (1 + count)
        
        return float(predicted_reward + exploration_bonus)

    def score_rules(
        self,
        rule_library: Dict[str, Dict[str, str]],
        context: np.ndarray,
    ) -> List[Tuple[str, str, float, str]]:
        """
        Calculate UCB scores for all rules in the library.
        
        Args:
            rule_library: {group: {rule_id: description}}
            context: Context feature vector
            
        Returns:
            List of (group, rule_id, ucb_score, description) sorted by score descending
        """
        scored_rules = []
        for group, rules in rule_library.items():
            for rule_id, rule_desc in rules.items():
                score = self.calculate_ucb_score(rule_id, context)
                scored_rules.append((group, rule_id, score, rule_desc))
        
        # Sort by UCB score descending
        scored_rules.sort(key=lambda x: x[2], reverse=True)
        return scored_rules
    
    def update(self, rule_id: str, context: np.ndarray, reward: float):
        """
        Update single rule statistics after observing a reward.
        
        LinUCB update rules:
        - A += x · x^T  (outer product)
        - b += reward · x
        
        Args:
            rule_id: Rule that was applied
            context: Context vector when rule was selected
            reward: Observed reward (typically in [-1, 1])
        """
        stats = self._get_or_init_rule(rule_id)

        x1 = np.asarray(context, dtype=np.float64).reshape(-1)
        A = self._A_to_numpy(stats)
        A += x1.reshape(-1, 1) @ x1.reshape(1, -1)
        stats["A"] = self._encode_A_matrix(A)

        b = self._b_to_numpy(stats)
        b += reward * x1
        stats["b"] = self._encode_b_vector(b)
        
        # Update counters
        stats["count"] += 1
        stats["total_reward"] += reward
        stats["avg_reward"] = stats["total_reward"] / stats["count"]
        
        self._save_stats()

    @staticmethod
    def compute_sequence_reward(original_cost: float, rewritten_cost: float) -> float:
        """
        Convert cost reduction to a bounded sequence reward.

        reward = clip((original_cost - rewritten_cost) / original_cost, -1.0, 1.0)
        """
        if original_cost <= 0:
            return 0.0
        raw = (original_cost - rewritten_cost) / original_cost
        return float(max(-1.0, min(1.0, raw)))

    def update_with_sequence_reward(
        self,
        rule_sequence: List[str],
        context: np.ndarray,
        sequence_reward: float,
        rule_weights: Optional[Dict[str, float]] = None,
        position_decay: float = 0.90,
    ) -> Dict[str, float]:
        """
        Update statistics from sequence-level reward without any LLM.

        The full sequence gets a scalar reward from cost reduction.
        Reward distribution strategy:
        1) If rule_weights provided, use normalized positive weights for matching rules.
        2) Otherwise, use geometric position weights:
           w_i = position_decay ** i
           r_i = sequence_reward * w_i / sum_j w_j

        Args:
            rule_sequence: Ordered applied rule IDs.
            context: Context vector used for LinUCB update.
            sequence_reward: Sequence-level reward in [-1, 1].
            rule_weights: Optional per-rule positive weights from LLM effect scores.
            position_decay: Later rules get smaller weights when < 1.

        Returns:
            Dict[rule_id, allocated_reward]
        """
        if not rule_sequence:
            return {}
        if position_decay <= 0:
            position_decay = 1.0

        n = len(rule_sequence)
        if isinstance(rule_weights, dict) and rule_weights:
            raw = np.array(
                [max(0.0, float(rule_weights.get(rid, 0.0))) for rid in rule_sequence],
                dtype=np.float64,
            )
            if float(raw.sum()) > 0:
                weights = raw
            else:
                weights = np.array([position_decay ** i for i in range(n)], dtype=np.float64)
        else:
            weights = np.array([position_decay ** i for i in range(n)], dtype=np.float64)
        weight_sum = float(weights.sum()) if float(weights.sum()) > 0 else 1.0

        allocated: Dict[str, float] = {}
        for i, rule_id in enumerate(rule_sequence):
            reward_i = float(sequence_reward * (weights[i] / weight_sum))
            allocated[rule_id] = reward_i
            self.update(rule_id, context, reward_i)
        return allocated
    
    def get_rule_stats(self, rule_id: str) -> Optional[Dict]:
        """Get statistics for a specific rule"""
        return self.rule_stats.get(rule_id)
    
    def get_rule_stats_summary(self) -> Dict[str, Dict]:
        """Get summary of all rule statistics"""
        summary = {}
        for rule_id, stats in self.rule_stats.items():
            if not rule_id.startswith("_"):  # Skip internal keys
                summary[rule_id] = {
                    "count": stats["count"],
                    "avg_reward": stats["avg_reward"],
                    "total_reward": stats["total_reward"]
                }
        return summary


# Singleton instance for global access
_bandit_instance: Optional[RuleBandit] = None


def get_rule_bandit() -> RuleBandit:
    """Get or create global RuleBandit instance"""
    global _bandit_instance
    if _bandit_instance is None:
        _bandit_instance = RuleBandit()
    return _bandit_instance
