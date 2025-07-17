# src/immobiliare/preprocess/feature/density_features.py

import math
import statistics
from bs4 import Tag
from typing import List, Optional


class DensityFeatures:
    # Utility
    @staticmethod
    def safe_get_text(tag: Optional[Tag]) -> str:
        try:
            return tag.get_text(strip=True)
        except Exception:
            return ""

    # Helper
    @staticmethod
    def _direct_text_tokens(tag: Tag) -> List[str]:
        parent = tag.parent
        if not parent or not hasattr(parent, "stripped_strings"):
            return []
        return [s for s in parent.stripped_strings]

    @staticmethod
    def _direct_child_tags(tag: Tag) -> List[Tag]:
        parent = tag.parent
        if not parent:
            return []
        return [c for c in parent.find_all(recursive=False) if isinstance(c, Tag)]

    @staticmethod
    def _ancestors(tag: Tag) -> List[Tag]:
        return [p for p in tag.parents if isinstance(p, Tag)]

    @staticmethod
    def _descendants(tag: Tag) -> List[Tag]:
        return [d for d in tag.descendants if isinstance(d, Tag)]

    # Features
    @staticmethod
    def block_text_token_density(tag: Tag) -> float:
        toks = DensityFeatures._direct_text_tokens(tag)
        n_ch = len(DensityFeatures._direct_child_tags(tag))
        return len(toks) / n_ch if n_ch else 0.0

    @staticmethod
    def block_tag_density(tag: Tag) -> float:
        n_ch = len(DensityFeatures._direct_child_tags(tag))
        depth = len(DensityFeatures._ancestors(tag.parent)) if tag.parent else 1
        return n_ch / depth if depth else 0.0

    @staticmethod
    def ancestors_avg_tag_density(tag: Tag) -> float:
        anc = DensityFeatures._ancestors(tag)
        ratios = []
        for a in anc:
            if not a:
                continue
            ch = [c for c in a.find_all(recursive=False) if isinstance(c, Tag)]
            depth = len([p for p in a.parents if isinstance(p, Tag)]) or 1
            ratios.append(len(ch) / depth)
        return statistics.mean(ratios) if ratios else 0.0

    @staticmethod
    def descendants_text_density(tag: Tag) -> float:
        desc = DensityFeatures._descendants(tag)
        texts = [s for s in tag.descendants if not isinstance(s, Tag) and s.strip()]
        return len(texts) / len(desc) if desc else 0.0

    @staticmethod
    def sibling_text_density(tag: Tag) -> float:
        if not tag.parent:
            return 0.0
        sibs = [c for c in tag.parent.children if isinstance(c, Tag) and c is not tag]
        lengths = [len(DensityFeatures.safe_get_text(c)) for c in sibs]
        return statistics.mean(lengths) if lengths else 0.0

    @staticmethod
    def text_ratio_in_block(tag: Tag) -> float:
        parent = tag.parent
        gp = parent.parent if parent else None
        if not parent or not gp:
            return 0.0
        return len(DensityFeatures.safe_get_text(parent)) / len(DensityFeatures.safe_get_text(gp) or " ")

    @staticmethod
    def tag_ratio_in_block(tag: Tag) -> float:
        parent = tag.parent
        gp = parent.parent if parent else None
        if not parent or not gp:
            return 0.0
        return len(parent.find_all()) / len(gp.find_all() or [])

    @staticmethod
    def block_avg_word_length(tag: Tag) -> float:
        words = DensityFeatures.safe_get_text(tag.parent).split()
        return statistics.mean([len(w) for w in words]) if words else 0.0

    @staticmethod
    def block_std_word_length(tag: Tag) -> float:
        words = DensityFeatures.safe_get_text(tag.parent).split()
        return statistics.pstdev([len(w) for w in words]) if len(words) > 1 else 0.0

    @staticmethod
    def block_avg_sentence_count(tag: Tag) -> float:
        text = DensityFeatures.safe_get_text(tag.parent)
        toks = text.split()
        periods = text.count('.')
        return periods / len(toks) if toks else 0.0

    @staticmethod
    def ancestors_text_density_variance(tag: Tag) -> float:
        anc = DensityFeatures._ancestors(tag)
        densities = []
        for a in anc:
            if not a or not hasattr(a, "stripped_strings"):
                continue
            texts = [s for s in a.stripped_strings]
            ch = [c for c in a.find_all(recursive=False) if isinstance(c, Tag)]
            densities.append(len(texts) / len(ch) if ch else 0.0)
        return statistics.pvariance(densities) if len(densities) > 1 else 0.0

    @staticmethod
    def descendants_link_density(tag: Tag) -> float:
        desc = DensityFeatures._descendants(tag)
        links = [d for d in desc if d.name == 'a']
        return len(links) / len(desc) if desc else 0.0

    @staticmethod
    def descendants_img_density(tag: Tag) -> float:
        desc = DensityFeatures._descendants(tag)
        imgs = [d for d in desc if d.name == 'img']
        return len(imgs) / len(desc) if desc else 0.0

    @staticmethod
    def block_specialchar_density(tag: Tag) -> float:
        text = DensityFeatures.safe_get_text(tag.parent)
        sc = sum(1 for c in text if not c.isalnum() and not c.isspace())
        return sc / len(text) if text else 0.0

    @staticmethod
    def block_digit_density(tag: Tag) -> float:
        text = DensityFeatures.safe_get_text(tag.parent)
        dg = sum(c.isdigit() for c in text)
        return dg / len(text) if text else 0.0

    @staticmethod
    def token_distance_to_block_midpoint(position: int, parent_token_count: int) -> float:
        mid = (parent_token_count - 1) / 2
        return abs(position - mid) / parent_token_count if parent_token_count else 0.0

    @staticmethod
    def block_token_count(tag: Tag) -> int:
        return len(DensityFeatures._direct_text_tokens(tag))

    @staticmethod
    def block_depth_weighted_token_density(tag: Tag) -> float:
        btc = DensityFeatures.block_token_count(tag)
        depth = len(DensityFeatures._ancestors(tag)) or 1
        return btc / depth

    @staticmethod
    def siblings_tag_density(tag: Tag) -> float:
        if not tag.parent:
            return 0.0
        sibs = [c for c in tag.parent.find_all(recursive=False) if isinstance(c, Tag) and c is not tag]
        counts = [len([cc for cc in s.find_all(recursive=False) if isinstance(cc, Tag)]) for s in sibs]
        return statistics.mean(counts) if counts else 0.0

    @staticmethod
    def local_subtree_height(tag: Tag) -> int:
        def max_depth(t):
            return 1 + max((max_depth(c) for c in t.find_all(recursive=False) if isinstance(c, Tag)), default=0)
        return max_depth(tag.parent) if tag.parent else 0

    @staticmethod
    def parent_child_ratio(tag: Tag) -> float:
        p = tag.parent
        if not p:
            return 0.0
        return len([c for c in p.find_all(recursive=False) if isinstance(c, Tag)]) / (len(DensityFeatures._ancestors(tag)) or 1)

    @staticmethod
    def block_text_length_ratio(tag: Tag, total_text_len: int) -> float:
        blen = len(DensityFeatures.safe_get_text(tag.parent))
        return blen / total_text_len if total_text_len else 0.0

    @staticmethod
    def local_token_repetition_density(tag: Tag, page_token_counts: dict) -> float:
        parent = tag.parent
        if not parent or not hasattr(parent, "stripped_strings"):
            return 0.0
        tokens = list(parent.stripped_strings)
        if not tokens:
            return 0.0
        reps = sum(page_token_counts.get(t, 0) for t in tokens)
        return reps / len(tokens)

    @staticmethod
    def block_avg_token_distance(raw_positions: List[int], position: int) -> float:
        if not raw_positions:
            return 0.0
        idx = raw_positions.index(position)
        prev_d = position - raw_positions[idx - 1] if idx > 0 else 0
        next_d = raw_positions[idx + 1] - position if idx < len(raw_positions) - 1 else 0
        return (prev_d + next_d) / 2

    @staticmethod
    def normalized_block_area(tag: Tag, max_depth: int, total_tokens: int, position: int) -> float:
        depth = len(DensityFeatures._ancestors(tag)) + 1
        rel_pos = position / (total_tokens - 1) if total_tokens > 1 else 0.0
        rel_depth = depth / max_depth if max_depth else 0.0
        return (rel_pos + rel_depth) / 2
