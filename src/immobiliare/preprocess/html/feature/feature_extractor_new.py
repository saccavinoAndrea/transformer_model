import math
import re
import statistics
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Any

from bs4 import Tag

from immobiliare.core_interfaces.feature.ifeature_extractor import IFeatureExtractor
from immobiliare.domain import Token
from immobiliare.utils.logging.logger_factory import LoggerFactory


class FeatureExtractorOptimized(IFeatureExtractor):

    def __init__(self):
        self.logger = LoggerFactory.get_logger("feature_extractor")
        self.display_map = {"none": 0, "block": 1, "inline": 2}
        self.token_tuples = None

        # prezzo, con migliaia separatori e decimali, seguito da spazio opzionale e simbolo €
        self.price_re = re.compile(r"\b\d{1,3}(?:[.,]\d{3})*(?:[.,]\d+)?\s*€\b", re.I)

        # numero (intero o decimale) seguito da 'locali' o 'locale'
        self.locali_re = re.compile(r"\b\d+(?:[.,]\d+)?\s*locali?\b", re.I)

        self.title_terms = ["via", "trilocale", "bilocale", "appartamento"]


    def extract_features(self, token: str, tag: Tag, position: int, *, total_tokens: int, max_depth_in_doc: int,
                         all_tokens: List[str], total_text_len: int = None, page_token_counts: dict = None,
                         parent_tokens_positions: List[int] = None) -> Dict[str, float]:
        pass


    def process_all(self, token_tuples: List[Tuple[str, Tag]]) -> List[Token]:
        N = len(token_tuples)
        lengths = [len(text) for text, _ in token_tuples]
        window_size = 2
        texts = [t for t, _ in token_tuples]
        tags = [tag for _, tag in token_tuples]

        window_mean_length = []
        window_std_length = []

        for i in range(N):
            lo = max(0, i - window_size)
            hi = min(N, i + window_size + 1)
            window = lengths[lo:hi]

            mean_len = statistics.mean(window) if window else 0.0
            std_len = statistics.pstdev(window) if len(window) > 1 else 0.0

            window_mean_length.append(mean_len)
            window_std_length.append(std_len)


        # Costruzione dizionario dei token diretti per ogni tag
        self_len: Dict[Tag, int] = {tag: 0 for tag in tags}
        for _, tag in token_tuples:
            self_len[tag] += 1
        total_text_len = sum(len(t) for t in texts)
        freq = Counter(texts)

        # 1) Mappe parent/children
        parent_map: Dict[Tag, Tag] = {}
        children_map: Dict[Tag, List[Tag]] = defaultdict(list)
        for _, tag in token_tuples:
            p = tag.parent if isinstance(tag.parent, Tag) else None
            parent_map[tag] = p
            if p:
                children_map[p].append(tag)

        # 2) Profondità
        depth: Dict[Tag, int] = {}
        for tag in tags:
            depth[tag] = 0
        for tag in tags:
            p = parent_map[tag]
            depth[tag] = depth.get(p, 0) + 1

        max_depth = max(depth.values()) if depth else 0

        # 3) Siblings
        siblings_map: Dict[Tag, List[Tag]] = {}
        tag_sib_idx: Dict[Tag, int] = {}
        for tag in tags:
            p = parent_map[tag]
            sibs = children_map.get(p, [tag]) if p else [tag]
            siblings_map[tag] = sibs
            tag_sib_idx[tag] = sibs.index(tag)

        # 4) Precompute descendants (DFS)
        descendants_map: Dict[Tag, List[Tag]] = {}
        for tag in tags:
            stack = list(children_map.get(tag, []))
            descs = []
            while stack:
                node = stack.pop()
                descs.append(node)
                stack.extend(children_map.get(node, []))
            descendants_map[tag] = descs

        # 5) Block-level stats placeholders
        #     text_token_density, tag_density, link_density, img_density, texts list, desc_count
        block_stats: Dict[Tag, Dict[str, Any]] = {}
        for tag in tags:
            descs = descendants_map[tag]
            desc_count = len(descs)
            texts_in_block = [t for t, tg in token_tuples if tg in descs]
            lengths = [len(tx) for tx in texts_in_block]
            total_len = sum(lengths)
            link_count = sum(1 for d in descs if d.name == "a")
            img_count = sum(1 for d in descs if d.name == "img")
            block_stats[tag] = {
                "text_token_density": total_len / (desc_count or 1),
                "tag_density": desc_count / (len(set(d.name for d in descs)) or 1),
                "link_density": link_count / (desc_count or 1),
                "img_density": img_count / (desc_count or 1),
                "texts": texts_in_block,
                "desc_count": desc_count
            }

        # --- Part 5: precompute post‑processing metrics (O(N)) ---
        n = len(token_tuples)
        lengths = [len(text) for text, _ in token_tuples]

        # prepariamo array vuoti
        is_local_max = [0.0] * n
        peak_prominence = [0.0] * n
        gradient_sign = [0.0] * n
        rank_in_block = [0.0] * n
        z_to_block_med = [0.0] * n
        dist_from_price = [0.0] * n
        between_price_locali = [0.0] * n

        # individua indici di prezzo e locali
        price_idxs = [i for i, (text, _) in enumerate(token_tuples) if self.price_re.search(text)]
        locali_idxs = [i for i, (text, _) in enumerate(token_tuples) if self.locali_re.search(text)]
        last_price = price_idxs[-1] if price_idxs else None
        min_loc = min(locali_idxs) if locali_idxs else None
        max_loc = max(locali_idxs) if locali_idxs else None

        # mediana per blocchi
        block_meds = {}
        per_block_lengths = defaultdict(list)
        for i, (text, tag) in enumerate(token_tuples):
            per_block_lengths[tag.parent].append(len(text))
        for blk, lst in per_block_lengths.items():
            block_meds[blk] = statistics.median(lst)

        # calcolo passata unica
        for i in range(n):
            prev_l = lengths[i - 1] if i > 0 else lengths[i]
            next_l = lengths[i + 1] if i < n - 1 else lengths[i]

            # local max
            if lengths[i] > prev_l and lengths[i] > next_l:
                is_local_max[i] = 1.0
            # prominence
            denom = max(prev_l, next_l) or 1.0
            peak_prominence[i] = max(0.0, (lengths[i] - denom) / denom)
            # gradient sign
            gradient_sign[i] = float((next_l > prev_l) - (next_l < prev_l))

            # rank in block
            blk = token_tuples[i][1].parent
            block_list = per_block_lengths[blk]
            # ordinati desc
            sorted_blk = sorted(block_list, reverse=True)
            rank_in_block[i] = float(sorted_blk.index(lengths[i]) + 1)

            # z to block median
            med = block_meds.get(blk, lengths[i])
            stdev = statistics.pstdev(block_list) if len(block_list) > 1 else 1.0
            # evita zero-division
            den = stdev if stdev > 0 else 1.0
            z_to_block_med[i] = (lengths[i] - med) / den

            # dist_from_last_price
            if last_price is not None:
                dist_from_price[i] = float(abs(i - last_price))

            # between price and locali
            if last_price is not None and min_loc is not None and max_loc is not None:
                between_price_locali[i] = 1.0 if (min_loc < i < max_loc) else 0.0

        # --- Part 5b: altri pre‑calcoli opzionali ---

        # 1. Estrarre font-size e margin-top dallo style
        font_size_vals = []
        margin_top_vals = []
        display_code_vals = []
        for text, tag in token_tuples:
            style = tag.attrs.get("style", "")
            # font-size: assume formato “12px”, “1.2rem” ecc.
            m = re.search(r"font-size\s*:\s*([\d\.]+)", style)
            font_size_vals.append(float(m.group(1)) if m else 0.0)
            # margin-top
            m2 = re.search(r"margin-top\s*:\s*([\d\.]+)", style)
            margin_top_vals.append(float(m2.group(1)) if m2 else 0.0)
            # display code
            disp = ""
            m3 = re.search(r"display\s*:\s*([^;]+)", style)
            if m3:
                disp = m3.group(1).strip().lower()
            display_code_vals.append({"none":0,"block":1,"inline":2}.get(disp, 3))

        # 2. Sliding‑window stats on word lengths
        #    tracciamo per ogni token la media e la std dei word-length nel window [i-2..i+2]
        word_counts = [len(t.split()) for t,_ in token_tuples]
        window_mean_avg_word_length = []
        window_std_avg_word_length  = []
        for i, (text, _) in enumerate(token_tuples):
            lo = max(0, i - window_size)
            hi = min(N, i + window_size + 1)
            # estrai avg_word_length per ogni pos in window
            avgs = []
            for j in range(lo, hi):
                ws = token_tuples[j][0].split()
                avgs.append(sum(len(w) for w in ws) / (len(ws) or 1))
            window_mean_avg_word_length.append(statistics.mean(avgs))
            window_std_avg_word_length.append(statistics.pstdev(avgs) if len(avgs)>1 else 0.0)

        # 3. css_title_score: trova la prima classe contenente “title”
        css_title_vals = []
        for _, tag in token_tuples:
            classes = tag.attrs.get("class", [])
            idx = next((i for i,c in enumerate(classes) if "title" in c.lower()), None)
            css_title_vals.append(1.0/(idx+1) if idx is not None else 0.0)

        # 4. cap_run_norm: lunghezza massima sequenza di maiuscole / length
        cap_run_norm_vals = []
        for text, _ in token_tuples:
            runs = re.findall(r"[A-ZÀ-ÖØ-Ý]+", text)
            max_run = max((len(r) for r in runs), default=0)
            cap_run_norm_vals.append(max_run / (len(text) or 1))


        # 6) Ancestor density: average and variance of text_token_density
        anc_avg: Dict[Tag, float] = {}
        anc_var: Dict[Tag, float] = {}
        for tag in tags:
            vals = []
            p = parent_map.get(tag)
            # risaliamo finché p è un tag valido in block_stats
            while p and p in block_stats:
                vals.append(block_stats[p]["text_token_density"])
                p = parent_map.get(p)
            if vals:
                anc_avg[tag] = statistics.mean(vals)
                anc_var[tag] = statistics.pvariance(vals) if len(vals) > 1 else 0.0
            else:
                anc_avg[tag] = 0.0
                anc_var[tag] = 0.0

        # 7) Global distributions for z-scores
        lengths = [len(t) for t in texts]
        mean_len = statistics.mean(lengths) if lengths else 0.0
        std_len  = statistics.pstdev(lengths) if len(lengths) > 1 else 1.0

        n_words_list = [len(t.split()) for t in texts]
        mean_words = statistics.mean(n_words_list) if n_words_list else 0.0
        std_words  = statistics.pstdev(n_words_list) if len(n_words_list) > 1 else 1.0

        # 8) Sliding window stats on token lengths
        window_size = 2
        window_stats: Dict[int, Dict[str, float]] = {}
        for i in range(N):
            lo = max(0, i - window_size)
            hi = min(N, i + window_size + 1)
            seg = lengths[lo:hi]
            window_stats[i] = {
                "mean_len": sum(seg) / len(seg),
                "std_len": statistics.pstdev(seg) if len(seg) > 1 else 0.0
            }

        # --- Part X: Precompute arrays per tutte le feature vettoriali (O(N)) ---

        # 1) Sibling text density (densità media di text_len tra fratelli)
        sibling_text_density_vals = []
        for text, tag in token_tuples:
            sibs = siblings_map[tag]
            lens = [len(s.get_text(strip=True)) for s in sibs]
            sibling_text_density_vals.append(statistics.mean(lens) if lens else 0.0)

        # 2) Block avg/std word length & avg sentence count
        block_avg_word_length_vals = []
        block_std_word_length_vals = []
        block_avg_sentence_count_vals = []
        block_special_char_density_vals = []
        block_digit_density_vals = []
        token_dist_to_block_mid_vals = []
        block_depth_wtd_token_den_vals = []
        siblings_tag_density_vals = []
        local_subtree_height_vals = []
        parent_child_ratio_vals = []
        local_token_rep_density_vals = []
        block_avg_token_distance_vals = []
        normalized_block_area_vals = []

        # Pre‑restituisci una lista di token indices per ogni block
        per_block_idxs = defaultdict(list)
        for i, (_, tag) in enumerate(token_tuples):
            per_block_idxs[tag.parent].append(i)

        for i, (text, tag) in enumerate(token_tuples):
            # raccogli i text dei discendenti
            descs = descendants_map[tag]
            desc_idxs = [j for j, (_, tg) in enumerate(token_tuples) if tg in descs]

            # 2a) block_avg_word_length & std
            words = []
            sents = []
            specials = 0
            digits = 0
            for j in desc_idxs:
                t = token_tuples[j][0]
                ws = t.split()
                words += ws
                sents.append(len(re.findall(r'[.!?]+', t)))
                specials += sum(not c.isalnum() and not c.isspace() for c in t)
                digits += sum(c.isdigit() for c in t)
            block_avg_word_length_vals.append(
                sum(len(w) for w in words) / len(words) if words else 0.0
            )
            block_std_word_length_vals.append(
                statistics.pstdev([len(w) for w in words]) if len(words) > 1 else 0.0
            )
            block_avg_sentence_count_vals.append(
                sum(sents) / len(sents) if sents else 0.0
            )
            block_special_char_density_vals.append(
                specials / (sum(len(t) for t in (token_tuples[j][0] for j in desc_idxs)) or 1)
            )
            block_digit_density_vals.append(
                digits / (sum(len(t) for t in (token_tuples[j][0] for j in desc_idxs)) or 1)
            )

            # 2b) token_dist_to_block_mid
            idxs = per_block_idxs[tag.parent]
            mid = (len(idxs) - 1) / 2
            token_dist_to_block_mid_vals.append(
                abs(idxs.index(i) - mid) / (len(idxs) or 1)
            )

            # 2c) block_depth_wtd_token_den
            block_depth_wtd_token_den_vals.append(
                sum(
                    self_len[c] / (depth[c] or 1)
                    for c in descendants_map[tag]
                ) / (len(descendants_map[tag]) or 1)
            )

            # 2d) siblings_tag_density (densià di tag figli dei fratelli)
            sibs = children_map.get(parent_map[tag], [])
            counts = [len(children_map.get(s, [])) for s in sibs if s is not tag]
            siblings_tag_density_vals.append(
                statistics.mean(counts) if counts else 0.0
            )

            # 2e) local_subtree_height (altezza massima discendenza)
            def height(num):
                return 1 + max((height(c) for c in children_map.get(num, [])), default=0)

            local_subtree_height_vals.append(float(height(tag)))

            # 2f) parent_child_ratio
            pc = len(children_map.get(parent_map[tag], [])) or 1
            parent_child_ratio_vals.append(float(len(children_map.get(tag, []))) / pc)

            # 2g) local_token_rep_density (ripetizioni token nel blocco)
            reps = Counter(token_tuples[j][0] for j in desc_idxs)
            local_token_rep_density_vals.append(
                sum(reps[t] for t in reps) / len(desc_idxs) if desc_idxs else 0.0
            )

            # 2h) block_avg_token_distance (distanza media tra token nel blocco)
            dists = []
            for a in desc_idxs:
                for b in desc_idxs:
                    dists.append(abs(a - b))
            block_avg_token_distance_vals.append(
                statistics.mean(dists) if dists else 0.0
            )

            # 2i) normalized_block_area: (depth_ratio * relative_position)
            normalized_block_area_vals.append(
                (float(depth[tag] + 1) / max_depth) * (float(i) / (N - 1) if N > 1 else 0.0)
            )

        descendants_with_digits: Dict[Tag, int] = {}
        descendants_with_links: Dict[Tag, int] = {}
        descendants_with_buttons: Dict[Tag, int] = {}

        for tag in tags:
            descs = descendants_map[tag]
            descendants_with_digits[tag] = sum(1 for d in descs if any(c.isdigit() for c in (d.get_text() or "")))
            descendants_with_links[tag] = sum(1 for d in descs if d.name == "a")
            descendants_with_buttons[tag] = sum(1 for d in descs if d.name == "button")

        # 9) Costruzione dei Token
        result: List[Token] = []
        for idx, (text, tag) in enumerate(token_tuples):
            words = text.split()
            nw = len(words)

            feats = {
                # Input diretto
                "text": text,
                "tag": tag,

                "window_mean_length": float(window_mean_length[idx]),
                "window_std_length": float(window_std_length[idx]),
                "window_mean_avg_word_length": float(window_mean_avg_word_length[idx]),
                "window_std_avg_word_length": float(window_std_avg_word_length[idx]),

                # A. Testo base
                "length": float(lengths[idx]),
                "n_words": float(nw),
                "avg_word_length": float(sum(len(w) for w in words) / nw) if nw else 0.0,
                "n_digits": float(sum(c.isdigit() for c in text)),
                "n_uppercase": float(sum(c.isupper() for c in text)),
                "n_periods": float(text.count('.')),
                "n_special_chars": float(sum(not c.isalnum() and not c.isspace() for c in text)),
                "n_sentences": float(len(re.findall(r'[.!?]+', text))),
                "is_numeric_token": float(text.isdigit()),
                "is_mixed_alphanum": float(any(c.isalpha() for c in text) and any(c.isdigit() for c in text)),

                # B. DOM
                "n_parents": float(depth[tag]),
                "n_ancestors_with_class": float(sum(1 for p in tag.parents if p.get("class"))),
                "n_ancestors_with_id": float(sum(1 for p in tag.parents if p.get("id"))),
                "n_ancestors_div": float(sum(1 for p in tag.parents if p.name == "div")),
                "n_ancestors_section": float(
                    sum(1 for p in tag.parents if p.name in {"section", "article", "nav", "aside"})),
                "has_grandparent": float(len(list(tag.parents)) > 1),
                "n_children": float(len(children_map.get(tag, []))),
                "n_descendants": float(len(descendants_map.get(tag, []))),
                "max_child_depth": float(max((depth.get(c, 0) for c in children_map.get(tag, [])), default=0)),
                "has_only_text_children": float(all(not isinstance(c, Tag) for c in tag.contents)),
                "n_siblings": float(len(siblings_map[tag]) - 1),
                "tag_index_among_siblings": float(tag_sib_idx[tag]),
                "n_siblings_with_text": float(sum(1 for s in siblings_map[tag] if s.get_text(strip=True))),
                "avg_sibling_text_length": float(
                    statistics.mean([len(s.get_text(strip=True)) for s in siblings_map[tag]])) if len(
                    siblings_map[tag]) > 1 else 0.0,
                "n_unique_tags_among_siblings": float(len(set(s.name for s in siblings_map[tag]))),

                # C. Posizione
                "relative_position": float(idx) / (N - 1) if N > 1 else 0.0,
                "relative_depth_ratio": float(depth[tag] + 1) / max_depth if max_depth else 0.0,

                # D. Statistiche testuali
                "text_density": float(block_stats[tag]["text_token_density"]),
                "normalized_token_length": float(lengths[idx]) / (total_text_len / N) if N else 0.0,
                "percentile_token_length": float(sum(1 for t in texts if len(t) <= lengths[idx])) / N,
                "token_distance_to_next": float(len(texts[idx + 1])) if idx < N - 1 else 0.0,
                "token_distance_to_prev": float(len(texts[idx - 1])) if idx > 0 else 0.0,

                # E. Stile inline
                "has_style_attribute": float("style" in tag.attrs),
                "n_style_properties": float(len(tag.attrs.get("style", "").split(";"))) if tag.attrs.get(
                    "style") else 0.0,
                "font_size_extracted": font_size_vals[idx],
                "margin_top_extracted": margin_top_vals[idx],
                "display_type_code": display_code_vals[idx],

                # F. Contesto locale
                "parent_tag_token_count": float(block_stats.get(parent_map.get(tag), {}).get("desc_count", 0.0)),
                "n_child_tokens_with_digits": float(
                    sum(any(ch.isdigit() for ch in c.get_text()) for c in children_map.get(tag, []))),
                "n_descendants_with_digits": float(descendants_with_digits[tag]),
                "n_descendants_with_links": float(descendants_with_links[tag]),
                "n_descendants_with_buttons": float(descendants_with_buttons[tag]),

                # G. Pagina intera
                "token_repeated_in_page": float(freq[text]),

                # H. Class/id/path
                "n_classes": float(len(tag.attrs.get("class", []))),
                "class_name_length_avg": float(
                    statistics.mean([len(c) for c in tag.attrs.get("class", [])])) if tag.attrs.get("class") else 0.0,
                "id_name_length": float(len(tag.attrs.get("id", ""))),
                "dom_path_length": float(len(list(tag.parents))),

                # T. Link e titolo
                "is_link": float(tag.name == "a"),
                "link_href_depth": float(tag.attrs.get("href", "").count("/")) if tag.name == "a" else 0.0,
                "css_title_score": css_title_vals[idx],
                "capital_ratio": float(
                    sum(1 for c in text if c.isupper()) / (sum(1 for c in text if c.isalpha()) or 1)),
                "keyword_score_title": float(
                    sum(1 for w in words if w.lower() in self.title_terms) / nw) if nw else 0.0,

                # I. Entropia e z‑score
                "char_entropy": float(
                    -sum(p * math.log2(p) for p in [(cnt / lengths[idx]) for cnt in Counter(text).values()] if p > 0)),
                "length_zscore": float((lengths[idx] - mean_len) / std_len),
                "n_words_zscore": float((nw - mean_words) / std_words),
                "tf_log": float(math.log2(freq[text] + 1)),
                "distinct_ratio": float(len(set(text)) / lengths[idx]) if lengths[idx] else 0.0,
                "uniq_ratio": float(len(set(words)) / len(set(w for t in texts for w in t.split()))) if texts else 0.0,

                # J. Avanzate su caratteri
                "vowel_ratio": float(sum(1 for c in text.lower() if c in "aeiouàèéìòù") / lengths[idx]) if lengths[
                    idx] else 0.0,
                "cv_balance": float((sum(1 for c in text.lower() if c.isalpha() and c not in "aeiouàèéìòù") - sum(
                    1 for c in text.lower() if c in "aeiouàèéìòù")) / lengths[idx]) if lengths[idx] else 0.0,
                "max_digit_run": float(max((len(m) for m in re.findall(r"\d+", text)), default=0)),
                "uniq_bigram_ratio": float(
                    len(set(text[i:i + 2] for i in range(lengths[idx] - 1))) / max(lengths[idx] - 1, 1)),
                "neigh_len_diff": float(((abs(lengths[idx] - len(texts[idx - 1])) if idx > 0 else 0) + (
                    abs(lengths[idx] - len(texts[idx + 1])) if idx < N - 1 else 0)) / 2),
                "cap_run_norm": cap_run_norm_vals[idx],

                # Density-level features
                "block_text_token_density": float(block_stats[tag]["text_token_density"]),
                "block_tag_density": float(block_stats[tag]["tag_density"]),
                "ancestors_avg_tag_density": float(anc_avg[tag]),
                "descendants_text_density": float(block_stats[tag]["text_token_density"]),
                "sibling_text_density": sibling_text_density_vals[idx],
                "text_ratio_in_block": float(lengths[idx] / (block_stats[tag]["text_token_density"] or 1)),
                "tag_ratio_in_block": float(1 / (block_stats[tag]["desc_count"] or 1)),
                "block_avg_word_length": block_avg_word_length_vals[idx],
                "block_std_word_length": block_std_word_length_vals[idx],
                "block_avg_sentence_count": block_avg_sentence_count_vals[idx],
                "ancestors_text_density_variance": float(anc_var[tag]),
                "descendants_link_density": float(block_stats[tag]["link_density"]),
                "descendants_img_density": float(block_stats[tag]["img_density"]),
                "block_special_char_density": block_special_char_density_vals[idx],
                "block_digit_density": block_digit_density_vals[idx],
                "token_dist_to_block_mid": token_dist_to_block_mid_vals[idx],
                "block_token_count": float(block_stats[tag]["desc_count"]),
                "block_depth_wtd_token_den": block_depth_wtd_token_den_vals[idx],
                "siblings_tag_density": siblings_tag_density_vals[idx],
                "local_subtree_height": local_subtree_height_vals[idx],
                "parent_child_ratio": parent_child_ratio_vals[idx],
                "block_text_length_ratio": float(block_stats[tag]["text_token_density"] / total_text_len),
                "local_token_rep_density": local_token_rep_density_vals[idx],
                "block_avg_token_distance": block_avg_token_distance_vals[idx],
                "normalized_block_area": normalized_block_area_vals[idx],

                # Post‑processing metrics
                "length_is_local_max": is_local_max[idx],
                "length_peak_prominence": peak_prominence[idx],
                "length_gradient_sign": gradient_sign[idx],
                "length_rank_in_block": rank_in_block[idx],
                "length_z_to_block_median": z_to_block_med[idx],
                "dist_from_last_price_token": dist_from_price[idx],
                "is_between_price_and_locali": between_price_locali[idx],

                # Meta
                "meta": {}
            }

            result.append(Token(**feats))

        return result

