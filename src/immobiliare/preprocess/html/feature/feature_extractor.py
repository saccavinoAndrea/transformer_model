# src/immobiliare/preprocess/feature/feature_extractor.py
import math
import re
import statistics
from typing import List, Dict, Tuple, Any

from bs4 import Tag

from domain import Token
from immobiliare.core_interfaces.feature.ifeature_extractor import IFeatureExtractor
from immobiliare.utils.logging.logger_factory import LoggerFactory
from preprocess.html.feature.density_features import DensityFeatures


def _extract_style_value(style: str, prop: str) -> float:
    """
    Cerca in style inline la proprietà `prop` e restituisce il valore numerico (es. '12px' → 12).
    """
    m = re.search(rf"{prop}\s*:\s*([\d\.]+)", style)
    return float(m.group(1)) if m else 0.0


def _extract_style_token(style: str, prop: str) -> str:
    """
    Cerca in style inline la proprietà `prop` e restituisce il suo valore testuale ('block', 'inline', ecc.).
    """
    m = re.search(rf"{prop}\s*:\s*([^;]+)", style)
    return m.group(1).strip() if m else ""


def _get_tag_path(tag: Tag) -> str:
    """
    Costruisce un semplice 'XPath-like' basato sui tag, es. 'html > body > div > span'
    """
    path = []
    current = tag
    while current and isinstance(current, Tag):
        path.append(current.name)
        current = current.parent
    return " > ".join(reversed(path))


def _depth(tag: Tag) -> int:
    return len([p for p in tag.parents if isinstance(p, Tag)])


class FeatureExtractor(IFeatureExtractor):
    """
    Estrae ~100 feature per ogni token in base a testo, DOM, stile e contesto.
    """

    def __init__(self):
        self.logger = LoggerFactory.get_logger("feature_extractor")
        self.display_map = {"none": 0, "block": 1, "inline": 2}
        self.token_tuples = None


    def process_all(self, token_tuples: List[Tuple[str, Tag]]) -> List[Token]:

        self.token_tuples = token_tuples

        total = len(token_tuples)
        all_texts = [t[0] for t in token_tuples]
        max_depth = max(_depth(tag) for _, tag in token_tuples)

        # 1) Estrai tutte le feature raw in un list di dict
        raw_features: List[Dict[str, float]] = []
        tags: List[Tag] = []  # <‑‑ lista parallela di tag

        for idx, (text, tag) in enumerate(token_tuples):
            try:
                feats = self.extract_features(
                    text,
                    tag,
                    position=idx,
                    total_tokens=total,
                    max_depth_in_doc=max_depth,
                    all_tokens=all_texts
                )
                raw_features.append(feats)
                tags.append(tag)
            except Exception as e:
                self.logger.log_exception(f"Errore su token #{idx} ? {text}", e)

        # 2) Nuove feature “a barre” (titolo = picco fra barre corte)
        FeatureExtractor._add_local_length_peaks(raw_features)
        FeatureExtractor._add_block_based_peaks(raw_features)

        # 3) Aggiungi le window‑stats su 'length' e 'avg_word_length'
        self._compute_window_stats_for_feature(raw_features, "length", window_size=2)
        self._compute_window_stats_for_feature(raw_features, "avg_word_length", window_size=2)

        # 4) Costruisci i Token definitivi
        tokens: List[Token] = []
        for idx, feats in enumerate(raw_features):
            try:
                text_val = feats.pop("text")  # rimuovi il campo text
                token_obj = Token(
                    text=text_val,
                    tag=tags[idx],  # <‑‑ tag corretto
                    meta={},  # meta vuoto (lo riempirai dopo)
                    **feats
                )
                tokens.append(token_obj)
            except Exception as e:
                self.logger.log_exception(f"Errore su token #{idx}", e)

        #self.logger.log_info(f"Token processati: {len(tokens)}")
        return tokens

    def extract_features(
            self,
            token: str,
            tag: Tag,
            position: int,
            *,
            total_tokens: int,
            max_depth_in_doc: int,
            all_tokens: List[str],
            total_text_len: int = None,
            page_token_counts: dict = None,
            parent_tokens_positions: List[int] = None
    ) -> Dict[str, float]:

        try:

            """
            Calcola fino a 100 feature numeriche per un singolo token/tag.
            Alcune feature (relative_position, relative_depth_ratio,
            token_repeated_in_page, percentile_token_length, normalized_token_length)
            richiedono dati globali sul documento che possono essere passati qui.
            """
            text = token.strip()
            length = len(text)

            # — A. Testo base —
            words = text.split()
            n_words = len(words)
            avg_word_length = sum(len(w) for w in words) / n_words if n_words else 0

            n_digits = sum(c.isdigit() for c in text)
            n_upper = sum(c.isupper() for c in text)
            n_lower = sum(c.islower() for c in text)
            #n_punct = sum(c in string.punctuation for c in text)
            #n_commas = text.count(",")
            n_periods = text.count(".")
            # caratteri non alfanumerici e non spazio
            n_special = sum(1 for c in text if not c.isalnum() and not c.isspace())

            # frasi
            n_sentences = len(re.findall(r"[.!?]+", text))

            is_numeric_token = 1.0 if text.isdigit() else 0.0
            has_alpha = any(c.isalpha() for c in text)
            is_mixed_alphanum = 1.0 if n_digits and has_alpha else 0.0

            # — B. DOM: antenati, parentela, fratelli, figli —
            parents = [p for p in tag.parents if isinstance(p, Tag)]
            n_parents = len(parents)
            n_ancestors_with_class = sum(1 for p in parents if p.get("class"))
            n_ancestors_with_id = sum(1 for p in parents if p.get("id"))
            n_ancestors_div = sum(1 for p in parents if p.name == "div")
            n_ancestors_section = sum(1 for p in parents if p.name in {"section", "article", "nav", "aside"})
            has_grandparent = 1.0 if n_parents >= 2 else 0.0
            #n_total_ancestors = n_parents

            # figli e discendenti
            #children = [c for c in tag.find_all(recursive=False) if isinstance(c, Tag)]
            children = [c for c in tag.contents if isinstance(c, Tag)]
            n_children = len(children)
            n_descendants = sum(1 for _ in tag.descendants)
            max_child_depth = self._compute_max_depth(tag)
            has_only_text_children = 1.0 if all(not isinstance(c, Tag) for c in tag.contents) else 0.0

            # fratelli
            #siblings = [c for c in tag.parent.children if isinstance(c, Tag)] if tag.parent else []
            if tag.parent:
                siblings = [c for c in tag.parent.contents if isinstance(c, Tag)]
            else:
                siblings = []
            n_siblings = len(siblings) - 1 if siblings else 0
            tag_index_among_siblings = float(siblings.index(tag)) if tag.parent else 0.0
            n_siblings_with_text = sum(1 for s in siblings if s.get_text(strip=True))
            avg_sibling_text_length = (
                sum(len(s.get_text(strip=True)) for s in siblings) / len(siblings)
                if siblings else 0
            )
            n_unique_tags_among_siblings = float(len(set(s.name for s in siblings)))

            # — C. Profondità e posizione —
            #depth_in_dom = n_parents + 1
            #position_in_block = tag_index_among_siblings

            relative_position = (
                position / (total_tokens - 1) if total_tokens and total_tokens > 1 else 0.0
            )
            relative_depth_ratio = (
                n_parents + 1 / max_depth_in_doc if max_depth_in_doc and max_depth_in_doc > 0 else 0.0
            )

            # — D. Statistiche testuali —
            text_density = length / n_children if n_children else 0.0
            normalized_token_length = (
                length / (sum(len(t) for t in all_tokens) / len(all_tokens))
                if all_tokens else 0.0
            )
            # percentile token length: placeholder 0.0
            percentile_token_length = 0.0

            # distanza tra token: placeholder 0.0
            token_distance_to_next = 0.0
            token_distance_to_prev = 0.0

            # — E. Stile inline —
            style = tag.get("style", "") or ""
            has_style_attribute = 1.0 if style else 0.0
            style_props = [kv for kv in style.split(";") if ":" in kv]
            n_style_properties = len(style_props)
            font_size_extracted = self._extract_style_value(style, "font-size")
            margin_top_extracted = self._extract_style_value(style, "margin-top")
            display_type = self._extract_style_token(style, "display")
            display_type_code = float(self.display_map.get(display_type, 3))

            # — F. Contesto locale e discendenti speciali —
            parent_tag_token_count = len([c for c in tag.parent.children if isinstance(c, Tag)]) if tag.parent else 0
            #parent_token_length = len(tag.parent.get_text(strip=True)) if tag.parent else 0
            n_child_tokens_with_digits = sum(
                1 for c in children if any(ch.isdigit() for ch in c.get_text())
            )
            n_descendants_with_digits = sum(
                1 for d in tag.descendants
                if not isinstance(d, Tag) and any(ch.isdigit() for ch in str(d))
            )
            n_descendants_with_links = sum(
                1 for d in tag.descendants if isinstance(d, Tag) and d.name == "a"
            )
            n_descendants_with_buttons = sum(
                1 for d in tag.descendants
                if isinstance(d, Tag) and (
                        d.name == "button" or d.get("role", "") == "button"
                )
            )

            # — G. Pagina intera —
            token_repeated_in_page = (
                all_tokens.count(text) if all_tokens else 0
            )

            # — H. Info su class/id e path —
            n_classes = len(tag.get("class", []))
            class_name_length_avg = (
                sum(len(cn) for cn in tag.get("class", [])) / n_classes
                if n_classes else 0
            )
            id_name_length = len(tag.get("id", "")) if tag.get("id") else 0
            dom_path = self._get_tag_path(tag).split(" > ")
            dom_path_length = len(dom_path)

            # — T. Feature specifiche per TITOLI —
            # 1) è un link?
            is_link = 1.0 if tag.name == "a" else 0.0

            # 2) profondità dell'href ( / annuncio / id / )
            href = tag.get("href", "") if tag.name == "a" else ""
            link_href_depth = href.count("/") if href else 0.0

            # 3) score basato su classi che contengono 'title'
            css_classes = tag.get("class", [])
            try:
                idx_title_cls = next(i for i, c in enumerate(css_classes) if "title" in c.lower())
                css_title_score = 1.0 / (idx_title_cls + 1)        # più bassa l'indice → score alto
            except StopIteration:
                css_title_score = 0.0

            # 4) rapporto lettere maiuscole / tutte le lettere
            n_letters = n_upper + n_lower
            capital_ratio = n_upper / n_letters if n_letters else 0.0

            # 5) keyword score ('via', 'trilocale', 'bilocale', 'appartamento')
            TITLE_KW = {"via", "trilocale", "bilocale", "appartamento"}
            kw_hits = sum(1 for w in words if w.lower() in TITLE_KW)
            keyword_score_title = kw_hits / n_words if n_words else 0.0

            # — I. Feature matematiche extra —
            # 1) char_entropy
            counts = {c: text.count(c) for c in set(text)}
            probs = [cnt / length for cnt in counts.values()]
            char_entropy = -sum(p * math.log2(p) for p in probs if p > 0)

            # 2) length_zscore, 3) n_words_zscore
            lengths = [len(t) for t in all_tokens]
            words_counts = [len(t.split()) for t in all_tokens]
            μ_len, σ_len = statistics.mean(lengths), statistics.stdev(lengths) if len(lengths) > 1 else 1
            μ_wrd, σ_wrd = statistics.mean(words_counts), statistics.stdev(words_counts) if len(words_counts) > 1 else 1
            length_zscore = (length - μ_len) / σ_len
            n_words_zscore = (n_words - μ_wrd) / σ_wrd

            # 4) tf_log
            freq = all_tokens.count(text)
            tf_log = math.log2(freq + 1)

            # 5) distinct_ratio
            distinct_ratio = len(set(text)) / length if length else 0.0

            # 6) uniq_ratio
            token_set = set(w.lower() for w in words)
            vocab_set = set(w.lower() for w in all_tokens for w in w.split())
            uniq_ratio = len(token_set) / len(vocab_set) if vocab_set else 0.0

            # — J. Nuove feature matematiche avanzate —
            # 1) vowel_ratio
            vowels = sum(1 for c in text.lower() if c in "aeiouàèéìòù")
            vowel_ratio = vowels / length if length else 0.0

            # 2) cv_balance
            consonants = sum(1 for c in text.lower() if c.isalpha() and c not in "aeiouàèéìòù")
            cv_balance = (consonants - vowels) / length if length else 0.0

            # 3) max_digit_run
            digit_runs = re.findall(r"\d+", text)
            max_digit_run = max((len(r) for r in digit_runs), default=0)

            # 4) uniq_bigram_ratio
            bigrams = [text[i:i + 2] for i in range(len(text) - 1)]
            uniq_bigram_ratio = len(set(bigrams)) / len(bigrams) if bigrams else 0.0

            # 5) neigh_len_diff
            prev_len = len(all_tokens[position - 1]) if position > 0 else 0
            next_len = len(all_tokens[position + 1]) if position < total_tokens - 1 else 0
            neigh_len_diff = (abs(length - prev_len) + abs(length - next_len)) / 2

            # 6) tfidf_score (richiede idf dict pre‑calcolata)
            #tfidf_score = sum((text.lower().split().count(w) * idf_dict.get(w, 0))
                              #for w in set(words)) / n_words if n_words else 0.0

            # 7) cap_run_norm
            cap_runs = re.findall(r"[A-ZÀ-ÖØ-Ý]+", text)
            max_cap_run = max((len(r) for r in cap_runs), default=0)
            cap_run_norm = max_cap_run / length if length else 0.0

            parent = tag.parent or tag
            #ancestors = DensityFeatures._ancestors(tag)
            #descendants = DensityFeatures._descendants(tag)
            total_text_len = len("".join(all_tokens))  # o somma len(t) per t in all_tokens
            page_token_counts = {t: all_tokens.count(t) for t in set(all_tokens)}
            parent_tokens_positions = [
                idx for idx, (t, _) in enumerate(self.token_tuples) if _ in parent.find_all(recursive=False)
            ]

            # Ritorna tutte le feature come dict
            return {
                # A. base
                "text": text,
                #"position": float(position),
                "length": float(length),
                "n_words": float(n_words),
                "avg_word_length": avg_word_length,
                "n_digits": float(n_digits),
                "n_uppercase": float(n_upper),
                #"n_lowercase": float(n_lower),
                #"n_punctuation": float(n_punct),
                #"n_commas": float(n_commas),
                "n_periods": float(n_periods),
                "n_special_chars": float(n_special),
                "n_sentences": float(n_sentences),
                "is_numeric_token": is_numeric_token,
                "is_mixed_alphanum": is_mixed_alphanum,

                # B. DOM
                "n_parents": float(n_parents),
                #"n_total_ancestors": float(n_total_ancestors),
                "n_ancestors_with_class": float(n_ancestors_with_class),
                "n_ancestors_with_id": float(n_ancestors_with_id),
                "n_ancestors_div": float(n_ancestors_div),
                "n_ancestors_section": float(n_ancestors_section),
                "has_grandparent": has_grandparent,
                "n_children": float(n_children),
                "n_descendants": float(n_descendants),
                "max_child_depth": float(max_child_depth),
                "has_only_text_children": has_only_text_children,
                "n_siblings": float(n_siblings),
                "tag_index_among_siblings": tag_index_among_siblings,
                "n_siblings_with_text": float(n_siblings_with_text),
                "avg_sibling_text_length": avg_sibling_text_length,
                "n_unique_tags_among_siblings": float(n_unique_tags_among_siblings),

                # C. posizione
                #"depth_in_dom": float(depth_in_dom),
                #"position_in_block": position_in_block,
                "relative_position": relative_position,
                "relative_depth_ratio": relative_depth_ratio,

                # D. statistiche
                "text_density": text_density,
                "normalized_token_length": normalized_token_length,
                "percentile_token_length": percentile_token_length,
                "token_distance_to_next": token_distance_to_next,
                "token_distance_to_prev": token_distance_to_prev,

                # E. stile inline
                "has_style_attribute": has_style_attribute,
                "n_style_properties": float(n_style_properties),
                "font_size_extracted": font_size_extracted,
                "margin_top_extracted": margin_top_extracted,
                "display_type_code": display_type_code,

                # F. contesto locale
                "parent_tag_token_count": float(parent_tag_token_count),
                #"parent_token_length": float(parent_token_length),
                "n_child_tokens_with_digits": float(n_child_tokens_with_digits),
                "n_descendants_with_digits": float(n_descendants_with_digits),
                "n_descendants_with_links": float(n_descendants_with_links),
                "n_descendants_with_buttons": float(n_descendants_with_buttons),

                # G. pagina intera
                "token_repeated_in_page": float(token_repeated_in_page),

                # H. class/id/path
                "n_classes": float(n_classes),
                "class_name_length_avg": class_name_length_avg,
                "id_name_length": float(id_name_length),
                "dom_path_length": float(dom_path_length),

                # T. feature Titolo
                "is_link": is_link,
                "link_href_depth": float(link_href_depth),
                "css_title_score": css_title_score,
                "capital_ratio": capital_ratio,
                "keyword_score_title": keyword_score_title,

                "char_entropy": char_entropy,
                "length_zscore": length_zscore,
                "n_words_zscore": n_words_zscore,
                "tf_log": tf_log,
                "distinct_ratio": distinct_ratio,
                "uniq_ratio": uniq_ratio,

                "vowel_ratio": vowel_ratio,
                "cv_balance": cv_balance,
                "max_digit_run": float(max_digit_run),
                "uniq_bigram_ratio": uniq_bigram_ratio,
                "neigh_len_diff": neigh_len_diff,
                #"tfidf_score": tfidf_score,
                "cap_run_norm": cap_run_norm,

                # DensityFeatures
                "ancestors_text_density_variance": DensityFeatures.ancestors_text_density_variance(tag),
                "sibling_text_density": DensityFeatures.sibling_text_density(tag),
                "block_text_token_density": DensityFeatures.block_text_token_density(tag),
                "block_tag_density": DensityFeatures.block_tag_density(tag),
                "descendants_text_density": DensityFeatures.descendants_text_density(tag),
                "ancestors_avg_tag_density": DensityFeatures.ancestors_avg_tag_density(tag),
                "text_ratio_in_block": DensityFeatures.text_ratio_in_block(tag),
                "tag_ratio_in_block": DensityFeatures.tag_ratio_in_block(tag),
                "block_avg_word_length": DensityFeatures.block_avg_word_length(tag),
                "block_std_word_length": DensityFeatures.block_std_word_length(tag),
                "block_avg_sentence_count": DensityFeatures.block_avg_sentence_count(tag),
                "descendants_link_density": DensityFeatures.descendants_link_density(tag),
                "descendants_img_density": DensityFeatures.descendants_img_density(tag),
                "block_special_char_density": DensityFeatures.block_specialchar_density(tag),
                "block_digit_density": DensityFeatures.block_digit_density(tag),
                "token_dist_to_block_mid": DensityFeatures.token_distance_to_block_midpoint(position, len(parent_tokens_positions)),
                "block_token_count": DensityFeatures.block_token_count(tag),
                "block_depth_wtd_token_den": DensityFeatures.block_depth_weighted_token_density(tag),
                "siblings_tag_density": DensityFeatures.siblings_tag_density(tag),
                "local_subtree_height": DensityFeatures.local_subtree_height(tag),
                "parent_child_ratio": DensityFeatures.parent_child_ratio(tag),
                "block_text_length_ratio": DensityFeatures.block_text_length_ratio(tag, total_text_len),
                "local_token_rep_density": DensityFeatures.local_token_repetition_density(tag, page_token_counts),
                "block_avg_token_distance": DensityFeatures.block_avg_token_distance(parent_tokens_positions, position),
                "normalized_block_area": DensityFeatures.normalized_block_area(tag, max_depth_in_doc, total_tokens, position),
            }

        except Exception as e:
            self.logger.log_exception("Errore in extract_features", e)
            raise

    def _compute_max_depth(self, tag: Tag) -> int:
        """Restituisce la profondità massima ricorsiva tra i figli."""
        max_d = 0
        for child in tag.find_all(recursive=False):
            if isinstance(child, Tag):
                d = 1 + self._compute_max_depth(child)
                if d > max_d:
                    max_d = d
        return max_d

    def _extract_style_value(self, style: str, prop: str) -> float:
        """
        Cerca in style inline la proprietà `prop` e restituisce il valore numerico (es. '12px' → 12).
        """
        m = re.search(rf"{prop}\s*:\s*([\d\.]+)", style)
        return float(m.group(1)) if m else 0.0

    def _extract_style_token(self, style: str, prop: str) -> str:
        """
        Cerca in style inline la proprietà `prop` e restituisce il suo valore testuale ('block', 'inline', ecc.).
        """
        m = re.search(rf"{prop}\s*:\s*([^;]+)", style)
        return m.group(1).strip() if m else ""

    def _get_tag_path(self, tag: Tag) -> str:
        """
        Costruisce un semplice 'XPath-like' basato sui tag, es. 'html > body > div > span'
        """
        path = []
        current = tag
        while current and isinstance(current, Tag):
            path.append(current.name)
            current = current.parent
        return " >".join(reversed(path))

    def _compute_window_stats_for_feature(
        self,
        raw_features: List[Dict[str, float]],
        feature_key: str,
        window_size: int = 2
    ) -> None:
        """
        Calcola media e std di raw_features[i][feature_key] su finestra ±window_size
        e le aggiunge in-place come:
          - window_mean_{feature_key}
          - window_std_{feature_key}
        """
        # Estrai la lista di valori
        values = [f.get(feature_key, 0.0) for f in raw_features]
        n = len(values)

        for i in range(n):
            lo = max(0, i - window_size)
            hi = min(n, i + window_size + 1)
            window = values[lo:hi]

            mean = sum(window) / len(window)
            std = (sum((x - mean) ** 2 for x in window) / len(window)) ** 0.5

            raw_features[i][f"window_mean_{feature_key}"] = mean
            raw_features[i][f"window_std_{feature_key}"]  = std


    # ---------------------------------------------------------------------
    #                 FEATURE GROUP 1  –  Picco locale (±1)
    # ---------------------------------------------------------------------
    @staticmethod
    def _add_local_length_peaks(raw_features: List[Dict[str, float]]) -> None:
        """
        Aggiunge:
            - length_is_local_max         (0/1)
            - length_peak_prominence      (>0 se picco)
            - length_gradient_sign        (-1/0/+1)
        Usa solo la lunghezza del token e dei suoi vicini immediati.
        """
        n = len(raw_features)
        lengths = [f["length"] for f in raw_features]

        for i in range(n):
            prev_len = lengths[i - 1] if i > 0 else lengths[i]
            next_len = lengths[i + 1] if i < n - 1 else lengths[i]

            raw_features[i]["length_is_local_max"] = float(lengths[i] > prev_len and lengths[i] > next_len)

            denom = max(prev_len, next_len) or 1.0
            raw_features[i]["length_peak_prominence"] = max(0.0, (lengths[i] - denom) / denom)

            raw_features[i]["length_gradient_sign"] = float((next_len > prev_len) - (next_len < prev_len))  # -1,0,+1

    # ---------------------------------------------------------------------
    #                 FEATURE GROUP 2  –  Picco nel “blocco annuncio”
    # ---------------------------------------------------------------------
    @staticmethod
    def _add_block_based_peaks(raw_features: List[Dict[str, float]], price_token_pattern="€") -> None:
        """
        Definisce un “blocco annuncio” come la sequenza di token compresa
        tra due token che contengono il carattere '€' (prezzo).
        Aggiunge per ogni token:
            - length_rank_in_block
            - length_z_to_block_median
            - dist_from_last_price_token
            - is_between_price_and_locali   (0/1)
        """
        import statistics

        #n = len(raw_features)
        #lengths = [f["length"] for f in raw_features]
        n = len(raw_features)
        lengths = [f["length"] for f in raw_features]

        # -------- inizializza a 0 per tutti i token -----------
        for f in raw_features:
            f["length_rank_in_block"] = 0.0
            f["length_z_to_block_median"] = 0.0
            f["dist_from_last_price_token"] = 0.0
            f["is_between_price_and_locali"] = 0.0

        # identifica indici dove compare '€'
        price_idx = [i for i, f in enumerate(raw_features) if price_token_pattern in f["text"]]

        # aggiungi sentinelle per gestire inizio/fine documento
        price_idx = [-1] + price_idx + [n]

        for i_price in range(len(price_idx) - 1):
            block_start = price_idx[i_price]        # indice del token con '€' (o -1 all’inizio)
            block_end   = price_idx[i_price + 1]    # indice del prossimo token con '€' (o n alla fine)
            block = list(range(block_start + 1, block_end))  # token “interni” all’annuncio
            if not block:
                continue

            block_lengths = [lengths[i] for i in block]
            median = statistics.median(block_lengths)
            stdev = statistics.stdev(block_lengths) if len(block) > 1 else 1.0

            # rank by desc length
            sorted_idx = sorted(block, key=lambda i: lengths[i], reverse=True)
            # primo token che contiene 'locali'
            locali_pos = next((i for i in block if "locali" in raw_features[i]["text"].lower()), None)

            for rank, i in enumerate(sorted_idx, start=1):
                raw_features[i]["length_rank_in_block"] = float(rank)
                raw_features[i]["length_z_to_block_median"] = (lengths[i] - median) / stdev
                raw_features[i]["dist_from_last_price_token"] = float(i - block_start)
                raw_features[i]["is_between_price_and_locali"] = float(
                    1 if (locali_pos is not None and i < locali_pos) else 0
                )


