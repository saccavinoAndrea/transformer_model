# src/immobiliare/config_domain/token.py

import json
from dataclasses import dataclass, asdict
from typing import Dict, Any

from bs4 import Tag

from immobiliare.utils.logging.logger_factory import LoggerFactory

"""
@dataclass(frozen=True)
class Token:
    text: str
    tag: Tag
    #position: int
    length: float
    n_words: float
    avg_word_length: float
    n_digits: float
    n_uppercase: float
    #n_lowercase: float
    #n_punctuation: float
    #n_commas: float
    n_periods: float
    n_special_chars: float
    n_sentences: float
    is_numeric_token: float
    is_mixed_alphanum: float
    n_parents: float
    #n_total_ancestors: float
    n_ancestors_with_class: float
    n_ancestors_with_id: float
    n_ancestors_div: float
    n_ancestors_section: float
    has_grandparent: float
    n_children: float
    n_descendants: float
    max_child_depth: float
    has_only_text_children: float
    n_siblings: float
    tag_index_among_siblings: float
    n_siblings_with_text: float
    avg_sibling_text_length: float
    n_unique_tags_among_siblings: float
    #depth_in_dom: float
    #position_in_block: float
    relative_position: float
    relative_depth_ratio: float
    text_density: float
    normalized_token_length: float
    percentile_token_length: float
    token_distance_to_next: float
    token_distance_to_prev: float
    has_style_attribute: float
    n_style_properties: float
    font_size_extracted: float
    margin_top_extracted: float
    display_type_code: float
    parent_tag_token_count: float
    #parent_token_length: float
    n_child_tokens_with_digits: float
    n_descendants_with_digits: float
    n_descendants_with_links: float
    n_descendants_with_buttons: float
    token_repeated_in_page: float
    n_classes: float
    class_name_length_avg: float
    id_name_length: float
    dom_path_length: float
    window_mean_length: float
    window_std_length: float
    window_mean_avg_word_length: float
    window_std_avg_word_length: float
    is_link: float
    link_href_depth: float
    css_title_score: float
    capital_ratio: float
    keyword_score_title: float
    char_entropy: float
    length_zscore: float
    n_words_zscore: float
    tf_log: float
    distinct_ratio: float
    uniq_ratio: float
    vowel_ratio: float
    cv_balance: float
    max_digit_run: float
    uniq_bigram_ratio: float
    neigh_len_diff: float
    #tfidf_score: float
    cap_run_norm: float
    length_is_local_max: float
    length_peak_prominence: float
    length_gradient_sign: float
    length_rank_in_block: float
    length_z_to_block_median: float
    dist_from_last_price_token: float
    is_between_price_and_locali: float
    block_text_token_density: float
    block_tag_density: float
    ancestors_avg_tag_density: float
    descendants_text_density: float
    sibling_text_density: float
    text_ratio_in_block: float
    tag_ratio_in_block: float
    block_avg_word_length: float
    block_std_word_length: float
    block_avg_sentence_count: float
    ancestors_text_density_variance: float
    descendants_link_density: float
    descendants_img_density: float
    block_special_char_density: float
    block_digit_density: float
    token_dist_to_block_mid: float
    block_token_count: float
    block_depth_wtd_token_den: float
    siblings_tag_density: float
    local_subtree_height: float
    parent_child_ratio: float
    block_text_length_ratio: float
    local_token_rep_density: float
    block_avg_token_distance: float
    normalized_block_area: float
    meta: Dict[str, Any]
"""

from dataclasses import dataclass, field
from typing import Dict, Any
from bs4 import Tag

@dataclass(frozen=True)
class Token:
    text: str | None = None
    tag: Tag | None = None
    position: int | None = None
    length: float | None = None
    n_words: float | None = None
    avg_word_length: float | None = None
    n_digits: float | None = None
    n_uppercase: float | None = None
    n_lowercase: float | None = None
    n_punctuation: float | None = None
    n_commas: float | None = None
    n_periods: float | None = None
    n_special_chars: float | None = None
    n_sentences: float | None = None
    is_numeric_token: float | None = None
    is_mixed_alphanum: float | None = None
    n_parents: float | None = None
    n_total_ancestors: float | None = None
    n_ancestors_with_class: float | None = None
    n_ancestors_with_id: float | None = None
    n_ancestors_div: float | None = None
    n_ancestors_section: float | None = None
    has_grandparent: float | None = None
    n_children: float | None = None
    n_descendants: float | None = None
    max_child_depth: float | None = None
    has_only_text_children: float | None = None
    n_siblings: float | None = None
    tag_index_among_siblings: float | None = None
    n_siblings_with_text: float | None = None
    avg_sibling_text_length: float | None = None
    n_unique_tags_among_siblings: float | None = None
    depth_in_dom: float | None = None
    position_in_block: float | None = None
    relative_position: float | None = None
    relative_depth_ratio: float | None = None
    text_density: float | None = None
    normalized_token_length: float | None = None
    percentile_token_length: float | None = None
    token_distance_to_next: float | None = None
    token_distance_to_prev: float | None = None
    has_style_attribute: float | None = None
    n_style_properties: float | None = None
    font_size_extracted: float | None = None
    margin_top_extracted: float | None = None
    display_type_code: float | None = None
    parent_tag_token_count: float | None = None
    parent_token_length: float | None = None
    n_child_tokens_with_digits: float | None = None
    n_descendants_with_digits: float | None = None
    n_descendants_with_links: float | None = None
    n_descendants_with_buttons: float | None = None
    token_repeated_in_page: float | None = None
    n_classes: float | None = None
    class_name_length_avg: float | None = None
    id_name_length: float | None = None
    dom_path_length: float | None = None
    window_mean_length: float | None = None
    window_std_length: float | None = None
    window_mean_avg_word_length: float | None = None
    window_std_avg_word_length: float | None = None
    is_link: float | None = None
    link_href_depth: float | None = None
    css_title_score: float | None = None
    capital_ratio: float | None = None
    keyword_score_title: float | None = None
    char_entropy: float | None = None
    length_zscore: float | None = None
    n_words_zscore: float | None = None
    tf_log: float | None = None
    distinct_ratio: float | None = None
    uniq_ratio: float | None = None
    vowel_ratio: float | None = None
    cv_balance: float | None = None
    max_digit_run: float | None = None
    uniq_bigram_ratio: float | None = None
    neigh_len_diff: float | None = None
    tfidf_score: float | None = None
    cap_run_norm: float | None = None
    length_is_local_max: float | None = None
    length_peak_prominence: float | None = None
    length_gradient_sign: float | None = None
    length_rank_in_block: float | None = None
    length_z_to_block_median: float | None = None
    dist_from_last_price_token: float | None = None
    is_between_price_and_locali: float | None = None
    block_text_token_density: float | None = None
    block_tag_density: float | None = None
    ancestors_avg_tag_density: float | None = None
    descendants_text_density: float | None = None
    sibling_text_density: float | None = None
    text_ratio_in_block: float | None = None
    tag_ratio_in_block: float | None = None
    block_avg_word_length: float | None = None
    block_std_word_length: float | None = None
    block_avg_sentence_count: float | None = None
    ancestors_text_density_variance: float | None = None
    descendants_link_density: float | None = None
    descendants_img_density: float | None = None
    block_special_char_density: float | None = None
    block_digit_density: float | None = None
    token_dist_to_block_mid: float | None = None
    block_token_count: float | None = None
    block_depth_wtd_token_den: float | None = None
    siblings_tag_density: float | None = None
    local_subtree_height: float | None = None
    parent_child_ratio: float | None = None
    block_text_length_ratio: float | None = None
    local_token_rep_density: float | None = None
    block_avg_token_distance: float | None = None
    normalized_block_area: float | None = None
    meta: Dict[str, Any] | None = field(default=None)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = None) -> str:
        logger = LoggerFactory.get_logger("token")
        try:
            return json.dumps(asdict(self), ensure_ascii=False, indent=indent)
        except Exception as e:
            logger.log_exception("Errore serializzazione Token in JSON", e)
            raise

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Token":
        logger = LoggerFactory.get_logger("token")
        try:
            return Token(**data)
        except Exception as e:
            logger.log_exception("Errore nella creazione Token da dizionario", e)
            raise

    @staticmethod
    def from_json(json_str: str) -> "Token":
        logger = LoggerFactory.get_logger("token")
        try:
            data = json.loads(json_str)
            return Token(**data)
        except Exception as e:
            logger.log_exception("Errore nella deserializzazione Token da JSON", e)
            raise
