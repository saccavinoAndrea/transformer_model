# src/immobiliare/pipeline/steps/feature_extraction_step.py

from typing import Any, List, Tuple, Dict
from bs4 import Tag

from immobiliare.core_interfaces.pipeline.ipipeline_step import IPipelineStep
from immobiliare.domain.token import Token
from immobiliare.preprocess.html.feature.feature_extractor import FeatureExtractor
from immobiliare.utils.logging.logger_factory import LoggerFactory
from immobiliare.utils.logging.decorators import log_exec

class FeatureExtractionStep(IPipelineStep):
    def __init__(self):
        self.extractor = FeatureExtractor()
        self.logger = LoggerFactory.get_logger("feature_extraction_step")

    @log_exec(logger_name="feature_extraction_step", method_name="run")
    def run(self, data: List[Tuple[str, Tag]]) -> List[Token]:
        """
        :param data: List[Dict] con chiavi 'filename' e 'content'
                     oppure List[Tuple[str, Tag]] se usi lo step precedente
        :return: List[Token] con tutti i campi (incl. tag e meta) riempiti
        """
        if not data:
            return []

        total_tokens = len(data)
        all_texts = [text for text, _ in data]
        # Se ricevi ancora tuple (text, tag), decomprimile
        token_tuples: List[Tuple[str, Tag]] = data  # type: ignore

        # ⚠️ Calcolo max_depth una sola volta a partire dal tag comune, tipicamente il parent del primo token
        root_tag = data[0][1]
        soup_root = root_tag.find_parent("html") or root_tag
        max_depth = self.extractor.compute_max_depth(soup_root)
        tokens: List[Token] = []

        for idx, (text, tag) in enumerate(token_tuples):
            try:

                # Estrai le feature numeriche (include 'token' nel dict)
                features: Dict[str, Any] = self.extractor.extract_features(
                    text,
                    tag,
                    position=idx,
                    total_tokens=len(token_tuples),
                    max_depth_in_doc=max_depth,
                    all_tokens=[t for t, _ in token_tuples]
                )

                # Costruisci il Token passando tag e meta
                token_obj = Token(
                    text=features.pop("text"),
                    tag=tag,
                    meta={"filename": getattr(data[idx], "filename", None)},
                    **features
                )

                tokens.append(token_obj)

            except Exception as e:
                self.logger.log_exception(f"Errore su token #{idx} → {text[:30]}", e)

        self.logger.log_info(f"Feature estratte e Token creati: {len(tokens)}")
        return tokens
