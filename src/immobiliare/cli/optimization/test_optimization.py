from immobiliare.pipeline.steps import FileLoadingStep, TokenizerParallelStep
from deprecated.tokenizer_html_and_feature_extraction_step import TokenizerHtmlAndFeatureExtractionStep



if __name__ == "__main__":

    html_pages_dir_to_predict = "data/html_pages_dir"
    page_number_to_predict = 100

    file_loader = FileLoadingStep(input_dir=str(html_pages_dir_to_predict), limit=page_number_to_predict)
    tokenizer_extractor = TokenizerHtmlAndFeatureExtractionStep()

    html_page_to_predict = file_loader.run()

    #all_tokens_with_features = tokenizer_extractor.run(data=html_page_to_predict)
    """for i in all_tokens_with_features[1:2]:
        token = i.to_dict()
        for k, v in token.items():
            print(k, v)"""

    tokenizer_extractor_parallel = TokenizerParallelStep()
    all_tokens_with_features_p = tokenizer_extractor_parallel.run(data=html_page_to_predict)
    """for i in all_tokens_with_features_p[1:2]:
        token = i.to_dict()
        for k, v in token.items():
            print(k, v)"""