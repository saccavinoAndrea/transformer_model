# src/immobiliare/cli/run_inference.py

from pipeline.steps import EvaluationStep


def main():

    inference_evaluation = (
        EvaluationStep(
            predictions_complete_path="data/transformer_model/inference/artifacts/predictions/predicted_complete.csv",
            ground_truth_path="data/transformer_model/inference/ground_truth/ground_truth.csv",
            report_json="data/transformer_model/inference/eval/report.json",
            report_csv="data/transformer_model/inference/eval/error_analysis.csv",
            cm_png="data/transformer_model/inference/eval/confusion_matrix.png",
        )
    )
    inference_evaluation.run(None)

if __name__ == "__main__":
    main()
