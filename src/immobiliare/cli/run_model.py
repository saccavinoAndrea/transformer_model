import time

from immobiliare.pipeline import PreprocessPipeline, LabelingPipeline, TrainingPipeline, InferencePipeline


def main():
    start_time = time.time()  # ← inizio cronometro

    preprocess = PreprocessPipeline()
    labeling = LabelingPipeline()
    training = TrainingPipeline()
    inference = InferencePipeline()

    preprocess.run(None)
    labeling.run(None)
    training.run(None)
    inference.run(None)

    total_time = time.time() - start_time
    m, s = divmod(total_time, 60)
    h, m = divmod(m, 60)
    print(f"\n⏱️  Tempo totale esecuzione modello preprocess, labeling, training, inference: {int(h)}h {int(m)}m {int(s)}s")

if __name__ == "__main__":
    main()