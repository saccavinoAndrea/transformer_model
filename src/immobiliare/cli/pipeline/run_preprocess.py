# src/immobiliare/cli/run_preprocess.py

from immobiliare.pipeline import PreprocessPipeline


def main():

    pipeline = PreprocessPipeline()
    pipeline.run(None)

if __name__ == "__main__":
    main()
