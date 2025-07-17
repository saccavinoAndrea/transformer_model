# src/immobiliare/cli/run_preprocess.py

from pipeline import PreprocessPipeline


def main():

    pipeline = PreprocessPipeline()
    pipeline.run(None)

if __name__ == "__main__":
    main()
