# src/immobiliare/cli/run_labeling.py

from immobiliare.pipeline import LabelingPipeline

def main():

    pipeline = LabelingPipeline()
    pipeline.run(None)

if __name__ == "__main__":
    main()
