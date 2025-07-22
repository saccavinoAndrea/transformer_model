# src/immobiliare/cli/run_inference.py
import time

from immobiliare.pipeline import InferencePipeline


def main():
    start_time = time.time()  # ← inizio cronometro

    inference = InferencePipeline()
    inference.run(None)

    total_time = time.time() - start_time
    m, s = divmod(total_time, 60)
    h, m = divmod(m, 60)
    print(f"\n⏱️  Tempo totale di inference: {int(h)}h {int(m)}m {int(s)}s")

if __name__ == "__main__":
    main()
