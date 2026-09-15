from pathlib import Path

from worker.vllm import entry as vllm_entry


def main():
    parser = vllm_entry.parser("Native finite request batch with atomic core permission")
    parser.add_argument("--cache", type=Path, required=True)
    vllm_entry.run_batch(parser.parse_args())


if __name__ == "__main__":
    main()
