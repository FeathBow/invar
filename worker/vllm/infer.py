from worker.hf.infer import arguments, run
from worker.vllm import entry as vllm_entry


def main():
    options = arguments(parser=vllm_entry.parser("Native inference with a bound PEFT handoff"))
    vllm_entry.run(options, protocol=run)


if __name__ == "__main__":
    main()
