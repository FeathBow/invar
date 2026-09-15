from worker.vllm import entry as vllm_entry


def main():
    vllm_entry.inspect(vllm_entry.batch_arguments("Identify an actual native PEFT materialization"))


if __name__ == "__main__":
    main()
