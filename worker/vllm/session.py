from worker.vllm import entry as vllm_entry


def main():
    # Import the numerical protocol only after diagnostics have a dedicated stream.
    def protocol(options, **dependencies):
        from worker.hf.session import serve

        serve(options, **dependencies)

    options = vllm_entry.batch_arguments("Native inference session with one prepared engine")
    vllm_entry.run(options, protocol=protocol)


if __name__ == "__main__":
    main()
