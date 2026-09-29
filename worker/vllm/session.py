from pathlib import Path

from worker.vllm import entry as vllm_entry


def main():
    # Import the numerical protocol only after diagnostics have a dedicated stream.
    def protocol(options, **dependencies):
        from worker.session import serve

        serve(options, **dependencies)

    from worker.session import declare

    arguments = declare(vllm_entry.parser("Native inference session with one prepared engine"))
    arguments.add_argument("--cache", type=Path, required=True)
    arguments.add_argument("--adapter", type=Path, required=True)
    options = arguments.parse_args()
    vllm_entry.run(options, protocol=protocol)


if __name__ == "__main__":
    main()
