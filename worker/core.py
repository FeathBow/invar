import json
import subprocess


def unique(pairs):
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("Duplicate JSON fields")
    return result


def invalid_constant(value):
    raise ValueError(f"Invalid JSON numeric constant: {value}")


def decode(encoded):
    return json.loads(encoded, object_pairs_hook=unique, parse_constant=invalid_constant)


def invoke(arguments, *, executable, stdin=None):
    completed = subprocess.run([str(executable), *map(str, arguments)], capture_output=True, text=True, check=False, input=stdin)
    if completed.returncode != 0:
        raise ValueError(completed.stderr.strip() or f"Invar core process exited with status {completed.returncode}")
    value = decode(completed.stdout)
    if not isinstance(value, dict):
        raise ValueError("Expected an Invar core response object")
    return value


def exchange(arguments, *, executable, handler):
    with subprocess.Popen([str(executable), *map(str, arguments)], stdin=subprocess.PIPE,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0) as process:
        try:
            result = None
            for line in process.stdout:
                value = decode(line)
                if not isinstance(value, dict):
                    raise ValueError("Expected an Invar core response object")
                if "codec" not in value:
                    result = value
                    break
                handler(value, process.stdin)
            process.stdin.close()
            process.stdin = None
            trailing, error = process.communicate()
            if process.returncode:
                raise ValueError(error.decode().strip() or f"Invar core process exited with status {process.returncode}")
            if result is None or trailing:
                raise ValueError("Expected exactly one completed Invar core comparison")
            return result
        except BaseException:
            process.kill()
            process.communicate()
            raise
