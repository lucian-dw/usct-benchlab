"""Lazy public entry: optional neural dependencies are not imported at discovery."""


class FTTTAlgorithm:
    name = "fttt"

    def run(self, case, config):
        try:
            from .adapter import reconstruct
        except ModuleNotFoundError as exc:
            if exc.name == "torch":
                raise RuntimeError(
                    "FTTT requires PyTorch; install usct-benchlab[neural]."
                ) from exc
            raise
        return reconstruct(case, config)
