import json

class HyperParameters:
    def __init__(self):
        self._MODEL_THRESHOLD = float(0.7)
        self._MIN_LOG_TOKENS = int(1)
        f = open('/etc/opni/hyperparameters.json', 'r')
        data = json.load(f)
        if "hyperparameters" not in data:
            return
        hyperparameters = data["hyperparameters"]
        if "modelThreshold" in hyperparameters["modelThreshold"]:
            self._MODEL_THRESHOLD = float(data["modelThreshold"])
        if "minLogTokens" in hyperparameters["minLogTokens"]:
            self._MODEL_THRESHOLD = float(data["minLogTokens"])
        f.close()


    @property
    def MODEL_THRESHOLD(self):
        return self._MODEL_THRESHOLD

    @property
    def MIN_LOG_TOKENS(self):
        return self._MIN_LOG_TOKENS