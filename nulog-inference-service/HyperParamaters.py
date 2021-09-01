import json

class HyperParameters:
    def __init__(self):
        self._MODEL_THRESHOLD = float(0.7)
        self._MIN_LOG_TOKENS = int(1)
        self._IS_CONTROL_PLANE = False
        f = open('/etc/opni/hyperparameters.json', 'r')
        data = json.load(f)
        if data is None:
            return
        if "modelThreshold" in data:
            self._MODEL_THRESHOLD = float(data["modelThreshold"])
        if "minLogTokens" in data:
            self._MIN_LOG_TOKENS = int(data["minLogTokens"])
        if "isControlPlane" in data:
            self._IS_CONTROL_PLANE = data["isControlPlane"].lower() == "true"
        f.close()


    @property
    def MODEL_THRESHOLD(self):
        return self._MODEL_THRESHOLD

    @property
    def MIN_LOG_TOKENS(self):
        return self._MIN_LOG_TOKENS

    @property
    def IS_CONTROL_PLANE(self):
        return self._IS_CONTROL_PLANE