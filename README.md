## Inference service

* This repository builds the nulog-inference-service image which is used for both workload/application log inferencing as well as control plane log inferencing
  * Set `IS_CONTROL_PLANE_SERVICE` to `"True"` if you'd like to inference on control plane logs - which doesn't require a GPU
  * Inferencing on workload/application logs (a model that is generated by a job nulog-training-controller kicks off) you must have an NVIDIA GPU attached to the cluster

### Install NVIDIA GPU driver
```
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.6.0/nvidia-device-plugin.yml
```

## Testing
Install libs for testing
```
pip install -r requirements.txt
pip install -r test-requirements.txt
```

Run pytest with coverage report on the src dir `opni_inference_service`
```
pytest --cov opni_inference_service
```

## Contributing
We use `pre-commit` for formatting auto-linting and checking import. Please refer to [installation](https://pre-commit.com/#installation) to install the pre-commit or run `pip install pre-commit`. Then you can activate it for this repo. Once it's activated, it will lint and format the code when you make a git commit. It makes changes in place. If the code is modified during the reformatting, it needs to be staged manually.

```
# Install
pip install pre-commit

# Install the git commit hook to invoke automatically every time you do "git commit"
pre-commit install

# (Optional)Manually run against all files
pre-commit run --all-files
```
