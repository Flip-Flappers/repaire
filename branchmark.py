from robustbench.utils import load_model
# Load a model from the model zoo
model = load_model(model_name='Sehwag2021Proxy_R18',
                   dataset='cifar10',
                   threat_model='Linf')

# Evaluate the Linf robustness of the model using AutoAttack
from robustbench.eval import benchmark
clean_acc, robust_acc = benchmark(model,
                                  dataset='cifar10',
                                  threat_model='Linf')