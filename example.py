import torch
import torchvision
from ast import literal_eval
import time
import energy_optimizer

# use last_sweep and sweep_next_mapping to control the order of the sweeps

# energy_optimizer.optimizer = energy_optimizer.EnergyOptimizer_BayesianOptimization()
# experiment_name = "BayesianOptimization" #use this to name the log file

# energy_optimizer.optimizer = energy_optimizer.EnergyOptimizer_Gradient_Descent(jump_learn_factor = 400)
# experiment_name = "Gradient_Descent_learning_rate_5" #use this to name the log file

energy_optimizer.optimizer = energy_optimizer.EnergyOptimizer_Gradient_Descent(arr_rate_thres_pcent = 10)
experiment_name = "Gradient_hist_reset_10pcent" #use this to name the log file

energy_optimizer.logs_text_prefix = "results/" + time.strftime("%Y%m%d-%H%M%S") + "_" + experiment_name + "_"
energy_optimizer.optimizer.set_baseline(IAT=0.050)

model = torchvision.models.resnet50(pretrained=True).cuda()
# new way is torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1).cuda()

@energy_optimizer.queue_add
def run_inference(inp):
    return model(inp)

def create_input():
    return torch.rand(1, 3, 224, 224)

for i in range(416*100):
    run_inference(create_input())
    time.sleep(0.050)

print("done launching")

energy_optimizer.queue_service_stop()
