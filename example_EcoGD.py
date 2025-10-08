import torch
import torchvision
from ast import literal_eval
import time
import EcoEdgeInfer

# initialize the energy optimizer. Warning: This will start the queue service
# EcoEdgeInfer.optimizer = EcoEdgeInfer.EnergyOptimizer_Gradient_Descent() #指定使用的优化方法
# EcoEdgeInfer.optimizer = EcoEdgeInfer.EnergyOptimizer_BayesianOptimization() # 使用贝叶斯优化方法
EcoEdgeInfer.optimizer = EcoEdgeInfer.EnergyOptimizer_MAB_multiDim()  # 使用MAB优化方法


# use the following lines to save the logs to a file
experiment_name = "EcoGD" 
EcoEdgeInfer.logs_text_prefix = "results/" + time.strftime("%Y%m%d-%H%M%S") + "_" + experiment_name + "_"

# uncomment the following lines to set the baseline from saved csv file
# EcoEdgeInfer.optimizer.set_baseline(IAT=0.050,fname = "master_reference_all_max.csv")

# Loading the model
model = torchvision.models.resnet50(pretrained=True).cuda()
# new versions of torchvision have the following syntax
# torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1).cuda()

# Defining the function to be optimized
@EcoEdgeInfer.queue_add
def run_inference(inp):
    return model(inp)

# Defining the function to create input
def create_input():
    return torch.rand(1, 3, 224, 224)

# Running the inference
# 每416次（cache_length=416）触发一次优化器
for i in range(416*100):
    run_inference(create_input())
    time.sleep(0.050)

print("done launching")

# Stop the queue service and save the logs
EcoEdgeInfer.queue_service_stop()
