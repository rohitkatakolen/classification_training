import torch
#import torch_pruning as tp
import numpy as np
import json
import yaml

'''

We tried two pruning approaches - 



1. torch_pruning library

    This library actually pruned the model and reduced the size of the model. 
    However, there was significant drop in accuracy.
    We had to retrain the model with enitre dataset.
    but Here we can see improvement in fps with this library

    
    Below are the stats - 

    resnet18 model
    intel image dataset (full retrained)
    conv_prune_amount:0.1
    dense_prune_amount:0.2
    before pruning : 85%acc, 11.2M params - 38.1 fps - 567 fps (trt-opt) 
    after pruning  : 85%acc, 7.7M params  - 46.6 fps - 1.09e+03 fps (trt-opt)
    
    
2. pytorch builtin prune library

    This library doesnt acutally prune the model. It introduces sparsity in models by setting certain weights to 0 (zero).
    Hence there is no size redcution. 
    Also there is no improvement in fps.
    But this model gives 4 different methods to choose from 

    Below are the stats -

    Resnet18 
    Cifar 10 dataset
    Unpruned model - 82.26% acc
    (No of parameters doesnt change in pytorch pruning methods)

    convulution prune amount = 0.2
    dense prune amount       = 0.2
    Pruning methods-

    [Structured pruning - entire channel of conv]
    ln_structured - (n=1)68.57,  (n=2)70.65,  (n=3)74.4%
    random_structured - (10-20)% acc

    [Unstrcutured pruning]
    l1_unstructured - 82.23% acc
    random_unstructured - (10-20)% acc

    Following are fps stas - 

    resnet18_normal                      - 39.4 fps - 1060 fps (trt_optimized) 
    resnet18_ln_structured_prune         - 39.4 fps - 1120 fps (trt_optimized)
    resnet18_random_structured_prune     - 39.1 fps - 1050 fps (trt_optimized)
    resnet18_l1_unstructured_prune       - 39.1 fps - 1040 fps (trt_optimized)
    resnet18_random_unstructured_prune   - 39.1 fps - 1040 fps (trt_optimized)



# CONCLUSION -
# Based on above comparision. torch_pruning lib was found to be more useful as 
# model size reduces, fps increases at the cost of retraining with full dataset

'''

class Pruning:
    ALEXNET_MODEL_FAMILY = ['alexnet']
    SQUEEZENET_MODEL_FAMILY = ['squeezenet1_0','squeezenet1_1']
    RESNET_MODEL_FAMILY = ['resnet18','resnet34','resnet50','resnet101','resnet152']
    DENSENET_MODEL_FAMILY = ['densenet121','densenet161','densenet169','densenet201']
    VGGNET_MODEL_FAMILY = ['vgg11','vgg13','vgg16','vgg19','vgg11_bn','vgg13_bn','vgg16_bn','vgg19_bn']
    MOBILENET_MODEL_FAMILY = ['mobilenet_v2']

    # def __init__(self, MODEL_PATH):
    #     self.MODEL_PATH = MODEL_PATH

    def initialise_pruning_params(self, config_data):
     
        # First read all required parameters from 
        if (config_data['PRUNE_STRATEGY']=='L1Strategy'):
            self.PRUNE_STRATEGY = tp.strategy.L1Strategy()
        else:
            print('Error: Specify a pruning stategy')
            return 0
        self.MODEL_PATH = config_data['MODEL_PATH']
        self.DEPENDENCY_GRAPH = tp.DependencyGraph()
        self.INPUT_SHAPE = config_data['INPUT_SHAPE']
        self.PRUNE_CONV2D_LAYERS = config_data['PRUNE_CONV2D_LAYERS']
        self.PRUNE_CONV2D_LAYER_PERCENTAGE = config_data['PRUNE_CONV2D_LAYER_PERCENTAGE']
        self.PRUNE_DENSE_LAYERS = config_data['PRUNE_DENSE_LAYERS']
        self.PRUNE_DENSE_LAYER_PERCENTAGE = config_data['PRUNE_DENSE_LAYER_PERCENTAGE']
        self.INPUT_SHAPE = config_data['INPUT_SHAPE']
        self.PRUNE_ROUNDS = config_data['PRUNE_ROUNDS']

    def prune_module(self,module):
        self.DEPENDENCY_GRAPH.build_dependency(self.MODEL, example_inputs=torch.randn(self.INPUT_SHAPE))
        pruning_idxs = self.PRUNE_STRATEGY(module.weight, amount=self.PRUNE_CONV2D_LAYER_PERCENTAGE) # or manually selected pruning_idxs=[2, 6, 9]
        pruning_plan = self.DEPENDENCY_GRAPH.get_pruning_plan( module, tp.prune_conv, idxs=pruning_idxs )
        #print(pruning_plan)
        #tp.prune_conv(model.conv1, )
        pruning_plan.exec()

    def get_model_parameter_count(self):
        params = sum([np.prod(p.size()) for p in self.MODEL.parameters()])
        return params

    

    def start_pruning(self):
        self.MODEL = torch.load(self.MODEL_PATH)
        
        params = self.get_model_parameter_count()
        print("Number of Parameters: %.1fM"%(params/1e6))

        # Prune different modules of the model
        for name,module in self.MODEL.named_modules():
            if (self.PRUNE_CONV2D_LAYERS and isinstance( module, torch.nn.Conv2d )):
                print("\"{}\",".format(name))
                self.prune_module(module)

        params = self.get_model_parameter_count()
        print("Number of Parameters: %.1fM"%(params/1e6))

        torch.save(self.MODEL, 'classification/models/resnet18_normal_pruned.pt')