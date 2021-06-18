import os
import yaml

if __name__ == '__main__':
        modelname = 'resnet18'
        s = os.system("python3 classification/main.py --config data_cfg/classification-config.yaml --model_arch {}".format(modelname))
    
