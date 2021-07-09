import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import yaml
import argparse
import csv

import torch

from data_loader import ImageClassificationDataLoader
from transforms import ImageClassifcationTransforms
from train import ImageClassificationTrain
from amp_train import ImageClassificationMixedPrecisionTraining
from model import ImageClassificationModel
from pruning import Pruning
from test import ImageClassificationTest

#DATA_PATH = './data'
# MODEL_PATH = 'resnet50/models/'
# INTERIM_MODEL_NAME = 'interim_model.pt'
# FINAL_MODEL_NAME = 'final_model.pt'

#CONFIG_PATH = 'cfg/default-config.yaml'

def train(args):
    print(args)
    CONFIG_PATH = args.config
    DATA_PATH = args.data
    MODEL_ARCH = args.model_arch
    TRAIN_DIR = os.path.join(DATA_PATH,'train')
    TEST_DIR = os.path.join(DATA_PATH,'test')
    USE_GPU = args.use_gpu
    MODE = args.mode


    with open(CONFIG_PATH) as file:
        config_data= yaml.safe_load(file)

    BATCH_SIZE = config_data['batch_size']
    EPOCHS = config_data['epochs']
    LEARNING_RATE = config_data['learning_rate']
    MOMENTUM = config_data['momentum']
    INTERIM_MODEL_PATH = os.path.join(config_data['model_path'] ,config_data['interim_model_name'])
    FINAL_MODEL_PATH = os.path.join(config_data['model_path'] ,config_data['final_model_name'])
    CLASSIFICATION_NUM_HIDDEN_LAYERS = config_data['classifer_num_hidden_layers']
    CLASSIFIER_LAYER_SIZE = config_data['classifier_layer_size']
    CLASSIFIER_LAYER_DROPOUT_PROB = config_data['classifier_layer_dropout_prob']
    TRAIN_VAL_SPLIT = config_data['train_val_split']

    # change made
    with open('hyperparameters.csv', 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames = ['train_val_split', 'epochs', 'batch_size', 'learning_rate', 'momentum'])
        writer.writeheader()
        writer.writerow({
            'train_val_split': TRAIN_VAL_SPLIT, 
            'epochs' : EPOCHS, 
            'batch_size' : BATCH_SIZE, 
            'learning_rate' : LEARNING_RATE, 
            'momentum' : MOMENTUM})
    # change made

    if(USE_GPU):
        if(torch.cuda.is_available()):
            print("Training on GPU \n")
            device = torch.device("cuda")
        else:
            print("GPU support not available")
            return 0
    else:
        print("Training on CPU \n")
        device = torch.device("cpu")

    if not (os.path.exists(config_data['model_path'])):
        os.mkdir(config_data['model_path'])

    print(DATA_PATH)
    print("Data directories: ",os.listdir(DATA_PATH),"\n")

    #Image Transforms
    image_transform_obj = ImageClassifcationTransforms(TRAIN_DIR, TEST_DIR)
    train_transforms, test_transforms, val_transforms = image_transform_obj.get_transforms()

    #Create Torch data loader
    class_data_loader_obj = ImageClassificationDataLoader(TRAIN_DIR, TEST_DIR, BATCH_SIZE,train_transforms, test_transforms, val_transforms, TRAIN_VAL_SPLIT)
    train_loader, val_loader, test_loader = class_data_loader_obj.get_data_loader()
    NUM_CLASSES = class_data_loader_obj.get_num_classes()

    #Model architecture
    model_obj = ImageClassificationModel(NUM_CLASSES, CLASSIFICATION_NUM_HIDDEN_LAYERS, CLASSIFIER_LAYER_SIZE, CLASSIFIER_LAYER_DROPOUT_PROB, MODEL_ARCH)
    model = model_obj.initialise_model(INTERIM_MODEL_PATH)

    if (MODE =='train'):
        print("Normal Training")
        train_obj = ImageClassificationTrain(model, device, EPOCHS, train_loader, val_loader, INTERIM_MODEL_PATH, FINAL_MODEL_PATH)
        train_obj.initiate_training()

    if (MODE == 'amp_train'):
        print("Mixed precision training")
        train_obj = ImageClassificationMixedPrecisionTraining(model, device, EPOCHS, train_loader, val_loader, INTERIM_MODEL_PATH, FINAL_MODEL_PATH)
        train_obj.initiate_training()

    if (MODE == 'prune'):
        print("Pruning")
        prune_obj = Pruning()
        prune_obj.initialise_pruning_params(config_data)
        prune_obj.start_pruning()

    if (MODE == 'test'):
        print("Testing")
        model = torch.load(FINAL_MODEL_PATH)
        test_obj = ImageClassificationTest(model, device, test_loader)
        test_obj.initiate_testing()

    return



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='data_cfg/classification-config.yaml')
    parser.add_argument('--data', default='data_cfg/data/')
    parser.add_argument('--model_arch')
    parser.add_argument('--use_gpu', default=1, type=int)
    parser.add_argument('--mode', default='train', type=str)
    args = parser.parse_args()
    train(args)
