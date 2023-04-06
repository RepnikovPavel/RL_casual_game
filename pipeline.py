import os
import config
import torchvision
import torch

import model_using
import support_func
import time_mesuament
import sys

import managers
import img_processing
import random
import numpy as np
import pprint
from torchsummary import summary


from PIL import Image



if __name__ == '__main__':
    timer = time_mesuament.Timer()
    timer.start()

    pretrained_model = config.img_models['resnet152v1']
    test_image_path = os.path.join(config.base_row_data_path, 'test_image.png')




    models_manager = managers.ModelsManager()
    model = models_manager.load_model(description_of_the_model=pretrained_model,
                                      reload_from_internet=False)

    embeddings = []

    # pprint.pprint(summary(model))
    # print(model)

    model_controller = model_using.ModelController(model=model)
    gpu_device = torch.device('cuda', index=0)
    cpu_device = torch.device('cpu', index=0)

    image_handler = img_processing.ImageHandler()
    img = Image.open(test_image_path)
    preprocessed_img_as_tensor = image_handler.prepare_row_PIL_image_to_work_with_resnet(img)

    X_batch = [preprocessed_img_as_tensor]
    model_controller.eval_model_on_the_list_of_elements(input_storage=X_batch,
                                                        output_storage=embeddings,
                                                        output_storage_device=cpu_device,
                                                        run_on_device=gpu_device,
                                                        batch_size=1000)
    pprint.pprint(embeddings)

    timer.stop()
    timer.print_execution_time()
