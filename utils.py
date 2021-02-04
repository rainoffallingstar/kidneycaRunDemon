""" helper function

author baiyu
"""
import os
import sys
import re
import datetime
import numpy



def get_network(args):
    """ return given network
    """

    if args.net == 'densenet':
        from models.densenet import densenet
        net = densenet()
    elif args.net == 'inceptionV1':
        from models.inceptionV1 import inceptionV1
        net = inceptionV1()
    elif args.net == 'inceptionV4':
        from models.inceptionV4 import inceptionV4
        net = inceptionV4()
    elif args.net == 'model_resnet152':
        from models.model_resnet152 import model_resnet152
        net = model_resnet152()
    elif args.net == 'modelv2':
        from models.modelv2 import modelv2
        net = modelv2()
    elif args.net == 'inception_resnet_v1':
        from models.inception_resnet_v1 import inception_resnet_v1
        net = inception_resnet_v1()
    elif args.net == 'inceptionV2':
        from models.inceptionV2 import inceptionV2
        net = inceptionV2()
    elif args.net == 'model_original':
        from models.model_original import model_original
        net = model_original()
    elif args.net == 'model_resnet50':
        from models.model_resnet50 import model_resnet50
        net = model_resnet50()
    

    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    if args.gpu: #use_gpu
        net = net.cuda()

    return net


