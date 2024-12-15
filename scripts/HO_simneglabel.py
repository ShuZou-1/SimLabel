import os
import argparse
import numpy as np
import torch
import time
from scipy import stats

from utils.common import setup_seed, get_num_cls, get_test_labels
from utils.detection_util import *
from utils.file_ops import save_as_dataframe, setup_log
from utils.plot_util import plot_distribution
from utils.train_eval_util import  set_model_clip, set_train_loader, set_val_loader, set_ood_loader_ImageNet
# sys.path.append(os.path.dirname(__file__))


def process_args():
    parser = argparse.ArgumentParser(description='Evaluates MCM Score for CLIP',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # setting for each run
    parser.add_argument('--in_dataset', default='ImageNet', type=str,
                        choices=['ImageNet', 'ImageNet10', 'ImageNet20', 'ImageNet100',
                                  'pet37', 'food101', 'car196', 'bird200'], help='in-distribution dataset')
    
    parser.add_argument('--number_sim', default=10, type=int)

    parser.add_argument('--root-dir', default="/home/u6768067/Downloads/datasets", type=str,
                        help='root dir of datasets')
    parser.add_argument('--name', default="eval_ood",
                        type=str, help="unique ID for the run")
    parser.add_argument('--seed', default=5, type=int, help="random seed")
    parser.add_argument('--gpu', default=0, type = int,
                        help='the GPU indice to use')
    parser.add_argument('-b', '--batch-size', default=512, type=int,
                        help='mini-batch size')
    parser.add_argument('--T', type=int, default=1,
                        help='temperature parameter')
    parser.add_argument('--model', default='CLIP', type=str, help='model architecture')
    parser.add_argument('--CLIP_ckpt', type=str, default='ViT-B/16',
                        choices=['ViT-B/32', 'ViT-B/16', 'ViT-L/14'], help='which pretrained img encoder to use')
    parser.add_argument('--score', default='SimLabel', type=str, choices=[
        'MCM', 'energy', 'max-logit', 'entropy', 'var', 'maha','SimLabel'], help='score options')
    # for Mahalanobis score
    parser.add_argument('--feat_dim', type=int, default=512, help='feat dim； 512 for ViT-B and 768 for ViT-L')
    parser.add_argument('--normalize', type = bool, default = False, help='whether use normalized features for Maha score')
    parser.add_argument('--generate', type = bool, default = True, help='whether to generate class-wise means or read from files for Maha score')
    parser.add_argument('--template_dir', type = str, default = 'img_templates', help='the loc of stored classwise mean and precision matrix')
    parser.add_argument('--subset', default = False, type =bool, help = "whether uses a subset of samples in the training set")
    parser.add_argument('--max_count', default = 250, type =int, help = "how many samples are used to estimate classwise mean and precision matrix")
    args = parser.parse_args()

    args.n_cls = get_num_cls(args)
    args.log_directory = f"results/{args.in_dataset}/{args.score}/{args.model}_{args.CLIP_ckpt}_T_{args.T}_ID_{args.name}"
    os.makedirs(args.log_directory, exist_ok=True)

    return args

def obtain_score(image_features,text_features,neg_text_features,sim_class,confidence_class,number_sim,alpha):
    _score = []
    to_np = lambda x: x.data.cpu().numpy()
    output = list()
    with torch.no_grad():
        for image_feature in image_features:
            similarity_ID  = to_np(image_feature@text_features.T)

            similarity_Neg = to_np(image_feature@neg_text_features.T)
            prediction = list()
            for class_index in range(len(text_features)):
                # if confidence_class[class_index][0] == 0:
                #     prediction.append(1/(20/number_sim+1))
                # else:
                pos_affinity  = np.sum(np.multiply(similarity_ID[sim_class[class_index][:number_sim]],confidence_class[class_index]))

                neg_affinity  = np.sum(similarity_Neg[class_index*20:(class_index+1)*20][:number_sim]/number_sim)

                prediction.append(similarity_ID[class_index]+alpha*pos_affinity)
                # prediction.append(similarity_ID[class_index]+ 4*(pos_affinity-neg_affinity))


                    # prediction.append(pos_affinity/(pos_affinity+neg_affinity))
            output.append(np.array(prediction))


    # for index in range(len(output)):
    #     prediction = output[index]

    #     _score.append(-np.max(prediction))  

    for index in range(len(output)):
        prediction = output[index]
        e_x = np.exp(prediction - np.max(prediction))
        prediction = e_x / e_x.sum(axis=0)

        _score.append(-np.max(prediction))  
    _score = np.array(_score)
    return _score

def obtain_HO_score(image_features,text_features,neg_text_features,sim_class,confidence_class,neg_class,  neg_confidence_class,number_sim):
    _score = []
    to_np = lambda x: x.data.cpu().numpy()
    output = list()
    with torch.no_grad():
        for image_feature in image_features:
            similarity_ID  = to_np(image_feature@text_features.T)

            similarity_Neg = to_np(image_feature@neg_text_features.T)

            prediction = list()
            for class_index in range(len(text_features)):
                # if confidence_class[class_index][0] == 0 or neg_confidence_class[class_index][0]==0 :
                #     prediction.append(1/2)
                # else:
                pos_affinity  = np.sum(np.multiply(similarity_ID[sim_class[class_index]],confidence_class[class_index]))

                neg_affinity  = np.sum(np.multiply(similarity_Neg[neg_class[class_index]],neg_confidence_class[class_index]))

                # neg_affinity  = np.sum(similarity_Neg[class_index*20:(class_index+1)*20][:number_sim]/number_sim)

                # prediction.append(similarity_ID[class_index]+pos_affinity)
                # prediction.append(pos_affinity/(pos_affinity+neg_affinity))
                prediction.append(similarity_ID[class_index]+4*pos_affinity-neg_affinity)


                    # prediction.append(pos_affinity/(pos_affinity+neg_affinity))
            output.append(np.array(prediction))


    # for index in range(len(output)):
    #     prediction = output[index]

    #     _score.append(-np.max(prediction))  

    for index in range(len(output)):
        prediction = output[index]
        e_x = np.exp(prediction - np.max(prediction))
        prediction = e_x / e_x.sum(axis=0)

        _score.append(-np.max(prediction))  
    _score = np.array(_score) 
    return _score

def main():
    args = process_args()
    setup_seed(args.seed)
    log = setup_log(args)
    assert torch.cuda.is_available()
    torch.cuda.set_device(args.gpu)
    device = torch.device("cuda")

    net, preprocess = set_model_clip(args)
    net.eval()

    if args.in_dataset in ['ImageNet10']: 
        out_datasets = ['ImageNet20']
    elif args.in_dataset in ['ImageNet20']: 
        out_datasets = ['ImageNet10']
    elif args.in_dataset in [ 'ImageNet', 'ImageNet100', 'bird200', 'car196', 'food101', 'pet37']:
         out_datasets = ['iNaturalist','SUN', 'places365', 'dtd']
    test_loader = set_val_loader(args, preprocess)
    test_labels = get_test_labels(args, test_loader)

    to_np = lambda x: x.data.cpu().numpy()
    concat = lambda x: np.concatenate(x, axis=0)

    image_features = torch.load(f'features/image_feature_{args.in_dataset}.pt', map_location=device)
    text_features = torch.load(f'features/text_feature_{args.in_dataset}.pt', map_location=device)
    neg_text_features = torch.load(f'features/full_neg_text_feature_{args.in_dataset}.pt', map_location=device)

    number_sim = args.number_sim

    sim_class,   confidence_class     = get_simclass(args,number_sim=number_sim)

    neg_class,   neg_confidence_class = get_negclass(args,number_sim=number_sim//3)

    for index in range(len(10)):
        alpha = index/2
        in_score = obtain_score(image_features,text_features,neg_text_features,sim_class,confidence_class,number_sim,alpha)
        # in_score = obtain_HO_score(image_features,text_features,neg_text_features,sim_class,confidence_class,neg_class,   neg_confidence_class,number_sim)

        print(in_score.shape)
        auroc_list, aupr_list, fpr_list = [], [], []
        for out_dataset in out_datasets:
            log.debug(f"Evaluting OOD dataset {out_dataset}")
            ood_image_features = torch.load(f'features/image_feature_{out_dataset}.pt', map_location=device)
            out_score = obtain_score(ood_image_features,text_features,neg_text_features,sim_class,confidence_class,number_sim,alpha)
            # out_score = obtain_HO_score(ood_image_features,text_features,neg_text_features,sim_class,confidence_class,neg_class,   neg_confidence_class,number_sim)


            # log.debug(f"in scores: {stats.describe(in_score)}")
            # log.debug(f"out scores: {stats.describe(out_score)}")
            plot_distribution(args, in_score, out_score, out_dataset)
            get_and_print_results(args, log, in_score, out_score,
                                auroc_list, aupr_list, fpr_list)
        log.debug('\n\nMean Test Results')
        print_measures(log, np.mean(auroc_list), np.mean(aupr_list),
                    np.mean(fpr_list), method_name=args.score)
        save_as_dataframe(args, out_datasets, fpr_list, auroc_list, aupr_list)



if __name__ == '__main__':
    main()
