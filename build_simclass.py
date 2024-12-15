import os
import argparse
import numpy as np
import torch
from scipy import stats
from transformers import CLIPTokenizer
from collections import Counter
import operator
import json
from tqdm import tqdm

from utils.common import setup_seed, get_num_cls, get_test_labels
from utils.detection_util import get_Mahalanobis_score, get_mean_prec, print_measures, get_and_print_results, get_ood_scores_clip, get_ood_scores_SimLabel, get_simclass,get_classwise_negLabel
from utils.file_ops import save_as_dataframe, setup_log
from utils.plot_util import plot_distribution
from utils.train_eval_util import  set_model_clip, set_train_loader, set_val_loader, set_ood_loader_ImageNet
# sys.path.append(os.path.dirname(__file__))


def process_args():
    parser = argparse.ArgumentParser(description='Evaluates MCM Score for CLIP',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--in_dataset', default='ImageNet', type=str,
                    choices=['ImageNet', 'ImageNet10', 'ImageNet20', 'ImageNet100',
                                'pet37', 'food101', 'car196', 'bird200'], help='in-distribution dataset')
    parser.add_argument('--number_sim', default=20, type=int)
    
    parser.add_argument('--root-dir', default="datasets", type=str,
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

def main():
    args = process_args()
    setup_seed(args.seed)
    log = setup_log(args)
    assert torch.cuda.is_available()
    torch.cuda.set_device(args.gpu)

    net, preprocess = set_model_clip(args)
    net.eval()

    test_loader = set_val_loader(args, preprocess)
    test_labels = get_test_labels(args, test_loader)
    

    to_np = lambda x: x.data.cpu().numpy()
    concat = lambda x: np.concatenate(x, axis=0)
    _score = []
    tokenizer = CLIPTokenizer.from_pretrained(args.ckpt)

    tqdm_object = tqdm(test_loader, total=len(test_loader))
    device = torch.device("cuda")



    if args.model == 'CLIP':
        text_inputs = tokenizer([f"a photo of a {c}" for c in test_labels], padding=True, return_tensors="pt")
        text_features = net.get_text_features(input_ids = text_inputs['input_ids'].cuda(), 
                                        attention_mask = text_inputs['attention_mask'].cuda()).float()
        text_features /= text_features.norm(dim=-1, keepdim=True) 

    image_features = None
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm_object):
            bz = images.size(0)
            labels = labels.long().cuda()
            images = images.cuda()
            
            image_features_batch = net.get_image_features(pixel_values = images).float()
            image_features_batch /= image_features_batch.norm(dim=-1, keepdim=True)

            if image_features == None:
                image_features = image_features_batch
            else:
                image_features = torch.cat((image_features,image_features_batch),0)



    output = image_features @ text_features.T
    output = to_np(output)
 

    max_index = np.argmax(output, axis=1)
    whole_dict = dict()
    dif_dict   = dict() 

    for index in range(len(test_labels)):
        sim_perclass  = output[[x for x in range(len(output)) if max_index[x]==index]]
        
        similar_classes = list([])
        rev_similar_classes = list([])
        
        for sim in sim_perclass:
            class_dict = {i: sim[i] for i in range(len(sim ))}
            sort_class = list(reversed(sorted(class_dict.items(), key=operator.itemgetter(1))))
            rev_class  = list((sorted(class_dict.items(), key=operator.itemgetter(1))))
            
            k = 20
            sim_class = [i[0] for i in sort_class[:k]]

            un_k = 20
            rev_sim_class = [i[0] for i in rev_class[:un_k]]

            similar_classes       = similar_classes      + sim_class
            rev_similar_classes   = rev_similar_classes  + rev_sim_class

        
        sim_class_dict = dict(Counter(similar_classes))
        rev_class_dict = dict(Counter(rev_similar_classes))

        # sort_class = list(reversed(sorted(sim_class_dict.items(), key=operator.itemgetter(1))))[:20]
        # rev_sort_C = list(reversed(sorted(rev_class_dict.items(), key=operator.itemgetter(1))))[:20]

        whole_dict[index] = [{
            'index': key,
            "occurance": value
            } for key, value in sim_class_dict.items()]
        
        dif_dict[index] = [{
            'index': key,
            "occurance": value
            } for key, value in rev_class_dict.items()]
        

        # record all the similar class
    j = json.dumps(whole_dict, indent=4)
    with open("result/sim_class_"+args.in_dataset+".json", 'w') as f:
        print(j, file=f)

    j = json.dumps(dif_dict, indent=4)
    with open("result/dis_sim_class_"+args.in_dataset+".json", 'w') as f:
        print(j, file=f)

if __name__ == '__main__':
    main()
