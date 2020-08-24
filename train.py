import argparse
import logging
import os
import sys
import timeit

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from unet import *

from dataset import WMHChallengeDataset
from tversky_loss import *

from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from tqdm import tqdm, trange
from data_util import Utrecht_postprocessing, GE3T_postprocessing
from evaluation import getDSC, getHausdorff, getLesionDetection, getAVD, getImages, getAUC

import SimpleITK as sitk
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

import matplotlib.pyplot as plt
#set up logger for outputing logs and writer for tensorboardX
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
tb_writer = SummaryWriter("experiments")

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=5,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch_size', metavar='B', type=int, nargs='?', default=1,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.1,
                        help='Learning rate', dest='lr')
    parser.add_argument('-g', '--number_of_gpu', dest='n_gpu', type=int, default=1,
                        help='Number of GPU')
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument("--save_dir", type=str, default="",
                        help="directory for saving the model")
    parser.add_argument("--data_dir", type=str, default="",
                        help="directory for raw data")
    parser.add_argument("--output_dir", type=str, default="",
                        help="directory for output results")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--model_dir", type=str, default="",
                        help="Model directory for loading")
    parser.add_argument("--test_subject", type=int, default=59,  nargs='+',
                        help="subject id for leave-one-out testing")
    parser.add_argument('--seed', type=int, default=25,
                    help='random seed')
    parser.add_argument('--model_type', type=int, default=0,
                    help='model type for training, 0 for regular unet, 1 for attention unet, 2 for recurrent unet, 3 for domainnet, 4 for domainnetX, 5 for domainXR, 6 for unet++')
    parser.add_argument('--eval_epochs', type=int, default=10,
                    help='interval in epochs for doing evaluation during training')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--silent', action='store_true',
                        help="whether to show progress bar for tqdm")
    parser.add_argument('--domain_knowledge', action='store_true',
                        help="whether to use altas(domain knowledge) as additional input")
    parser.add_argument('--aging', action='store_true',
                        help="whether to use aging atlas as additional input")
    parser.add_argument('--large', action='store_true',
                        help="whether to use more attention channels for domain net")
    parser.add_argument('--thresh_hold', type=float, default=0.5,
                        help="threshhold value for segmentation")
    parser.add_argument('--T1', action='store_true',
                        help="whether to use T1 as input")
    parser.add_argument('--alpha', type=float, default=0.7,
                        help="alpha value for various loss functions")
    parser.add_argument('--abvib', action='store_true',
                        help="whether to use the ABVIB dataset for training")
    parser.add_argument('--abvib_eval', action='store_true',
                        help="whether to evaluate on ABVIB")
    parser.add_argument('--weight_decay', type=float, default=0,
                        help="regularization term")
    parser.add_argument('--kernel', type=int, default=5,
                        help='kernel size for BAGAU-Net')
    return parser.parse_args()

def tversky_loss(y_pred, y_true, alpha):
    return 1 - tversky_coeff(y_pred, y_true, alpha=alpha)

def train(model, 
          device,
          dataset, 
          args):
    try:
        os.mkdir(args.save_dir)
        logging.info('Created model directory')
    except OSError:
        pass
    train_sampler = RandomSampler(dataset) if args.local_rank == -1 else DistributedSampler(dataset)
    train_dataloader = DataLoader(dataset, sampler=train_sampler, batch_size=args.batchsize)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    #preparing eval and test dataset
    if len(args.test_subject) == 1:
        args.test_subject = int(args.test_subject[0])
    if isinstance(args.test_subject, int):
        if args.abvib:
            eval_dataset = ABVIBDataset(directory=args.data_dir, train=False, test_subject=args.test_subject, domain_knowledge=args.domain_knowledge, aging=args.aging)
        else:
            eval_dataset = WMHChallengeDataset(directory=args.data_dir, train=False, test_subject=args.test_subject, aug=True, domain_knowledge=args.domain_knowledge, aging=args.aging, T1=args.T1)
    else:
        args.test_subject = [int(x) for x in args.test_subject]
        eval_dataset1, eval_dataset2 = [], []
        for idx, item in enumerate(args.test_subject):
            if args.abvib:
                item_dataset = ABVIBDataset(directory=args.data_dir, train=False, test_subject=item, domain_knowledge=args.domain_knowledge, aging=args.aging)
            else:
                item_dataset = WMHChallengeDataset(directory=args.data_dir, train=False, test_subject=item, aug=True, domain_knowledge=args.domain_knowledge, aging=args.aging, T1=args.T1)
            if idx % 2 == 0:
                eval_dataset1.append(item_dataset)
            else:
                eval_dataset2.append(item_dataset)
    test_subject = args.test_subject
    #loss function
    criterion = tversky_loss

    if args.abvib:
        standard = evaluateABVIB
        dataset_name = 'ABVIB'
    else:
        standard = evaluate
        dataset_name = 'WMH'

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Num Epochs = %d", args.epochs)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                    args.batchsize * args.gradient_accumulation_steps)
    logger.info("  Test Subject = %s", str(args.test_subject))
    logger.info("  Loss function : %s", str(criterion).split()[1])
    logger.info("  Dataset : %s", dataset_name)

    model.zero_grad()
    train_iterator = trange(int(args.epochs), desc="Epoch", disable=args.silent)
    epoch, global_step = 0, 0
    best_auc, curr_auc = 0, 0
    best_dsc, curr_dsc = 0, 0
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.silent)
        epoch += 1
        epoch_loss = 0
        for step, batch in enumerate(epoch_iterator):
            model.train()
            imgs, masks = batch['image'], batch['mask']
            imgs = imgs.to(device=device)
            masks = masks.to(device=device)
            masks_pred = model(imgs)
            loss = criterion(masks_pred, masks, alpha=args.alpha)

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()
            epoch_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                #add loss to tensorboardX
                tb_writer.add_scalar('loss', loss*args.gradient_accumulation_steps, global_step)

        #evalution during training
        if epoch % args.eval_epochs == 0:
            if isinstance(test_subject, int):
                try:
                    dsc, avd, recall, f1, auc = standard(model, device, eval_dataset, args, training=True)
                except:
                    #model has yet ready for prediction
                    logger.info('Evaluation skipped')
                    continue
                tb_writer.add_scalar('dsc', dsc, epoch)
                tb_writer.add_scalar('avd', avd, epoch)
                tb_writer.add_scalar('recall', recall, epoch)
                tb_writer.add_scalar('f1', f1, epoch)
                tb_writer.add_scalar('auc', auc, epoch)
                logger.info('-' * 100)
                log_str = '| Eval {:3d} at epoch {:>8d} ' \
                        '| dsc {:5.2f} | avd {:5.2f} | recall {:5.2f} ' \
                        '| f1 {:5.2f} | auc {:5.2f} '.format(
                    epoch // args.eval_epochs, epoch, dsc, avd, recall, f1, auc)
                logger.info(log_str)
                logger.info('-' * 100)
                curr_auc = auc
                curr_dsc = dsc
            else:
                results = np.zeros(5)
                for i in range(len(eval_dataset1)):
                    try:
                        args.test_subject = eval_dataset1[i].test_subject
                        metrics = standard(model, device, eval_dataset1[i], args, training=True)
                        results += np.asarray(metrics)
                    except:
                        logger.info('Evaluation skipped')
                        continue
                results /= len(eval_dataset1)
                logger.info('-' * 110)
                log_str = '| Eval1 {:3d} at epoch {:>8d} ' \
                        '| dsc {:7.4f} | avd {:7.4f} | recall {:7.4f} ' \
                        '| f1 {:7.4f} | auc {:7.4f} '.format(
                    epoch // args.eval_epochs, epoch, results[0], results[1], results[2], results[3], results[4])
                logger.info(log_str)
                logger.info('-' * 110)
                curr_auc = results[4]
                curr_dsc = results[0]

                results = np.zeros(5)
                for i in range(len(eval_dataset2)):
                    try:
                        args.test_subject = eval_dataset2[i].test_subject
                        metrics = standard(model, device, eval_dataset2[i], args, training=True)
                        results += np.asarray(metrics)
                    except:
                        logger.info('Evaluation skipped')
                        continue
                results /= len(eval_dataset2)
                logger.info('-' * 110)
                log_str = '| Eval2 {:3d} at epoch {:>8d} ' \
                        '| dsc {:7.4f} | avd {:7.4f} | recall {:7.4f} ' \
                        '| f1 {:7.4f} | auc {:7.4f} '.format(
                    epoch // args.eval_epochs, epoch, results[0], results[1], results[2], results[3], results[4])
                logger.info(log_str)
                logger.info('-' * 110)
                curr_auc = results[4]
                curr_dsc = results[0]
            #save the model when it achieves highest auc or highest dsc
            if curr_auc > best_auc:
                torch.save(model.state_dict(),
                    args.save_dir + f'/fold_{args.fold}.pth')
                logger.info(f'model saved !')
                best_auc = curr_auc
            if curr_dsc > best_dsc:
                torch.save(model.state_dict(),
                    args.save_dir + f'/dsc_fold_{args.fold}.pth')
                logger.info(f'model saved !')
                best_dsc = curr_dsc
     
        logger.info("  Epoch = {}, loss = {}".format(epoch, epoch_loss/(step+1)))
    if args.do_train:
        torch.save(model.state_dict(),
            args.save_dir + f'/model.pth')
    tb_writer.close()

def evaluate(model, 
         device,
         dataset, 
         args,
         training=False):
    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    eval_sampler = SequentialSampler(dataset) if args.local_rank == -1 else DistributedSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.batchsize)
    if not training and not args.thresh_hold_finding:
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(dataset))
        logger.info("  Batch size = %d", args.batchsize)

    start_time = timeit.default_timer()
    pred = torch.tensor([], dtype=torch.float32, device=device)

    for batch in tqdm(eval_dataloader, desc="Evaluating", disable=args.silent):
        model.eval()
        with torch.no_grad():
            imgs, masks = batch['image'], batch['mask']
            imgs = imgs.to(device=device)
            masks = masks.to(device=device)
            masks_pred = model(imgs).squeeze(dim=1)
            masks_pred = (masks_pred > args.thresh_hold).float()
            pred = torch.cat((pred, masks_pred), dim=0)

    if torch.cuda.is_available():
        pred = pred.cpu()
    
    pred = torch.unsqueeze(pred, -1).numpy()
    
    if args.test_subject < 20: 
        original = GE3T_postprocessing(dataset.eval_mask, pred)
    else: 
        original = Utrecht_postprocessing(dataset.eval_mask, pred)
    filename_resultImage = args.output_dir + f'/{args.test_subject}.nii.gz'
    sitk.WriteImage(sitk.GetImageFromArray(original), filename_resultImage)
    testImage, resultImage = getImages(dataset.eval_dir, filename_resultImage)
    dsc = getDSC(testImage, resultImage)
    avd = getAVD(testImage, resultImage) 
    #h95 distance is comment out due to numeric issues with python3 with evalution script
    #h95 = getHausdorff(testImage, resultImage)
    recall, f1 = getLesionDetection(testImage, resultImage)
    auc = getAUC(dataset.eval_dir, filename_resultImage)
    
    if not training and not args.thresh_hold_finding:
        evalTime = timeit.default_timer() - start_time
        logger.info("  Evaluation done in total %f secs (%f sec per example)", evalTime, evalTime / len(dataset))
        logger.info("***** Evaluation Results *****")
        logger.info(f'Result of patient {args.test_subject}')
        logger.info(f'Dice {dsc} (higher is better, max=1)')
        #logger.info(f'HD {h95}mm (lower is better, min=0)')
        logger.info(f'AVD {avd}% (lower is better, min=0)')
        logger.info(f'Lesion detection {recall} (higher is better, max=1)')
        logger.info(f'Lesion F1 {f1} (higher is better, max=1)')
        logger.info(f'AUC {auc} (higher is better, max=1)')
    return dsc, avd, recall, f1, auc

if __name__ == '__main__':
    args = get_args()
    # Set the random seed manually for reproducibility.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device {device}')
    
    channel = 1 + args.domain_knowledge + args.aging + args.T1

    if args.model_type == 0:
        net = UNet(n_channels=channel, n_classes=1)
        model_type = "U-net"
    else:
        net = Dk_UNet_XR(n_channels=channel, n_classes=1, aging=args.aging, size=args.kernel)
        model_type = "BAGAU-Net"
   
    logger.info(f'Network: {model_type}\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Dilated conv"} upscaling')
      
    if args.model_dir:
        net.load_state_dict(
            torch.load(args.model_dir, map_location=device)
        )
        logger.info(f'Model loaded from {args.model_dir}')
    net.to(device=device)
    logger.info("***** Preparing dataset *****")
    
    if args.do_train:
        if args.abvib:
            train_dataset = ABVIBDataset(directory=args.data_dir, train=True, test_subject=args.test_subject, domain_knowledge=args.domain_knowledge, aging=args.aging)
        else:
            train_dataset = WMHChallengeDataset(directory=args.data_dir, train=True, test_subject=args.test_subject, aug=True, domain_knowledge=args.domain_knowledge, aging=args.aging, T1=args.T1)
        train(net, device, train_dataset, args)
    if args.do_eval:
        args.test_subject = args.test_subject[0]
        eval_dataset = WMHChallengeDataset(directory=args.data_dir, train=False, test_subject=args.test_subject, aug=True, domain_knowledge=args.domain_knowledge, aging=args.aging, T1=args.T1)
        evaluate(net, device, eval_dataset, args)