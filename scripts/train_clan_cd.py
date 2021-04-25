# some_file.py
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../')
import argparse
import csv
import json
import os
import time
from copy import deepcopy
from datetime import datetime

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch import nn
from torch.utils.data import DataLoader

from helper_utils.EarlyStopping import EarlyStopping
from helper_utils.data_list import LabeledDataset
from helper_utils.logger import Logger
from helper_utils.network import SentimentClassifier, Discriminator, GradReverse
import helper_utils.vocab

languages = ['en', 'fr', 'de', 'ja']
domains = ['books', 'dvd', 'music']


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--trail', type=str, help="trail number")
    parser.add_argument('--gpu_id', type=int, nargs='?', help="device id to run")


    parser.add_argument('--source_lang', choices=languages, help='source languages')
    parser.add_argument('--target_lang', choices=languages, help='target languages')
    parser.add_argument('--source_domain', choices=domains, help='source domains')
    parser.add_argument('--target_domain', choices=domains, help='target domains')

    parser.add_argument('--unlabeled', help='unlabeled data - tokenized and binarized')
    parser.add_argument('--train', help='training data - tokenized and binarized')
    parser.add_argument('--val', help='validation data - tokenized and binarized')
    parser.add_argument('--test',  help='test data - tokenized and binarized')

    parser.add_argument('--lambd_lm', type=float, help='coefficient of the language modeling')
    parser.add_argument('--lambd_dis', type=float,  help='coefficient of the adversarial loss')
    parser.add_argument('--lambd_clf', type=float, help='coefficient of the classification loss')

    parser.add_argument('--num_iterations', type=int, help='upper step limit')
    parser.add_argument('-bs', '--batch_size', type=int, help='batch size')
    parser.add_argument('-cbs', '--train_batch_size', type=int,  help='training batch size')
    parser.add_argument('-tbs', '--test_batch_size', type=int, help='testing batch size')
    parser.add_argument('--bptt', type=int,  help='sequence length')

    parser.add_argument('-lr', '--lr', type=float,  help='initial learning rate ')

    parser.add_argument('--seed', type=int,  help='random seed')
    parser.add_argument('--test_interval', type=int,  metavar='N', help='validation and logging interval')
    parser.add_argument('--snapshot_interval', type=int)

    parser.add_argument('--patience', type=int, metavar='N', help='Early stopping patience')



















    parser.set_defaults(
        gpu_id=0,
        trial='CLAN',
        # trial='6',




        source_lang='en',
        target_lang='de',
        source_domain='dvd',
        target_domain='books',
        bptt=70,
        test_batch_size=256,
        train_batch_size=20,
        batch_size=30,
        lr=0.003,
        test_interval=200,
        patience=3000,
        # patience=2,
        seed =0,
        num_iterations=50000,
        snapshot_interval=5000,
        lambd_clf=0.01,
        lambd_dis= 0.1,
        lambd_lm=1,

        unlabeled='unlabeled.pth.tar',
        train='train.pth.tar',
        val='val.pth.tar',
        test='test.pth.tar',

    )
    args = parser.parse_args()

    return args


def set_deterministic_params(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_msg(msg,outfile):
    print()
    print("=" * 50)
    print(" " * 15, msg)
    print("=" * 50)
    print()


    outfile.write('\n')
    outfile.write("=" * 25)
    outfile.write(" " * 5 +  msg)
    outfile.write("=" * 25)
    outfile.write('\n')
    outfile.flush()



def make_batches(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    size = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, size * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data  # shape (size, batch_size)

def data_setup(config, args):
    print_msg("DATASETS Loading ...", config['out_file'])
    s_l, s_d, t_l, t_d = args.source_lang, args.source_domain, args.target_lang, args.target_domain
    batch_size = args.batch_size
    senti_batch_size = args.train_batch_size
    batch_size_adaptation = args.test_batch_size

    unlabeled_data = torch.load(config['unlabeled_data_path'])
    train_data = torch.load(config['train_data_path'])
    val_data = torch.load(config['val_data_path'])
    test_data = torch.load(config['test_data_path'])


    train_x, train_y, train_l = train_data[s_l][s_d]
    val_x, val_y, val_l = val_data[t_l][t_d]
    test_x, test_y, test_l = test_data[t_l][t_d]

    train_x, train_y, train_l = train_x.cuda(), train_y.cuda(), train_l.cuda()
    val_x, val_y, val_l  = val_x.cuda(), val_y.cuda(), val_l.cuda()
    test_x, test_y, test_l = test_x.cuda(), test_y.cuda(), test_l.cuda()


    dset_loaders ={}
    dsets ={}
    dsets['train_s'] = LabeledDataset(train_x, train_y, train_l)
    dsets['val_t'] =  LabeledDataset(val_x, val_y, val_l)
    dsets['test_t'] =  LabeledDataset(test_x, test_y, test_l)
    dset_loaders['train_s']  = DataLoader(dsets['train_s'], batch_size=args.train_batch_size)
    dset_loaders['train_s-intest']  = DataLoader( dsets['train_s'], batch_size=args.test_batch_size)
    dset_loaders['val_t'] = DataLoader(dsets['val_t'], batch_size=args.test_batch_size)
    dset_loaders['test_t'] = DataLoader( dsets['test_t'] , batch_size=args.test_batch_size)

    s_vocab = train_data[s_l]['vocab']
    t_vocab = train_data[t_l]['vocab']
    l_d_pairs = [[s_l, s_d], [s_l, t_d], [t_l, t_d]]
    id_pairs = [[0, 0], [0, 1], [1, 1]]

    unlabeled = [make_batches(unlabeled_data[lang][dom], batch_size).cuda() for lang, dom in l_d_pairs]

    vocab = {'s': s_vocab, 't': t_vocab}
    print_msg("DATASETS loaded ", config['out_file'])

    return dset_loaders, vocab, [unlabeled, l_d_pairs, id_pairs]


def network_setup(config, args, vocab):
    model = SentimentClassifier(n_classes=2, n_langs=2, n_doms=2,
                                vocab_sizes=[len(vocab['s']), len(vocab['t'])])


    dis = Discriminator(input_size=600, hidden_size=400, output_size=2, num_layers=2)



    params = [{'params': model.models.parameters(), 'lr': args.lr},
              {'params': model.clfs.parameters(), 'lr': args.lr}]

    lm_opt = torch.optim.Adam(params, lr=config['lr'] , weight_decay=1.2e-6, betas=(0.7, 0.999))
    dis_opt = torch.optim.Adam(dis.parameters(), lr=config['lr'] , weight_decay=1.2e-6,
                               betas=(0.7, 0.999))

    model.cuda(), dis.cuda()


    # crit = crit.cuda()
    # gpus =[0,1]
    # if len(gpus) > 1:
    #     dis = nn.DataParallel(dis, device_ids=[int(i) for i in gpus])
    #     model = nn.DataParallel(model, device_ids=[int(i) for i in gpus])
    #     crit = nn.DataParallel(crit, device_ids=[int(i) for i in gpus])
    models = {'base': model, 'dis': dis}
    optims = {'lm_opt': lm_opt, 'dis_opt': dis_opt}

    print('Parameters:')
    total_params = sum([np.prod(x.size()) for x in model.parameters()])
    print('\ttotal params:   {}'.format(total_params))
    print('\tparam list:     {}'.format(len(list(model.parameters()))))
    for name, x in model.named_parameters():
        print('\t' + name + '\t', tuple(x.size()))
    for name, x in dis.named_parameters():
        print('\t' + name + '\t', tuple(x.size()))
    print()
    return models, optims



def test(loader, model, num_classes, logs_path, l_id, d_id, num_iterations=0,
         is_training=True, ret_cm=False):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            langs = data[2]

            inputs = inputs.cuda()
            langs = langs.cuda()
            # labels = labels.cuda()
            # _, pred = model(inputs, langs, l_id, d_id).max(-1)
            raw_outputs = model(inputs, langs, l_id, d_id)
            outputs = nn.Softmax(dim=1)(raw_outputs)
            if start_test:
                all_output = outputs.cpu()
                all_label = labels.cpu()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.cpu()), 0)
                all_label = torch.cat((all_label, labels.cpu()), 0)

    val_loss = nn.CrossEntropyLoss()(all_output, all_label)

    val_loss = val_loss.numpy().item()

    all_output = all_output.float()
    _, predict = torch.max(all_output, 1)

    all_label = all_label.float()
    val_accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])

    all_output_numpy = all_output.numpy()
    predict_numpy = predict.numpy()
    all_label_numpy = all_label.numpy()

    with open(logs_path + '/_' + (str(num_iterations) if is_training else "Final") + '_confidence_values_.csv',
              mode='w') as file:
        csv_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        csv_writer.writerow(['True Label', 'Prediction'] + list(map(lambda x: 'class_' + str(x) + '_conf', np.arange(
            num_classes))))  # ['Image_Name', 'Prediction', 'class_0_conf', 'class_1_conf']

        for value in range(len(all_output_numpy)):
            csv_writer.writerow(
                [int(all_label_numpy[value]), predict_numpy[value]] +
                list(map(lambda x: all_output_numpy[value][x], np.arange(num_classes))))

    conf_mat = confusion_matrix(all_label, torch.squeeze(predict).float())
    val_info = {"acc": val_accuracy, "loss": val_loss}

    if ret_cm:
        val_info = {**val_info, "conf_mat": conf_mat}
    return val_info


def train(config, args, dset_loaders, unlabeled_data, models, optims):

    ####################################
    # Setup logging and early stoppage#
    ####################################
    logger = Logger(config["logs_path"] + "tensorboard/" + config['timestamp'])

    early_stopping = EarlyStopping(patience=config["patience"], verbose=True,delta=.00001)



    base_network = models['base']
    dis = models['dis']
    # crit = nn.CrossEntropyLoss()
    # crit.cuda()

    lm_opt = optims['lm_opt']
    dis_opt = optims['dis_opt']

    train_s_iter = iter(dset_loaders["train_s"])

    [unlabeled, l_d_pairs, id_pairs] = unlabeled_data
    bs = args.batch_size
    dis_y = torch.tensor([0] * bs + [1] * bs).cuda()

    bptt = args.bptt
    best_val_acc = 0.
    best_val_loss = np.infty
    final_test_acc = 0.
    print_msg("Training Started:", config['out_file'])
    p = 0
    ptrs = np.zeros(3, dtype=np.int64)
    total_loss = np.zeros(3)  # shape (n_lang, n_dom)
    total_clf_loss = 0
    total_dis_loss = 0
    start_time = time.time()
    base_network.train()
    base_network.reset()
    is_training = True
    val_loss = np.infty
    for itr in range(args.num_iterations):

        loss = 0
        lm_opt.zero_grad()
        dis_opt.zero_grad()


        seq_len = max(5, int(np.random.normal(bptt if np.random.random() < 0.95 else bptt / 2., 5)))
        lr0 = lm_opt.param_groups[0]['lr']
        lm_opt.param_groups[0]['lr'] = lr0 * seq_len / args.bptt

        # language modeling loss
        dis_x = []
        for i, ((lid, did), lm_x) in enumerate(zip(id_pairs, unlabeled)):
            if ptrs[i] + bptt + 1 > lm_x.size(0):
                ptrs[i] = 0
                base_network.reset(lid=lid, did=did)
            p = ptrs[i]
            xs = lm_x[p:p + bptt].t().contiguous()
            ys = lm_x[p + 1:p + 1 + bptt].t().contiguous()
            lm_raw_loss, lm_loss, hid = base_network.lm_loss(xs, ys, lid=lid, did=did, return_h=True)

            loss = loss + lm_loss * args.lambd_lm
            if lid == 0 and did == 0:
                feature = hid[-1].mean(1)
                raw_outputs = base_network(xs, None, lid=0, did=0)
                softmax_output = nn.Softmax(dim=1)(raw_outputs)
                op_out = torch.bmm(softmax_output.unsqueeze(2), feature.unsqueeze(1))
                dis_in = op_out.view(-1, softmax_output.size(1) * feature.size(1))

                dis_x.append(dis_in)
            elif lid == 1 and did == 1:
                _, _, hid = base_network.lm_loss(xs, ys, lid=1, did=0, return_h=True)

                feature = hid[-1].mean(1)
                raw_outputs = base_network(xs, None, lid=1, did=0)
                softmax_output = nn.Softmax(dim=1)(raw_outputs)
                op_out = torch.bmm(softmax_output.unsqueeze(2), feature.unsqueeze(1))
                dis_in = op_out.view(-1, softmax_output.size(1) * feature.size(1))

                dis_x.append(dis_in)
                # dis_x.append(hid[-1].mean(1))


            total_loss[i] += lm_raw_loss.item()
            ptrs[i] += bptt

        # entropy = loss.Entropy(softmax_out)
        # transfer_loss = loss.CDAN([features, softmax_out], ad_net, entropy, network.calc_coeff(itr))


        # language adversarial loss
        dis_x_rev = GradReverse.apply(torch.cat(dis_x, 0))
        dis_loss = nn.CrossEntropyLoss()(dis(dis_x_rev), dis_y)
        loss = loss + args.lambd_dis * dis_loss
        total_dis_loss += dis_loss.item()
        loss.backward()

        # sentiment classification loss
        try:
            xs, ys, ls = next(train_s_iter)
        except StopIteration:
            train_s_iter = iter(dset_loaders["train_s"])
            xs, ys, ls = next(train_s_iter)
        clf_loss = nn.CrossEntropyLoss()(base_network(xs, ls, lid=0, did=0), ys)
        total_clf_loss += clf_loss.item()
        (args.lambd_clf * clf_loss).backward()

        nn.utils.clip_grad_norm_(base_network.parameters(), 0.25)

        lm_opt.step()
        dis_opt.step()
        lm_opt.param_groups[0]['lr'] = lr0

        if (itr + 1) % args.test_interval == 0:

            config['out_file'].write('\n')
            config['out_file'].write('\n')




            total_loss /= args.test_interval
            total_clf_loss /= args.test_interval
            total_dis_loss /= args.test_interval
            elapsed = time.time() - start_time

            info = {'adaption_loss': total_loss.mean(),
                    'classifier_loss': total_clf_loss, 'adversarial_loss': total_dis_loss}



            print('|  itr {:5d} | Adaption_loss {:7.4f} |  Senti_classifier loss {:7.4f} | Adversarial loss {:7.4f} |'.format(
                itr, total_loss.mean(), total_clf_loss, total_dis_loss))

            config['out_file'].write('|  itr {:5d} | Adaption_loss {:7.4f} |  Senti_classifier loss {:7.4f} | Adversarial loss {:7.4f} |'.format(
                itr, total_loss.mean(), total_clf_loss, total_dis_loss))
            config['out_file'].write('\n')

            total_loss[:], total_clf_loss, total_dis_loss = 0, 0, 0
            start_time = time.time()


            base_network.eval()
            with torch.no_grad():

                train_info = test(dset_loaders['val_t'], base_network, 2, logs_path=config['logs_path'], l_id=1, d_id=0,
                                num_iterations=itr, is_training=is_training, ret_cm=False)
                # train_acc = train_info['acc']
                # val_acc = evaluate(model, dset_loaders["val_t"], 1, 0)
                # test_acc = evaluate(model, dset_loaders["test_t"], 1, 0)
                val_info = test(dset_loaders['val_t'], base_network, 2, logs_path=config['logs_path'], l_id=1, d_id=0,
                                num_iterations=itr, is_training=is_training, ret_cm=False)
                val_acc,val_loss = val_info['acc'],val_info['loss']
                info = {**info, 'loss': val_loss, 'val_acc': val_info['acc']}
                print('| step {:5d} | train acc {:.4f} |  train loss {:.4f} | val {:.4f} | val loss {:.4f} |'
                      .format(itr, train_info['acc'],train_info['loss'], val_acc, val_loss))
                config['out_file'].write('| step {:5d} | train acc {:.4f} |  train loss {:.4f} | val {:.4f} | val loss  {:.4f} |'
                                         .format(itr, train_info['acc'],train_info['loss'], val_acc, val_loss))
                config['out_file'].write('\n')


                if (itr+1) % config["snapshot_interval"] ==0:

                    torch.save(base_network, os.path.join(config["model_path"], "backup/model_iter_{:05d}_model.pth.tar".format(itr)))

                if val_loss < best_val_loss:
                    best_val_acc = val_acc
                    best_val_loss = val_loss
                    torch.save(base_network, os.path.join(config["model_path"], "best_model_model.pth.tar"))

                config["out_file"].flush()
            for tag, value in info.items():
                logger.scalar_summary(tag, value, itr)

            with open(config["logs_path"] + '/loss_values_.csv', mode='a') as file:
                csv_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

                csv_writer.writerow(
                    [itr]+list(info.values()))




            base_network.train()
            start_time = time.time()

        early_stopping(val_loss)

        if early_stopping.early_stop:

            print_msg("Early stopping" ,config['out_file'])
            print_msg("Saving Model ...", config['out_file'])
            print()
            print()

            break




    print_msg("Training Ended", config['out_file'])

    print_msg('Training ended with {} iterations'.format(itr + 1),config['out_file'])
    print_msg('Best validation acc {}\t ,val loss {}'.format(best_val_acc,best_val_loss),config['out_file'])


    base_network = torch.load(os.path.join(config["model_path"], "best_model_model.pth.tar"))

    test_info = test(dset_loaders['test_t'], base_network, 2, logs_path=config['logs_path'], l_id=1, d_id=0,
                                num_iterations=itr, is_training=False, ret_cm=False)

    print_msg('Test set  acc {} ,val loss {}'.format(test_info['acc'],test_info['loss']),config['out_file'])

if __name__ == '__main__':
    ####################################
    #   Parse args or TODO:load from args.json
    ####################################
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    set_deterministic_params(args.seed)

    ####################################
    # Default Project Folders#
    ####################################

    project_root = "../"
    data_root = project_root + "data/"
    models_root = project_root + "models/"

    model_name = args.source_lang + "-" + args.source_domain + '=>' + args.target_lang + "-" + args.target_domain

    log_output_dir_root = project_root + 'expt-logs-cd/' + model_name + '/'
    models_output_dir_root = project_root + 'models-cd/' + model_name + '/'

    now = datetime.now()
    timestamp = now.strftime("%d/%m/%Y %H:%M:%S")
    timestamp = timestamp.replace("/", "_").replace(" ", "_").replace(":", "_").replace(".", "_")

    trial_number = args.trial + "_" + timestamp

    config = {}
    config['trial_number'] = trial_number
    config["gpu"] = args.gpu_id
    config["num_iterations"] = args.num_iterations
    config["test_interval"] = args.test_interval
    config["snapshot_interval"] = args.snapshot_interval
    config["patience"] = args.patience
    log_output_path = log_output_dir_root + '/' + trial_number + '/'
    trial_results_path = models_output_dir_root + '/' + trial_number + '/'
    config["model_path"] = trial_results_path
    config["logs_path"] = log_output_path
    config['timestamp'] = timestamp
    config['lr'] =     args.lr
    if not os.path.exists(config["logs_path"]):
        os.makedirs(config["logs_path"])

    if not os.path.exists(config["model_path"] + "/backup/"):
        os.makedirs(config["model_path"] + "/backup/")

    config["out_file"] = open(os.path.join(config["logs_path"], "log.txt"), "w")
    config['train_data_path'] = data_root + args.train
    config['val_data_path'] = data_root + args.val
    config['test_data_path'] = data_root + args.test
    config['unlabeled_data_path'] = data_root + args.unlabeled
    ####################################
    # Dump arguments #
    ####################################
    # with open(config["logs_path"]+  "args.yml", "w") as f:
    #     yaml.dump(args, f)

    param_dict = dict(vars(args))
    with open(config["logs_path"] + "config.json", 'w') as fout:
        json.dump(param_dict, fout, indent=4)

    print('Configuration:')
    print('\n'.join('\t{:15} {}'.format(k + ':', str(v)) for k, v in sorted(dict(vars(args)).items())))
    print()
    print()
    print()


    ####################################
    # Load Datasets#
    ####################################

    # dset_loaders, vocab, [unlabeled, l_d_pairs, id_pairs] = data_setup(config, args)
    dset_loaders, vocab, unlabeled_data = data_setup(config, args)

    ####################################
    #  Network Setup
    ####################################

    models, optims = network_setup(config, args, vocab)
    ####################################
    #  Start Training
    ####################################

    train(config, args, dset_loaders, unlabeled_data, models, optims, )
