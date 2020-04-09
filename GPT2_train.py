
import transformers
import torch
import os
import json
import random
import numpy as np
import argparse
from datetime import datetime
from tqdm import tqdm
from torch.nn import DataParallel
import logging
import time
from transformers import GPT2LMHeadModel,GPT2Config
logging.basicConfig(level=logging.INFO)
from transformers.configuration_utils import PretrainedConfig

# 加入变量
parser = argparse.ArgumentParser()
# 设置使用的gpu
parser.add_argument('--device', default='6', type=str, required=False, help='使用的gpu的名称')
parser.add_argument('--model_config', default='/data/mileywang/GPT2_lm/gpt2-config.json', type=str, required=False, help='模型参数')
parser.add_argument('--tokenizer_data_path',default='/data/mileywang/GPT2_lm/', type=str, required=False, help = 'tokenized语料存放的位置')
parser.add_argument('--tokenizer_path',default='/data/mileywang/GPT2_lm/', type=str, required=False, help = 'tokenized的字典')
parser.add_argument('--epochs', default= 5 ,type= int, required=False,help='循环训练的次数')
parser.add_argument('--batch_size',default= 8, type=int, required=False, help = '训练的batch size')
parser.add_argument('--lr', default= 1.5e-4, type=float, required=False, help= '学习率')
parser.add_argument('--num_pieces',default=100 , type=int, required=False, help = '将训练语料分成的份数')
parser.add_argument('--min_length',default=2,type=int,required=False,help='最短收录文章的长度')
#parser.add_argument('--pretrained_model', default='', type=str, required=False, help='模型起点路径')
parser.add_argument('--output_dir', default='/data/mileywang/GPT2_lm/lm_result', type=str, required=False, help='结果输出路径')
parser.add_argument('--log_step', default=1, type=int, required=False, help='多少步汇报一次')
parser.add_argument('--stride', default=24, type=int, required=False, help='取数据的窗口步长')
parser.add_argument('--warmup_steps', default=2000, type=int, required=False, help='warm up步数')
#parser.add_argument('--log_step', default=1, type=int, required=False, help='多少步汇报一次loss，设置为gradient accumulation的整数倍')
parser.add_argument('--gradient_accumulation', default=1, type=int, required=False, help='梯度积累')
#parser.add_argument('--output_dir',default='/data/mileywang/GPT2_lm/pytorch_model.bin',type=str, required = False ,help = '模型输出的路径')
parser.add_argument('--pretrained_model', default='/data/mileywang/GPT2_lm/', type=str, required=False, help='模型训练起点路径')
parser.add_argument('--writer_dir',default = 'tensorboard_summary/',type = str ,required = False ,help = 'Tensorboard路径' )
parser.add_argument('--val_dataset', metavar='PATH', type=str, default=None, help='Dataset for validation loss, defaults to --dataset.')
parser.add_argument('--val_batch_size', metavar='SIZE', type=int, default=2, help='Batch size for validation.')
parser.add_argument('--val_batch_count', metavar='N', type=int, default=40, help='Number of batches for validation.')
parser.add_argument('--val_every', metavar='STEPS', type=int, default=0, help='Calculate validation loss every STEPS steps.')
parser.add_argument('--max_grad_norm', type=float, default=0.5, help='max grand norm.')


output_dir = '/data/mileywang/GPT2_lm/pytorch_model.bin'
args = parser.parse_args()
#将模型的基本参数打印出来
print('args:\n' + args.__repr__())
# datapath='C:/Users/ubt/Desktop/train_max24.tok'
# f = open(datapath,encoding = 'utf-8')
# for lines in f:
#     print(len(lines))
#     print(lines)
os.environ["CUDA_VISIBLE_DEVICES"] = args.device  # 此处设置程序使用哪些显卡
print(args.model_config)

#if not args.pretrained_model:
#    model = transformers.modeling_gpt2.GPT2LMHeadModel(config=args.model_config)
#else: 
#    model = transformers.modeling_gpt2.GPT2LMHeadModel(config=args.model_config)
# Load config file
config = GPT2Config.from_json_file(args.model_config)
model = GPT2LMHeadModel(config)
model.train()
model.cuda()

num_parameters = 0
parameters = model.parameters()
for parameter in parameters:
    num_parameters += parameter.numel()
print('number of parameters: {}'.format(num_parameters))

full_len = 0
for i in tqdm(range(args.num_pieces)):
    with open(args.tokenizer_data_path + 'tokenizer_train_{}.txt'.format(i), 'r') as f:
        full_len += len([int(item) for item in f.read().strip().split()])
total_steps = int(full_len / args.stride * args.epochs / args.batch_size / args.gradient_accumulation)
print('total steps = {}'.format(total_steps))



#定义优化器：
optimizer = transformers.AdamW(model.parameters(),lr = args.lr, correct_bias=True)
#scheduler = transformers.WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps,t_total=total_steps)

n_ctx = 24

if torch.cuda.device_count() > 1:
    print("let's use ", torch.cuda.device_count(), "GPUs!")
    model = DataParallel(model , device_ids= [int (i) for i in args.device.split(",")])
    multi_gpu = True

overall_step = 0
running_loss = 0
for epoch in range(args.epochs):
    print('epoch {}'.format(epoch + 1))
    now = datetime.now()
    print('time : {}'.format(now))
    #随机化
    x = np.linspace(0, args.num_pieces - 1, args.num_pieces , dtype = np.int32)
    random.shuffle(x)
    piece_num = 0
    for i in x:
        with open(args.tokenizer_data_path + 'tokenizer_train_{}.txt'.format(i), 'r') as f:
            line = f.read().strip()
        tokens = line.split()
        tokens = [int(token) for  token in tokens ]
        start_point = 0
        samples= []
        while start_point < len(tokens)- n_ctx :
            samples.append(tokens[start_point:start_point + n_ctx])
            start_point += args.stride
        if start_point < len(tokens):
            samples.append(tokens[len(tokens)-n_ctx:])
        random.shuffle(samples)

        ##s数据准备 prepare data
        for step in range(len(samples) //args.batch_size):
            batch = samples[step * args.batch_size: (step + 1) * args.batch_size]
            batch_inputs = []
            for ids in batch:
                int_ids = [int(x) for x in ids]
                batch_inputs.append(int_ids)
            batch_inputs = torch.tensor(batch_inputs).long().cuda()
        ##forward pass
            outputs = model.forward(input_ids = batch_inputs , labels= batch_inputs )
            loss, logits = outputs[:2]

        ##get loss
            #if  multi_gpu:
           #     loss = loss.mean()
            if  args.gradient_accumulation >1 :
                loss = loss / args.gradient_accumulation

        ## loss backward
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), args.max_grad_norm)

        ## optimizer step
            if (overall_step + 1) % args.gradient_accumulation == 0:
                running_loss += loss.item()
                optimizer.step()
                optimizer.zero_grad()
                #scheduler.step()
            if (overall_step + 1) % args.log_step == 0:
                print('now time: {}:{}. Step {} of piece {} of epoch {}, loss {}'.format(
                datetime.now().hour,
                datetime.now().minute,
                step + 1,
                piece_num,
                epoch + 1,
                running_loss * args.gradient_accumulation / (args.log_step / args.gradient_accumulation)))
            running_loss = 0
        overall_step += 1
        piece_num += 1

    # for batch in range(args.batch_size):

        print('saving model for epoch {}'.format(epoch + 1))
        if not os.path.exists(args.output_dir + 'model_epoch{}'.format(epoch + 1)):
            os.mkdir(args.output_dir + 'model_epoch{}'.format(epoch + 1))
        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save_pretrained(args.output_dir + 'model_epoch{}'.format(epoch + 1))
        # torch.save(scheduler.state_dict(), output_dir + 'model_epoch{}/scheduler.pt'.format(epoch + 1))
        # torch.save(optimizer.state_dict(), output_dir + 'model_epoch{}/optimizer.pt'.format(epoch + 1))
        print('epoch {} finished'.format(epoch + 1))

        then = datetime.now()
        print('time: {}'.format(then))
        print('time for one epoch: {}'.format(then - now))

    print('training finished')
    if not os.path.exists(args.output_dir + 'final_model'):
        os.mkdir(args.output_dir + 'final_model')
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(args.output_dir + 'final_model')
    # torch.save(scheduler.state_dict(), output_dir + 'final_model/scheduler.pt')
    # torch.save(optimizer.state_dict(), output_dir + 'final_model/optimizer.pt')

    
