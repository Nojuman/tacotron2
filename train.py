import argparse
import sys
import matplotlib.pyplot as plt
plt.switch_backend('agg')

import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler
from tqdm import tqdm

import hyperparams as hp
from audio import wav_to_spectrogram
from datasets import *
from decoding_helpers import TacoTeacher, TacoGenerator
from models import MelSpectrogramNet
from sgdr import SGDRScheduler, LRFinderScheduler
from text import text_to_sequence, sequence_to_text
from utils import make_stop_targets
from visualize import show_spectrogram, show_attention

parser = argparse.ArgumentParser(description='PyTorch Tacotron Spectrogram Training')
parser.add_argument('--data',
                    default='/home/ubuntu/LJSpeech-1.1',
                    help='path to dataset')
parser.add_argument('--epochs', '-e', default=50, type=int,
                    help='number of total epochs to run')
parser.add_argument('--batch_size', '-bs', default=8, type=int, help='mini-batch size (default: 12)')
parser.add_argument('--name', '-n', default='melnet', help='experiment name', type=str)
parser.add_argument('--find_lr', default=False, type=bool,
                    help='runs training with LR finding scheduler,'
                         ' check tensorboard plots to choose max_lr')
parser.add_argument('--checkpoint', '-cp', default=None, type=str, help='path to checkpoint')



def generate(model, dataset, batch_size=1, save_interval=50, exp_name='taco', device=1, step=0):
    
    model.eval()
    writer = SummaryWriter(f'runs/{exp_name}')
    sampler = SequentialSampler(dataset)
    batch_sampler = RandomBatchSampler(sampler, batch_size)
    loader = DataLoader(dataset, collate_fn=my_collate, pin_memory=True, num_workers=6)
    tacogenerator = TacoGenerator()
    pbar = tqdm(loader, total=len(loader), unit=' batches')
    for b, (text_batch, _, _) in enumerate(pbar):

        text_batch = text_batch.squeeze(1)
        
        # update loop
        text = Variable(text_batch).cuda()
        outputs, stop_tokens, attention = model(text, tacogenerator)

        # plot the first sample in the batch
        attention_plot = show_attention(attention[0], return_array=True)
        output_plot = show_spectrogram(outputs.data.permute(1, 2, 0)[0],
                                       sequence_to_text(text.data[0]),
                                       return_array=True)
        writer.add_image('attention', attention_plot, step)
        writer.add_image('output', output_plot, step)
        step += 1
        if b == 5:
            break
            
            



def train(model, optimizer, scheduler, dataset, num_epochs, batch_size=1,
          save_interval=50, exp_name='taco', device=1, step=0):
    model.train()
    writer = SummaryWriter(f'runs/{exp_name}')
    sampler = SequentialSampler(dataset)
    batch_sampler = RandomBatchSampler(sampler, batch_size)
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=my_collate, pin_memory=True, num_workers=6)
    
    tacoteacher = TacoTeacher()
    for _ in tqdm(range(num_epochs), total=num_epochs, unit=' epochs'):
        pbar = tqdm(loader, total=len(loader), unit=' batches')
        for b, (text_batch, audio_batch, ID) in enumerate(pbar):
            if step < 50000:
                tf = np.cos((step)*np.pi/100000)
            text_lengths = text_batch.size(2)
            audio_lengths = audio_batch.size(2)
            
            audio_batch = audio_batch.permute(2,0,1)
            text_batch = text_batch.squeeze(1)
            # update loop
            text = Variable(text_batch).cuda(device)
            targets = Variable(audio_batch, requires_grad=False).cuda(device)
            stop_targets = make_stop_targets(targets, audio_lengths)
            tacoteacher.set_targets(targets)
            outputs, stop_tokens, attention = model(text, tacoteacher, tf)
            spec_loss = F.mse_loss(outputs, targets)
            stop_loss = F.binary_cross_entropy_with_logits(stop_tokens, stop_targets)
            loss = spec_loss + stop_loss
            optimizer.zero_grad()
            loss.backward()
            # clip_grad_norm(model.parameters(), hp.max_grad_norm, norm_type=2)  # prevent exploding grads
            scheduler.step()
            optimizer.step()

            # logging
            pbar.set_description(f'loss: {loss.data[0]:.4f}')
            writer.add_scalar('loss', loss.data[0], step)
            writer.add_scalar('lr', scheduler.lr, step)
            writer.add_scalar('tf', tf, step)
            if step % save_interval == 0:
                torch.save(model.state_dict(), f'checkpoints/{exp_name}_{str(step)}.pt')

                # plot the first sample in the batch
                attention_plot = show_attention(attention[0], return_array=True)
                output_plot = show_spectrogram(outputs.data.permute(1, 2, 0)[0],
                                               sequence_to_text(text.data[0]),
                                               return_array=True)
                target_plot = show_spectrogram(targets.data.permute(1, 2, 0)[0],
                                               sequence_to_text(text.data[0]),
                                               return_array=True)
                writer.add_image('attention', attention_plot, step)
                writer.add_image('output', output_plot, step)
                writer.add_image('target', target_plot, step)
            step += 1


def main():
    
    if not os.path.exists("checkpoints/"):
        os.makedirs("checkpoints/")
        print("Created a 'checkpoints' folder to save/load the model")
    
    args = parser.parse_args()
    MAKE_DATA = True
    step = 0
    exp_name = f'{args.name}_{hp.max_lr}_{hp.cycle_length}'

    if MAKE_DATA:
        data_gen = make_data()
        data_gen.make_text_data()
        data_gen.make_audio_data()
        
    # dataset = LJSpeechDataset(path=args.data, text_transforms=text_to_sequence,
    #                           audio_transforms=wav_to_spectrogram, cache=False)
    dataset = VCTKSets()
    model = MelSpectrogramNet()
    
    
    if args.checkpoint:
        weights = torch.load(args.checkpoint)
        model.load_state_dict(weights)
        step = int(args.checkpoint.split('/')[-1].split('_')[-1].split('.')[0])
        exp_name = "_".join(args.checkpoint.split('/')[-1].split('_')[:-1])

    model.cuda(device=0)
    optimizer = Adam(model.parameters(), lr=hp.max_lr, weight_decay=hp.weight_decay,
                     betas=(0.9, 0.999), eps=1e-6)

    if args.find_lr:
        scheduler = LRFinderScheduler(optimizer)
    else:
        scheduler = SGDRScheduler(optimizer, min_lr=hp.min_lr,
                                  max_lr=hp.max_lr, cycle_length=hp.cycle_length, current_step=step)

    #generate(model, dataset, batch_size=args.batch_size, save_interval=50, exp_name=exp_name, device=10, step=step)    
    
    train(model, optimizer, scheduler, dataset, args.epochs, args.batch_size, save_interval=50, exp_name=exp_name, device=0, step=step)


if __name__ == '__main__':
    main()
