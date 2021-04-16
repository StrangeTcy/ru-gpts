# coding=utf-8
# Copyright (c) 2020, Sber.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Sample Generate GPT3"""

import os
import time

import torch
# from transformers.tokenization_gpt2 import GPT2Tokenizer
from transformers import GPT2Tokenizer

from src import mpu
from src.arguments import get_args
from src.fp16 import FP16_Module
from src.model import DistributedDataParallel as DDP
from src.model import GPT3Model
from pretrain_gpt3 import generate
from pretrain_gpt3 import initialize_distributed
from pretrain_gpt3 import set_random_seed
from src.utils import (Timers, export_to_huggingface_model, 
    Timers, report_memory,
    save_checkpoint, load_checkpoint, load_huggingface_model,
    print_args, print_rank_0,
    get_sparse_attention_config, top_k_logits, DEEPSPEED_WRAP    
)

try: 
    import    cPickle as pickle
except:
    import pickle
    
    
# Flag to use Pytorch ddp which uses overlapping communication and computation.
USE_TORCH_DDP = False
if USE_TORCH_DDP:
    from torch.nn.parallel.distributed import DistributedDataParallel as DDP
else:
    from src.model import DistributedDataParallel as DDP    


def get_model(args):
    """Build the model."""

    print_rank_0('building GPT3 model ...')
    print ("Calling GPT3Model constructor...")  
    model = GPT3Model(num_layers=args.num_layers,
                      vocab_size=args.vocab_size,
                      hidden_size=args.hidden_size,
                      num_attention_heads=args.num_attention_heads,
                      embedding_dropout_prob=args.hidden_dropout,
                      attention_dropout_prob=args.attention_dropout,
                      output_dropout_prob=args.hidden_dropout,
                      max_sequence_length=args.max_position_embeddings,
                      checkpoint_activations=args.checkpoint_activations,
                      checkpoint_num_layers=args.checkpoint_num_layers,
                      parallel_output=False)

    if mpu.get_data_parallel_rank() == 0:
        print(' > number of parameters on model parallel rank {}: {}'.format(
            mpu.get_model_parallel_rank(),
            sum([p.nelement() for p in model.parameters()])), flush=True)

    # GPU allocation.
    print (f"placing the model on device {torch.cuda.current_device()}")
    model.cuda(torch.cuda.current_device())

    # Fp16 conversion.
    if args.fp16:
        rint ("we have NOT halfed the model before, and now we're wrapping it into a fp16_module. For...some reason...")
        model = FP16_Module(model)

    # Wrap model for distributed training.
    print ("Setting up distributed training...")
    print ("No classic pytorch DDP this time; \nUsing sberbank magic DDP")
    model = DDP(model)

    input ("ready to return model")
    return model


def magic_get_model(args):
    """Build the model."""

    print_rank_0('building GPT3 model ...')
    print ("asserting we have a correct number of attention heads...")
    assert args.num_attention_heads % args.model_parallel_size == 0
    num_local_heads = args.num_attention_heads // args.model_parallel_size
    deepspeed_sparsity_config = None
    if DEEPSPEED_WRAP and args.deepspeed:
        print ("we're using deepspeed, and so we're getting a sparse attention config")
        deepspeed_sparsity_config = get_sparse_attention_config(args, num_local_heads)
    if deepspeed_sparsity_config is not None:
        print_rank_0(f"Using sparse attention with mode {args.sparse_mode}")
    print ("Calling GPT3Model constructor...")    
    model = GPT3Model(num_layers=args.num_layers,
                      vocab_size=args.vocab_size,
                      hidden_size=args.hidden_size,
                      num_attention_heads=args.num_attention_heads,
                      embedding_dropout_prob=args.hidden_dropout,
                      attention_dropout_prob=args.attention_dropout,
                      output_dropout_prob=args.hidden_dropout,
                      max_sequence_length=args.max_position_embeddings,
                      checkpoint_activations=args.checkpoint_activations,
                      checkpoint_num_layers=args.checkpoint_num_layers,
                      parallel_output=True,
                      deepspeed_sparsity_config=deepspeed_sparsity_config,
                      sparse_mode=args.sparse_mode)

#     if args.load_huggingface is not None:
#         print ("Loading huggingface model...")
#         model = load_huggingface_model(model, args.load_huggingface, args.huggingface_double_pos_embeddings)

    if mpu.get_data_parallel_rank() == 0:
        print(' > number of parameters on model parallel rank {}: {}'.format(
            mpu.get_model_parallel_rank(),
            sum([p.nelement() for p in model.parameters()])), flush=True)

    # To prevent OOM for model sizes that cannot fit in GPU memory in full precision
    if DEEPSPEED_WRAP and args.deepspeed and args.fp16:
        print ("We've had deepspeed AND fp16, so we're halfing the model...")
        model.half()

    # GPU allocation.
    print (f"placing the model on device {torch.cuda.current_device()}")
    model.cuda(torch.cuda.current_device())

    # Fp16 conversion.
    if args.fp16:
        print ("we've halfed the model before, but now we're wrapping it into a fp16_module. For...some reason...")
        model = FP16_Module(model)

    # Wrap model for distributed training.
    print ("Setting up distributed training...")
    if USE_TORCH_DDP:
        i = torch.cuda.current_device()
        print (f"Using classic pytorch DDP with device {i}")
        model = DDP(model, device_ids=[i], output_device=i,
                    process_group=mpu.get_data_parallel_group())
    else:
        print ("Using sberbank magic DDP")
        model = DDP(model)

#     input ("ready to return model")
    print ("ready to return model")
    return model



def get_optimizer(model, args):
    """Set up the optimizer."""

    # Build parameter groups (weight decay and non-decay).
    while isinstance(model, (DDP, FP16_Module)):
        print ("we're dealing with a DDP/FP16_Module, extracting the module...")
        model = model.module
    print ("Getting param groups for weight decay optimization...")    
    param_groups = gpt3_get_params_for_weight_decay_optimization(model)

    # Add model parallel attribute if it is not set.
    for param_group in param_groups:
        for param in param_group['params']:
            if not hasattr(param, 'model_parallel'):
                param.model_parallel = False

    if args.cpu_optimizer:
        if args.cpu_torch_adam:
            cpu_adam_optimizer = torch.optim.Adam
        else:
            from deepspeed.ops.adam import DeepSpeedCPUAdam
            cpu_adam_optimizer = DeepSpeedCPUAdam
        optimizer = cpu_adam_optimizer(param_groups,
                                       lr=args.lr, weight_decay=args.weight_decay)
    else:
        # Use FusedAdam.
        optimizer = Adam(param_groups,
                         lr=args.lr, weight_decay=args.weight_decay)

    print(f'Optimizer = {optimizer.__class__.__name__}')
    
    print ("let's save our optimizer...")
    with open("/notebooks/sberbank_rugpts/our_model/optimizer.pkl", "wb") as f:
        pickle.dump(optimizer, f)
    
    if DEEPSPEED_WRAP and args.deepspeed:
        # fp16 wrapper is not required for DeepSpeed.
        print (f"we're using deepspeed, and so returning our optimizer {optimizer}")
        return optimizer

    # Wrap into fp16 optimizer.
    if args.fp16:
        print ("Wrapping into fp16")
        optimizer = FP16_Optimizer(optimizer,
                                   static_loss_scale=args.loss_scale,
                                   dynamic_loss_scale=args.dynamic_loss_scale,
                                   dynamic_loss_args={
                                       'scale_window': args.loss_scale_window,
                                       'min_scale': args.min_scale,
                                       'delayed_shift': args.hysteresis})
        
    print (f" we've probably wrapped our optimizer in fp16, \nand now we're eturning our optimizer {optimizer}")
    return optimizer


def get_learning_rate_scheduler(optimizer, args):
    """Build the learning rate scheduler."""

    # Add linear learning rate scheduler.
    if args.lr_decay_iters is not None:
        num_iters = args.lr_decay_iters
    else:
        num_iters = args.train_iters
    num_iters = max(1, num_iters)
    init_step = -1
    warmup_iter = args.warmup * num_iters
    lr_scheduler = AnnealingLR(optimizer,
                               start_lr=args.lr,
                               warmup_iter=warmup_iter,
                               num_iters=num_iters,
                               decay_style=args.lr_decay_style,
                               last_iter=init_step,
                               min_lr=args.min_lr)

    print (f"and now we're returning a scheduler {lr_scheduler}")
    return lr_scheduler


def setup_model_and_optimizer(args):
    """Setup model and optimizer."""

    print ("setting up model...")
    model = get_model(args)
    print ("setting up optimizer...")
    optimizer = get_optimizer(model, args)
    print ("setting up lr scheduler...")
    lr_scheduler = get_learning_rate_scheduler(optimizer, args)

    
    if DEEPSPEED_WRAP and args.deepspeed:
        print_rank_0("DeepSpeed is enabled.")

        print ("Calling deepspeed.initialize with our model, optimizer and scheduler")
        model, optimizer, _, lr_scheduler = DEEPSPEED_WRAP.deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            args=args,
            lr_scheduler=lr_scheduler,
            mpu=mpu,
            dist_init_required=False
        )
        print ("We've wrapped our model, optimizer and scheduler in DeepSpeed")

    if args.load is not None:
        print_rank_0("Load checkpoint from " + args.load)
        args.iteration = load_checkpoint(model, optimizer, lr_scheduler, args, deepspeed=DEEPSPEED_WRAP and args.deepspeed)
        print_rank_0("Checkpoint loaded")
    else:
        args.iteration = 0

    print ("returning our model, optimizer and scheduler")    
    return model, optimizer, lr_scheduler





def setup_model(args):
    """Setup model and optimizer."""

    model = magic_get_model(args)
#     if DEEPSPEED_WRAP and args.deepspeed:
#         print_rank_0("DeepSpeed is enabled.")

# #         optimizer='adam'
#         print ("Restoring our optimizer from a pickle...")
#         with open("/notebooks/sberbank_rugpts/our_model/optimizer.pkl", "rb") as f:
#             optimizer = pickle.load(f)
#         print (f"I'm pickle Riiick! I mean, optimizer now is {optimizer}")
#         model, optimizer, _, lr_scheduler = DEEPSPEED_WRAP.deepspeed.initialize(
#             model=model,
#             optimizer=optimizer,
#             args=args,
#             lr_scheduler=None,
#             mpu=mpu,
#             dist_init_required=False
#         )
#         optimizer = "FusedAdam"
#         model, optimizer, _, lr_scheduler = DEEPSPEED_WRAP.deepspeed.initialize(
#             model=model,
#             optimizer=None,
#             args=args,
#             lr_scheduler=None,
#             mpu=mpu,
#             dist_init_required=False
#         )


    print("Load checkpoint from " + args.load)
    _ = load_checkpoint(model, None, None, args, deepspeed=DEEPSPEED_WRAP and args.deepspeed)
#     _ = load_checkpoint(model, None, None, args, deepspeed=True)
    model.eval()
    print("Loaded")
    if args.export_huggingface is not None:
        export_to_huggingface_model(model, args.export_huggingface)
        print(f"Exported in huggingface format to {args.export_huggingface}")

    return model


def generate_samples(model, tokenizer, args):
    print (f"generate_samples was called with model {model} \n and tokenizer {tokenizer}")
    model.eval()
    with torch.no_grad():
        while True:
            torch.distributed.barrier(group=mpu.get_model_parallel_group())
            terminate_runs = 0
            print (f"terminate_runs = {terminate_runs}")

            if mpu.get_model_parallel_rank() == 0:
                print ("get_model_parallel_rank() was 0")
#                 raw_text = input("\nContext prompt (stop to exit) >>> ")
                raw_text = "localStorage.getItem("
                while not raw_text:
                    print('Prompt should not be empty!')
                    raw_text = input("\nContext prompt (stop to exit) >>> ")

                if "stop" in raw_text:
                    terminate_runs = 1
                else:
                    context_tokens = tokenizer(raw_text)['input_ids']
                    context_length = len(context_tokens)

                    if context_length >= args.seq_length // 2:
                        print("\nContext length", context_length,
                              "\nPlease give smaller context (half of the sequence length)!")
                        continue
            else:
                print (f"get_model_parallel_rank() was NOT 0 but {mpu.get_model_parallel_rank()}")
                _ = tokenizer("EMPTY TEXT")['input_ids']

            terminate_runs_tensor = torch.cuda.LongTensor([terminate_runs])
            torch.distributed.broadcast(terminate_runs_tensor, mpu.get_model_parallel_src_rank(),
                                        group=mpu.get_model_parallel_group())
            terminate_runs = terminate_runs_tensor[0].item()

            if terminate_runs == 1:
                return

            start_time = time.time()
            print ("generating...")
            generated = generate(
                model, tokenizer, raw_text,
                out_seq_length=args.out_seq_length,
                seq_length=args.seq_length,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p
            )

            if mpu.get_model_parallel_rank() == 0:
                print ("We should clear the terminal and print results...")
                os.system('clear')
                print("\nTime taken: {:.2f}\n".format(time.time() - start_time), flush=True)
                print("\nContext:", raw_text, flush=True)
                print("\nGPT:", generated, flush=True)
            raw_text = None

            torch.distributed.barrier(group=mpu.get_model_parallel_group())


def prepare_tokenizer(args):
    print (f"we've got args.tokenizer_path {args.tokenizer_path}")
    tokenizer = GPT2Tokenizer.from_pretrained(args.tokenizer_path)
    eod_token = tokenizer.encoder['<pad>']
    num_tokens = len(tokenizer)

    args.tokenizer_num_tokens = num_tokens
    args.eod_token = eod_token

    after = num_tokens
    while after % args.make_vocab_size_divisible_by != 0:
        after += 1

    args.vocab_size = after
    print(f"prepare tokenizer done, size {after}", flush=True)

    return tokenizer


def main():
    """Main training program."""

    print('Generate Samples')

    # Disable CuDNN.
    torch.backends.cudnn.enabled = False

    # Timer.
    _ = Timers()

    # Arguments.
    args = get_args()
    print (f"in main, we've got args.tokenizer_path {args.tokenizer_path}")

    # Pytorch distributed.
    initialize_distributed(args)

    # Random seeds for reproducability.
    set_random_seed(args.seed)

    # get the tokenizer
    tokenizer = prepare_tokenizer(args)

    # Model, optimizer, and learning rate.
    model = setup_model(args)

    # setting default batch size to 1
    args.batch_size = 1

    # generate samples
    generate_samples(model, tokenizer, args)


if __name__ == "__main__":
    main()
