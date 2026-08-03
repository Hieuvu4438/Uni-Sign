import torch
from torch.utils.data import DataLoader
from models import Uni_Sign
import utils as utils
from datasets import S2T_Dataset
import os
import time
import argparse, json, datetime
from pathlib import Path
import math
import sys
from timm.optim import create_optimizer
from models import get_requires_grad_dict
from SLRT_metrics import translation_performance, islr_performance, wer_list
from transformers import get_scheduler
from config import *


def create_finetuning_optimizer(args, model):
    """Create AdamW groups with a conservative mT5 learning rate when set."""
    if args.mt5_lr <= 0:
        return create_optimizer(args, model)
    if args.opt.lower() != 'adamw':
        raise ValueError("--mt5-lr currently requires --opt AdamW")

    no_decay_terms = ('bias', 'LayerNorm.weight', 'layer_norm.weight')
    groups = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        is_mt5 = name.startswith('mt5_model.')
        uses_decay = not name.endswith(no_decay_terms)
        key = (is_mt5, uses_decay)
        groups.setdefault(key, []).append(parameter)

    parameter_groups = []
    for (is_mt5, uses_decay), parameters in groups.items():
        parameter_groups.append({
            'params': parameters,
            'lr': args.mt5_lr if is_mt5 else args.lr,
            'weight_decay': args.weight_decay if uses_decay else 0.0,
        })
    if not parameter_groups:
        raise ValueError("No trainable parameters remain after freezing")
    return torch.optim.AdamW(
        parameter_groups,
        eps=args.opt_eps,
        betas=tuple(args.opt_betas) if args.opt_betas is not None else (0.9, 0.999),
    )


def get_closed_vocabulary(args):
    if not args.closed_vocabulary:
        return []
    if args.task != 'ISLR':
        raise ValueError("--closed-vocabulary is only supported for --task ISLR")
    if not args.label_vocab:
        raise ValueError("--closed-vocabulary requires --label-vocab")
    return utils.load_label_vocabulary(args.label_vocab)


def validate_closed_vocabulary(datasets, vocabulary):
    if not vocabulary:
        return
    allowed = set(vocabulary)
    for split_name, dataset in datasets.items():
        observed = {utils.normalize_label(sample['text']) for sample in dataset.raw_data.values()}
        unknown = observed.difference(allowed)
        if unknown:
            raise ValueError(
                f"{split_name} labels are absent from --label-vocab: {sorted(unknown)}")


def checkpoint_payload(model, args, epoch=None):
    payload = {
        'model': get_requires_grad_dict(model),
        'args': vars(args),
    }
    if epoch is not None:
        payload['epoch'] = epoch
    return payload

def main(args):
    utils.init_distributed_mode_ds(args)

    print(args)
    utils.set_seed(args.seed)
    closed_vocabulary = get_closed_vocabulary(args)

    print(f"Creating dataset:")
        
    train_data = S2T_Dataset(path=train_label_paths[args.dataset], 
                             args=args, phase='train')
    print(train_data)
    train_sampler = (torch.utils.data.distributed.DistributedSampler(train_data, shuffle=True)
                     if args.distributed else torch.utils.data.RandomSampler(train_data))
    train_dataloader = DataLoader(train_data,
                                 batch_size=args.batch_size, 
                                 num_workers=args.num_workers, 
                                 collate_fn=train_data.collate_fn,
                                 sampler=train_sampler, 
                                 pin_memory=args.pin_mem,
                                 drop_last=False)
        
    test_data = S2T_Dataset(path=test_label_paths[args.dataset], 
                            args=args, phase='test')
    print(test_data)
    # test_sampler = torch.utils.data.distributed.DistributedSampler(test_data,shuffle=False)
    test_sampler = torch.utils.data.SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data,
                                 batch_size=args.batch_size,
                                 num_workers=args.num_workers, 
                                 collate_fn=test_data.collate_fn,
                                 sampler=test_sampler, 
                                 pin_memory=args.pin_mem)

    if "How2Sign" not in args.dataset:
        dev_data = S2T_Dataset(path=dev_label_paths[args.dataset],
                               args=args, phase='dev')
        print(dev_data)
        # dev_sampler = torch.utils.data.distributed.DistributedSampler(dev_data,shuffle=False)
        dev_sampler = torch.utils.data.SequentialSampler(dev_data)
        dev_dataloader = DataLoader(dev_data,
                                    batch_size=args.batch_size,
                                    num_workers=args.num_workers,
                                    collate_fn=dev_data.collate_fn,
                                    sampler=dev_sampler,
                                    pin_memory=args.pin_mem)
    else:
        dev_dataloader = test_dataloader

    validate_closed_vocabulary(
        {'train': train_data, 'dev': dev_data if "How2Sign" not in args.dataset else test_data,
         'test': test_data},
        closed_vocabulary)

    print(f"Creating model:")
    model = Uni_Sign(args=args)

    if args.finetune != '':
        print('***********************************')
        print('Load Checkpoint...')
        print('***********************************')
        state_dict = torch.load(args.finetune, map_location='cpu')['model']

        ret = model.load_state_dict(state_dict, strict=True)
        print('Missing keys: \n', '\n'.join(ret.missing_keys))
        print('Unexpected keys: \n', '\n'.join(ret.unexpected_keys))

    model.configure_mt5_trainability(args.freeze_mt5, args.unfreeze_mt5_last_n)
    model.cuda()
    model.train()
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.float32)
    
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    n_parameters = utils.count_parameters_in_MB(model_without_ddp)
    n_trainable_parameters = sum(p.numel() for p in model_without_ddp.parameters() if p.requires_grad) / 1e6
    print(f'number of params: {n_parameters}M, trainable: {n_trainable_parameters}M')

    optimizer = create_finetuning_optimizer(args, model_without_ddp)
    lr_scheduler = get_scheduler(
                name='cosine',
                optimizer=optimizer,
                num_warmup_steps=int(args.warmup_epochs * len(train_dataloader)/args.gradient_accumulation_steps),
                num_training_steps=int(args.epochs * len(train_dataloader)/args.gradient_accumulation_steps),
            )
    
    model, optimizer, lr_scheduler = utils.init_deepspeed(args, model, optimizer, lr_scheduler)
    model_without_ddp = model.module.module
    # print(model_without_ddp)
    print(optimizer)

    output_dir = Path(args.output_dir)

    start_time = time.time()
    max_accuracy = 0
    if args.task == "CSLR":
        max_accuracy = 1000
    
    if args.eval:
        if utils.is_main_process():
            if "How2Sign" not in args.dataset:
                print("📄 dev result")
                evaluate(args, dev_dataloader, model, model_without_ddp, phase='dev',
                         closed_vocabulary=closed_vocabulary)
            print("📄 test result")
            evaluate(args, test_dataloader, model, model_without_ddp, phase='test',
                     closed_vocabulary=closed_vocabulary)

        return 
    print(f"Start training for {args.epochs} epochs")

    for epoch in range(0, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        
        train_stats = train_one_epoch(args, model, train_dataloader, optimizer, epoch)

        if args.output_dir:
            checkpoint_paths = [output_dir / f'checkpoint_{epoch}.pth']
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    **checkpoint_payload(model_without_ddp, args, epoch),
                }, checkpoint_path)

        # single gpu inference
        if utils.is_main_process():
            dev_stats = evaluate(args, dev_dataloader, model, model_without_ddp, phase='dev',
                                 closed_vocabulary=closed_vocabulary)
            if args.eval_test_each_epoch:
                evaluate(args, test_dataloader, model, model_without_ddp, phase='test',
                         closed_vocabulary=closed_vocabulary)

            if args.task == "SLT":
                if max_accuracy < dev_stats["bleu4"]:
                    max_accuracy = dev_stats["bleu4"]
                    if args.output_dir and utils.is_main_process():
                        checkpoint_paths = [output_dir / 'best_checkpoint.pth']
                        for checkpoint_path in checkpoint_paths:
                            utils.save_on_master({
                                **checkpoint_payload(model_without_ddp, args, epoch),
                            }, checkpoint_path)

                print(f"BLEU-4 of the network on the {len(dev_dataloader)} dev videos: {dev_stats['bleu4']:.2f}")
                print(f'Max BLEU-4: {max_accuracy:.2f}%')
            
            elif args.task == "ISLR":
                selection_metric = args.selection_metric
                if selection_metric == 'auto':
                    selection_metric = 'top1_acc_pc' if args.dataset == 'CoSign' else 'top1_acc_pi'
                if selection_metric not in dev_stats:
                    raise ValueError(f"Selection metric {selection_metric!r} is not available")
                if max_accuracy < dev_stats[selection_metric]:
                    max_accuracy = dev_stats[selection_metric]
                    if args.output_dir and utils.is_main_process():
                        checkpoint_paths = [output_dir / 'best_checkpoint.pth']
                        for checkpoint_path in checkpoint_paths:
                            utils.save_on_master({
                                **checkpoint_payload(model_without_ddp, args, epoch),
                            }, checkpoint_path)

                print(f"PI accuracy of the network on the {len(dev_dataloader)} dev videos: {dev_stats['top1_acc_pi']:.2f}")
                print(f'Max PI accuracy: {max_accuracy:.2f}%')
            
            elif args.task == "CSLR":
                if max_accuracy > dev_stats["wer"]:
                    max_accuracy = dev_stats["wer"]
                    if args.output_dir and utils.is_main_process():
                        checkpoint_paths = [output_dir / 'best_checkpoint.pth']
                        for checkpoint_path in checkpoint_paths:
                            utils.save_on_master({
                                **checkpoint_payload(model_without_ddp, args, epoch),
                            }, checkpoint_path)
                            
                print(f"WER of the network on the {len(dev_dataloader)} dev videos: {dev_stats['wer']:.2f}")
                print(f'Min WER: {max_accuracy:.2f}%')
        
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'dev_{k}': v for k, v in dev_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters}
            
        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
        
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

def train_one_epoch(args, model, data_loader, optimizer, epoch):
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    print_freq = 10
    optimizer.zero_grad()

    target_dtype = None
    if model.bfloat16_enabled():
        target_dtype = torch.bfloat16

    for step, (src_input, tgt_input) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if target_dtype != None:
            for key in src_input.keys():
                if isinstance(src_input[key], torch.Tensor):
                    src_input[key] = src_input[key].to(target_dtype).cuda()

        if args.task == "CSLR":
            tgt_input['gt_sentence'] = tgt_input['gt_gloss']
        stack_out = model(src_input, tgt_input)
        
        total_loss = stack_out['loss']
        model.backward(total_loss)
        model.step()

        loss_value = total_loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
            
        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    return  {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def evaluate(args, data_loader, model, model_without_ddp, phase, closed_vocabulary=None):
    model.eval()
    closed_vocabulary = closed_vocabulary or []

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    target_dtype = None
    if model.bfloat16_enabled():
        target_dtype = torch.bfloat16
        
    with torch.no_grad():
        tgt_pres = []
        tgt_refs = []
        tgt_name = []
        prediction_rows = []
 
        for step, (src_input, tgt_input) in enumerate(metric_logger.log_every(data_loader, 10, header)):
            if target_dtype != None:
                for key in src_input.keys():
                    if isinstance(src_input[key], torch.Tensor):
                        src_input[key] = src_input[key].to(target_dtype).cuda()
            
            if args.task == "CSLR":
                tgt_input['gt_sentence'] = tgt_input['gt_gloss']
            stack_out = model(src_input, tgt_input)
            
            total_loss = stack_out['loss']
            metric_logger.update(loss=total_loss.item())

            references = tgt_input['gt_sentence']
            names = src_input['name_batch']
            if closed_vocabulary:
                scores = model_without_ddp.score_candidate_labels(stack_out, closed_vocabulary)
                top_k = min(5, len(closed_vocabulary))
                ranked = torch.argsort(scores, dim=1, descending=True)[:, :top_k].cpu().tolist()
                for i, label_indices in enumerate(ranked):
                    prediction = closed_vocabulary[label_indices[0]]
                    tgt_pres.append(prediction)
                    tgt_refs.append(references[i])
                    tgt_name.append(names[i])
                    prediction_rows.append({
                        'sample': names[i],
                        'prediction': prediction,
                        'reference': references[i],
                        'top_labels': [closed_vocabulary[index] for index in label_indices],
                        'top_scores': [float(scores[i, index].item()) for index in label_indices],
                    })
            else:
                max_new_tokens = args.max_new_tokens
                if max_new_tokens <= 0:
                    if args.task == 'ISLR' and args.label_vocab:
                        labels = utils.load_label_vocabulary(args.label_vocab)
                        token_lengths = model_without_ddp.mt5_tokenizer(labels).input_ids
                        max_new_tokens = max(len(tokens) for tokens in token_lengths)
                    else:
                        max_new_tokens = 12 if args.task == 'ISLR' else 100
                output = model_without_ddp.generate(stack_out,
                                                     max_new_tokens=max_new_tokens,
                                                     num_beams=4)
                predictions = model_without_ddp.mt5_tokenizer.batch_decode(
                    output, skip_special_tokens=True)
                for i, prediction in enumerate(predictions):
                    tgt_pres.append(prediction)
                    tgt_refs.append(references[i])
                    tgt_name.append(names[i])
                    prediction_rows.append({
                        'sample': names[i],
                        'prediction': prediction,
                        'reference': references[i],
                    })

    # fix mt5 tokenizer bug
    if args.dataset == 'CSL_Daily' and args.task == "SLT":
        tgt_pres = [' '.join(list(r.replace(" ",'').replace("\n",''))) for r in tgt_pres]
        tgt_refs = [' '.join(list(r.replace("，", ',').replace("？","?").replace(" ",''))) for r in tgt_refs]

    if args.task == "SLT":
        bleu_dict, rouge_score = translation_performance(tgt_refs, tgt_pres)
        for k,v in bleu_dict.items():
            metric_logger.meters[k].update(v)
        metric_logger.meters['rouge'].update(rouge_score)
        if args.eval and (args.dataset == 'How2Sign' or args.dataset == 'OpenASL'):
            # BLEURT # follow GloFE
            # Due to the long processing time, only --eval will be executed.
            from bleurt import score
            checkpoint = "./BLEURT-20"
            scorer = score.BleurtScorer(checkpoint)
            scores_bleurt = scorer.score(references=tgt_refs[:], candidates=tgt_pres[:])
            # assert isinstance(scores, list) and len(scores) == 1
            print('BLEURT:', sum(scores_bleurt)/len(scores_bleurt))

    elif args.task == "ISLR":
        tgt_pres = [utils.normalize_label(item) for item in tgt_pres]
        tgt_refs = [utils.normalize_label(item) for item in tgt_refs]
        top1_acc_pi, top1_acc_pc = islr_performance(tgt_refs, tgt_pres)
        metric_logger.meters['top1_acc_pi'].update(top1_acc_pi)
        metric_logger.meters['top1_acc_pc'].update(top1_acc_pc)
        if closed_vocabulary:
            top5_acc = sum(
                reference in row['top_labels']
                for reference, row in zip(tgt_refs, prediction_rows)
            ) / len(tgt_refs) * 100 if tgt_refs else 0.0
            metric_logger.meters['top5_acc'].update(top5_acc)
        
    elif args.task == "CSLR":
        wer_results = wer_list(hypotheses=tgt_pres, references=tgt_refs)
        print(wer_results)
        for k,v in wer_results.items():
            metric_logger.meters[k].update(v)

    # # gather the stats from all processes
    # metric_logger.synchronize_between_processes()
    
    if utils.is_main_process() and utils.get_world_size() == 1 and args.eval:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / f'{phase}_predictions.jsonl').open('w', encoding='utf-8') as f:
            for row in prediction_rows:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
        
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser = argparse.ArgumentParser('Uni-Sign scripts', parents=[utils.get_args_parser()])
    args = parser.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
