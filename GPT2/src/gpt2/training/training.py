import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from gpt2.data.corpus import Dataset
from gpt2.training import TrainingSpec, TrainConfig, Recorder
from typing import Dict, Optional

try:
    from apex import amp
except ModuleNotFoundError:
    pass

import warnings
warnings.filterwarnings(action='ignore')


class Trainer(object):
    def __init__(self, spec: TrainingSpec, config: TrainConfig):
        self.spec = spec
        self.config = config
        self.no_cuda = True

    def train(self,
              no_cuda, 
              from_checkpoint: Optional[str] = None,
              from_pretrained: Optional[str] = None):
        
        if self.config.distributed:
            mp.spawn(self._train, args=(no_cuda, from_checkpoint, from_pretrained),
                     nprocs=self.config.gpus)
        else:
            self._train(no_cuda, 0, from_checkpoint, from_pretrained)

    def _train(self,
               no_cuda, 
               rank: int,
               from_checkpoint: Optional[str] = None,
               from_pretrained: Optional[str] = None):
        
        if not no_cuda:
            if self.config.distributed:
                torch.cuda.set_device(rank)
                dist.init_process_group(backend='nccl',
                                        init_method='tcp://127.0.0.1:8000',
                                        world_size=self.config.gpus,
                                        rank=rank)

        # Initialize training environment and prepare datasets.
        self.spec.initialize()
        #train_dataset, eval_dataset = self.spec.prepare_datasets()
        train_dataset, eval_dataset = self.spec.prepare_datasets()

        # Construct a model and load its pretrained weights.
        model = self.spec.construct_model()
        print("no cuda", no_cuda)
        if not no_cuda:
            model = model.cuda()

        if from_pretrained:
            if not no_cuda:
                ckpt = torch.load(from_pretrained, map_location='cuda')
            else:
                ckpt = torch.load(from_pretrained)

            model.load_state_dict(ckpt['model'])

            # Because the weights data allocates quite a lot of GPU memories,
            # we need to free the memories explicitly.
            del ckpt
            if not no_cuda:
                torch.cuda.empty_cache()

        # Create an optimizer and learning rate scheduler.
        optimizer, scheduler = self.spec.create_optimizer(model.parameters())
        recorder = Recorder()

        if self.config.use_amp:
            model, optimizer = amp.initialize(
                model, optimizer, opt_level='O2', verbosity=0)

        if self.config.distributed:
            model = nn.parallel.DistributedDataParallel(
                model, device_ids=[rank])

        start_step = 0
        # Restore last training states from checkpoint.
        if from_checkpoint:
            if not no_cuda:
                ckpt = torch.load(from_checkpoint, map_location='cuda')
            else:
                ckpt = torch.load(from_checkpoint)

            start_step = ckpt['step']
            recorder = ckpt['recorder']

            model.load_state_dict(ckpt['model'])
            optimizer.load_state_dict(ckpt['optimizer'])
            scheduler.load_state_dict(ckpt['scheduler'])

            train_dataset.assign(ckpt['train_dataset'])
            eval_dataset.assign(ckpt['eval_dataset'])

            if self.config.use_amp:
                amp.load_state_dict(ckpt['amp'])

            # Because the checkpoint data allocates quite a lot of GPU
            # memories, we need to free the memories explicitly.
            del ckpt
            if not no_cuda:
                torch.cuda.empty_cache()

        if rank == 0:
            # Create tqdm iterator in master process to show the progress of
            # training.
            training_iters = tqdm.tqdm(
                range(start_step + 1, self.config.total_steps),
                total=self.config.total_steps,
                desc=self.config.description,
                dynamic_ncols=True)
            training_iters.update(start_step + 1)
        else:
            # In other processes, use simple iterator rather than tqdm one.
            training_iters = range(start_step + 1, self.config.total_steps)
        
        for step in training_iters:
            # Clear CUDA cache which is used for training.
            torch.cuda.empty_cache()


            recorder.record(
                self._train_step(rank, train_dataset, model, optimizer,
                                 scheduler, calculate_acc=True),
                scope='train')
            
            # Clear CUDA cache which is used for evaluation.
            if not no_cuda:
                torch.cuda.empty_cache()

            if (step + 1) % self.config.eval_steps == 0:
                recorder.record(
                    self._eval_step(rank, eval_dataset, model), scope='eval')
                recorder.stamp(step)

                if rank == 0:
                    training_iters.set_postfix_str(
                        recorder.format(self.config.log_format))
            # Save training states to checkpoint file.
            if rank == 0 and (step + 1) % self.config.save_steps == 0:
                ckpt = {'step': step,
                        'recorder': recorder,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'train_dataset': train_dataset.where(),
                        'eval_dataset': eval_dataset.where()}

                if self.config.use_amp:
                    ckpt['amp'] = amp.state_dict()

                torch.save(ckpt, self.config.save_checkpoint_path)

                # Because the checkpoint data allocates quite a lot of GPU
                # memories, we need to free the memories explicitly.
                del ckpt
                if not no_cuda:
                    torch.cuda.empty_cache()

        # Since the model is wrapped with `DistributedDataParallel` class in
        # distributed training environment, the original model can be accessed
        # by `module` attribute.
        if self.config.distributed:
            model = model.module

        # Save trained model weights and metrics recorded during the training.
        if rank == 0:
            print("Saving model and metrics")
            torch.save({'model': model.cpu().state_dict(),
                        'metrics': recorder.batch_metrics},
                       self.config.save_model_path)

    def _train_step(self,
                    rank: int,
                    dataset: Dataset,
                    model: nn.Module,
                    optimizer: optim.Optimizer,
                    scheduler: optim.lr_scheduler._LRScheduler,
                    calculate_acc = False
                    ) -> Dict[str, float]:
        model.train()
        optimizer.zero_grad()

        data = self._fetch_from(dataset, rank, self.config.batch_train)
        metrics = self.spec.train_objective(data, model, calculate_acc)
        loss = metrics['loss']

        print("loss", loss)
        if self.config.use_amp:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        optimizer.step()
        scheduler.step()
        #print("metrics", metrics)
        #print("metrics", metrics.items())
        
        a = {k: self._to_value(v) for k, v in metrics.items()}
        print("metrics from train step", a)

        return {k: self._to_value(v) for k, v in metrics.items()}
            

    @torch.no_grad()
    def _eval_step(self, rank: int, dataset: Dataset, model: nn.Module
                   ) -> Dict[str, float]:
        model.eval()

        data = self._fetch_from(dataset, rank, self.config.batch_eval)
        metrics = self.spec.eval_objective(data, model)

        return {k: self._to_value(v) for k, v in metrics.items()}

    def _fetch_from(self, dataset: Dataset, rank: int, batch: int
                    ) -> Dict[str, torch.Tensor]:
        if self.config.distributed:
            # In distributed training environment, each process must ignore
            # sub-batches of other processes and fetch corresponding one only.
            batch = batch // self.config.gpus

            dataset.skip(rank * batch)
            data = dataset.fetch(batch)
            dataset.skip((self.config.gpus - rank - 1) * batch)
        else:
            data = dataset.fetch(self.config.batch_train)
        if not self.no_cuda:
            return {k: v.cuda() for k, v in data.items()}
        else:
            return {k: v for k, v in data.items()}

    def _to_value(self, inp): #tensor: torch.Tensor) -> float:
        if isinstance(inp, torch.Tensor):
            if self.config.distributed:
                tensor = inp.clone()
                dist.all_reduce(inp, op=dist.reduce_op.SUM)
                return (inp / self.config.gpus).item()
            else:
                return inp.item()
        else:
            return inp
