import asyncio
from multiprocessing.dummy import Pipe
import multiprocessing as mp
import threading

import numpy as np
import rpyc
from slora.models.llama.SFT_service import LlamaSFTBackwardService
import torch
import traceback
import time
import os
import json
from collections import defaultdict
from multiprocessing import Process, Pipe

from datetime import timedelta
from tqdm import tqdm
from typing import Dict, List, Tuple
from rpyc.utils.classic import obtain

from transformers.configuration_utils import PretrainedConfig
from slora.server.router.model_infer.infer_batch import InferBatch

from slora.common.configs.config import setting
from slora.models.llama.model import LlamaTpPartModel
from slora.models.llama2.model import Llama2TpPartModel
from slora.models.peft.lora_adapter import LoraTpPartAdapter
from slora.models.peft.lora_unordered_batch_infer import LoraUnorderedBatchInfer
from slora.models.peft.lora_unordered_batch_mixed import LoraUnorderedBatchMixed
from slora.models.peft.lora_single_batch_infer import LoraPEFTBatchInfer
from slora.models.bmm.lora_bmm_infer import LoraBmmInfer
from slora.server.router.model_infer.infer_adapter import InferAdapter
from slora.server.router.model_infer.infer_adapter_alt import InferAdapterAlt
from slora.server.router.model_infer.naive_infer_adapter import NaiveInferAdapter
from slora.utils.infer_utils import set_random_seed
from slora.utils.infer_utils import calculate_time, mark_start, mark_end
from slora.utils.model_utils import get_model_config
from .post_process import sample

from ..mixed_req_queue import rprint
from pprint import pprint
from slora.models.lora_adamW_optimizer import ManualAdamW, ManualAdamW_2

from enum import Enum

class BackwardResumePoint(Enum):
    BEFORE_OPTIMIZER = 0
    OPTIMIZER = 1

class ModelRpcServer(rpyc.Service):

    def exposed_init_model(self, rank_id, world_size, weight_dir, adapter_dirs,
                           max_total_token_num, load_way, mode, input_params,
			   prefetch_stream, 
               half_model=False, mem_manager_log_path=None, 
               enable_unified_mem_manager=False, gpu_profiler=None, unified_mem_manager_max_size=0):
        import torch
        import torch.distributed as dist
        if world_size != 1:
            trans_list = [obtain(e) for e in (rank_id, world_size, weight_dir, adapter_dirs,
                                              max_total_token_num, load_way, mode)]
            rank_id, world_size, weight_dir, adapter_dirs, max_total_token_num, load_way, mode = trans_list

        self.tp_rank = rank_id
        self.world_size = world_size
        self.load_way = load_way
        self.mode = mode
        self.input_params = input_params
        self.prefetch_stream = prefetch_stream

        self.cache = {}
        self.original_weights = {}
        self.backward_status = BackwardResumePoint.BEFORE_OPTIMIZER
        self.enable_unified_mem_manager = enable_unified_mem_manager

        dist.init_process_group('nccl', init_method=f'tcp://127.0.0.1:{setting["nccl_port"]}', rank=rank_id, world_size=world_size)
        torch.cuda.set_device(rank_id)

        model_cfg = get_model_config(weight_dir, dummy=input_params.dummy)
        if half_model:
            model_cfg["num_hidden_layers"] = int(model_cfg["num_hidden_layers"] / 2)
        print("weight dir", weight_dir)
        try:
            self.model_type = model_cfg["model_type"]
            if self.model_type == "llama":
                if "num_key_value_heads" in model_cfg.keys():
                    self.model = Llama2TpPartModel(rank_id, world_size, weight_dir,
                                                    max_total_token_num,
                                                    mem_adapter_size=input_params.pool_size_lora,
                                                    load_way=load_way, mode=mode,
                                                    dummy=input_params.dummy)
                    
                else:
                    self.model = LlamaTpPartModel(rank_id, world_size, weight_dir,
                                                    max_total_token_num,
                                                    mem_adapter_size=input_params.pool_size_lora,
                                                    load_way=load_way, mode=mode,
                                                    dummy=input_params.dummy, 
                                                    half_model=half_model, 
                                                    mem_manager_log_path=mem_manager_log_path,
                                                    enable_unified_mem_manager=enable_unified_mem_manager,
                                                    unified_mem_manager_max_size=unified_mem_manager_max_size)
                    if gpu_profiler is not None: gpu_profiler.mark_annotation("model_load")
            else:
                raise Exception(f"can not support {self.model_type} now")
        except Exception as e:
            print("#" * 16)
            print("load model error:", str(e), e, type(e))
            raise e

        ''' init adapters '''
        # TODO support TP for adapters
        # print("adapter_dirs", adapter_dirs)
        self.adapters = []
        self.adapter_id = {}
        target_adapter_dir = None
        num = 0
        for adapter_dir in tqdm(adapter_dirs, desc="load adapters"):
            print(f"Adding adapter from {adapter_dir}, number {num}")
            num += 1
            target_adapter_dir = adapter_dir
            self.adapter_id[adapter_dir] = len(self.adapters)
            self.adapters.append(LoraTpPartAdapter(rank_id, world_size, adapter_dir, model_cfg,
                                                   swap=input_params.swap, dummy=input_params.dummy,
                                                   no_lora_swap=input_params.no_lora_swap,
						   prefetch_stream=prefetch_stream))

        finetuning_lora_path = input_params.finetuning_params.finetuning_lora_path
        if finetuning_lora_path != "":
            self.adapter_id[finetuning_lora_path] = len(self.adapters)
            print("Loading finetuning adapter", finetuning_lora_path)
            self.adapters.append(LoraTpPartAdapter(rank_id, world_size, finetuning_lora_path, model_cfg,
                                                    swap=input_params.swap, dummy=input_params.dummy,
                                                    no_lora_swap=input_params.no_lora_swap,
                                                        prefetch_stream=prefetch_stream, is_finetuning=True))

            self.finetuning_adapter = self.adapters[-1]
            self.finetuning_adapter.is_finetuning = True
            self.current_epoch = 0
            self.total_epochs = input_params.finetuning_params.num_epochs
            # lora_params = []
            # for i, layer in enumerate(self.finetuning_adapter.layers):
            #     param = getattr(layer, 'w_combined_home_fp32')
            #     param.requires_grad = True
            #     lora_params.append(param)
            #     name = f"layer_{i}.w_combined_home"
            #     self.original_weights[name] = param.clone().detach().cpu()
            # self.finetuning_optimizer = torch.optim.AdamW(
            #     lora_params, 
            #     lr=input_params.finetuning_params.learning_rate, 
            #     betas=(0.9,0.999), 
            #     weight_decay=input_params.finetuning_params.weight_decay)
            # self.finetuning_scheduler = torch.optim.lr_scheduler.StepLR(
            #     self.finetuning_optimizer,
            #     step_size=1,      # every epoch
            #     gamma=input_params.finetuning_params.gamma        # multiply by 0.5
            # )
            self.backward_service = None
            self.bwd_pause_event = None
            if True:
                rpc_recv, bwd_send = Pipe()
                bwd_recv, rpc_send = Pipe()
                self.bwd_pause_event = mp.Event()
                self.bwd_pause_event.set()   # set = RUNNING; clear = PAUSED

                backward_service_obj = LlamaSFTBackwardService(
                    self.model.config, bwd_recv, bwd_send,
                    lr=input_params.finetuning_params.learning_rate,
                    weight_decay=input_params.finetuning_params.weight_decay,
                    gamma=input_params.finetuning_params.gamma
                )
                backward_service_obj.bwd_pause_event = self.bwd_pause_event
                backward_service_obj.receive_model_dict(self.model.export_model_dict())
                self.rpc_recv = rpc_recv
                self.rpc_send = rpc_send
                self.backward_service = Process(target=backward_service_obj.start_service, daemon=True)
                os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = "20"
                os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
                self.backward_service.start()
                os.environ.pop("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", None)
                os.environ.pop("CUDA_DEVICE_MAX_CONNECTIONS", None)
                self.rpc_send.send(self.finetuning_adapter.load_gpu_fp32_dict())
                self.rpc_send.send(self.model.alt_mem_manager.share_activation_dict())
                self.rpc_recv.recv()

            if self.input_params.finetuning_params.finetuning_type == "Alignment" or self.input_params.finetuning_params.finetuning_type == "Alignment Live":
                ref_adapter_path = input_params.finetuning_params.reference_lora_path
                self.adapter_id[ref_adapter_path] = len(self.adapters)
                self.adapters.append(LoraTpPartAdapter(rank_id, world_size, ref_adapter_path, model_cfg,
                                                    swap=input_params.swap, dummy=input_params.dummy,
                                                    no_lora_swap=input_params.no_lora_swap,
                                                        prefetch_stream=prefetch_stream))
                self.model.backward_engine.setup_alignment(True, alpha=input_params.finetuning_params.alpha,
                                                           beta=input_params.finetuning_params.beta,
                                                           lambdas=input_params.finetuning_params.lambdas)

        self.adapter_id[None] = len(self.adapters)
        self.adapters.append(None)

        if input_params.no_mem_pool:
            head_num = self.model.config["num_attention_heads"]
            self.infer_adapter = NaiveInferAdapter.init(self.model.config["num_hidden_layers"],
                                                        head_num,
                                                        self.model.config["hidden_size"] // head_num)
        else:
            if self.enable_unified_mem_manager:
                self.infer_adapter_alt = InferAdapterAlt.init(self.model.alt_mem_manager)
                self.infer_adapter = None
            else:
                self.infer_adapter = InferAdapter.init(self.model.mem_manager,
                                                    prefetch_stream)
                self.infer_adapter_alt = None
        ''' finish init adapters '''
        set_random_seed(2147483647)
        return

    @torch.no_grad()
    def exposed_load_adapters(self, adapter_dirs, prefetch=False):
        if not self.input_params.bmm:
            adapters = []
            for adapter_dir in adapter_dirs:
                if adapter_dir is not None:
                    adapters.append(self.adapters[self.adapter_id[adapter_dir]])
            if self.enable_unified_mem_manager:
                self.infer_adapter_alt.load_adapters(adapters)
            else:
                self.infer_adapter.load_adapters(adapters, prefetch=prefetch)

        else:
            for adapter_dir in adapter_dirs:
                if adapter_dir is not None:
                    self.adapters[self.adapter_id[adapter_dir]].load_to_gpu(prefetch=prefetch, bmm=True)
            print(f"load {len(adapter_dirs)} on gpu")


    @torch.no_grad()
    def exposed_offload_adapters(self, reserve_dirs=None, prefetch=False):
        if not self.input_params.bmm:
            if self.enable_unified_mem_manager:
                self.infer_adapter_alt.offload_adapters(reserve_dirs if reserve_dirs is not None else [])
            else:
                self.infer_adapter.offload_adapters(reserve_dirs if reserve_dirs is not None else [])
        else:
            reserve_dirs = reserve_dirs if reserve_dirs is not None else []
            for adapter_dir, id in self.adapter_id.items():
                if adapter_dir is not None and adapter_dir not in reserve_dirs:
                    self.adapters[id].offload_from_gpu()

    # @calculate_time(show=True, min_cost_ms=0.1)
    def exposed_add_batch(self, batch_id, reqs, dtype):
        if self.world_size != 1:
            batch_id, reqs, dtype = obtain(batch_id), obtain(reqs), obtain(dtype)
        import torch
        if dtype == "fp16":
            dtype = torch.float16
        else:
            assert False, "error dtype"
        if self.enable_unified_mem_manager:
            batch_data = InferBatch.init_batch(batch_id, reqs, dtype, torch.cuda.current_device(), 
                                            self.model.mem_manager, self.model.vocab_size, self.model.alt_mem_manager)
        else:
            batch_data = InferBatch.init_batch(batch_id, reqs, dtype, torch.cuda.current_device(), 
                                            self.model.mem_manager, self.model.vocab_size, None)
        self.cache[batch_id] = batch_data
        return
    
    # @calculate_time(show=True, min_cost_ms=300)
    # @calculate_time(show=True, min_cost_ms=0)
    def exposed_prefill_batch(self, batch_id, prefill_interrupt_event=None):
        return self.forward(batch_id, is_prefill=True, prefill_interrupt_event=prefill_interrupt_event)

    # @calculate_time(show=True, min_cost_ms=200)
    # @calculate_time(show=True, min_cost_ms=0)
    def exposed_decode_batch(self, batch_id, decode_count=-1):
        return self.forward(batch_id, is_prefill=False, decode_count=decode_count)

    # @calculate_time(show=True, min_cost_ms=0.1)
    def exposed_filter_batch(self, batch_id, req_id_list):
        if self.world_size != 1:
            batch_id, req_id_list = obtain(batch_id), obtain(req_id_list)
        batch = self.cache.pop(batch_id)
        filter_batch = batch.filter(req_id_list)
        del batch
        self.cache[batch_id] = filter_batch
        return

    # @calculate_time(show=True, min_cost_ms=0.1)
    def exposed_merge_batch(self, batch_id1, batch_id2):
        batch1 = self.cache.pop(batch_id1)
        batch2 = self.cache.pop(batch_id2)
        m_batch = InferBatch.merge(batch1, batch2)
        del batch1
        del batch2
        self.cache[batch_id1] = m_batch
        return

    # @calculate_time(show=True, min_cost_ms=10)
    def exposed_remove_batch(self, batch_id):
        batch = self.cache.pop(batch_id)
        batch.free_self()
        del batch
        # torch.cuda.empty_cache()
        return

    def forward(self, batch_id, is_prefill, prefill_interrupt_event=None, decode_count=-1):
        batch: InferBatch = self.cache.pop(batch_id)
        # print(batch.requests)
        # print([req["request_id"] for req in batch.requests])
        kwargs = {
            "batch_size": len(batch),
            "total_token_num": batch.nopad_total_token_num,
            "max_len_in_batch": batch.nopad_max_len_in_batch,
            "input_ids": batch.input_ids,
            "b_loc": batch.nopad_b_loc,
            "b_start_loc": batch.nopad_b_start_loc,
            "b_seq_len": batch.nopad_b_seq_len,
            "is_prefill": is_prefill,
        }

        # assert False, f"{kwargs}"

        assert len(batch.adapter_dirs) == len(batch), "batch.adapter_dirs != batch"
            # always use lora batch infer
        if (self.input_params.no_lora or self.input_params.no_kernel or
            self.input_params.scheduler == "peft" or set(batch.adapter_dirs) == {None}):
            engine = self.model
        else:
            adapters = [self.adapters[self.adapter_id[adapter_dir]] for adapter_dir in batch.adapter_dirs]
            if self.input_params.no_lora_compute:
                engine = LoraUnorderedBatchInfer(self.model, adapters)
            elif self.input_params.bmm:
                torch.cuda.empty_cache()
                compressed_dirs = [batch.adapter_dirs[0]]
                adapter_sep = [0]
                cnt = 1
                for i in range(1, len(batch.adapter_dirs)):
                    if batch.adapter_dirs[i] == batch.adapter_dirs[i-1]:
                        cnt += 1
                    else:
                        compressed_dirs.append(batch.adapter_dirs[i])
                        adapter_sep.append(adapter_sep[-1] + cnt)
                        cnt = 1
                adapters = [self.adapters[self.adapter_id[adapter_dir]] for adapter_dir in compressed_dirs]
                engine = LoraBmmInfer(self.model, adapters, adapter_sep)
            elif self.input_params.finetuning_params.finetuning_lora_path!="":
                engine = LoraUnorderedBatchMixed(
                        self.model, 
                        adapters, 
                        infer_adapter=self.infer_adapter, 
                        infer_adapter_alt=self.infer_adapter_alt,
                        finetuning_adapter=self.finetuning_adapter,
                        enable_unified_mem_manager=self.enable_unified_mem_manager)
                kwargs["finetune_mask"] = batch.finetune_mask
                kwargs["b_loc_key"] = batch.nopad_b_loc_key
                kwargs["b_loc_value"] = batch.nopad_b_loc_value
                kwargs["prefill_interrupt_event"] = prefill_interrupt_event
                if self.input_params.finetuning_params.finetuning_type == "Alignment" or self.input_params.finetuning_params.finetuning_type == "Alignment Live":
                    kwargs["ref_mask"] = batch.ref_mask
            else:
                engine = LoraUnorderedBatchInfer(self.model, adapters, infer_adapter=self.infer_adapter)

            kwargs["no_lora_compute"] = self.input_params.no_lora_compute
            # kwargs["no_lora_copy"] = self.input_params.no_lora_copy 
        if not is_prefill and decode_count == 2:
            kwargs["print_time_profile"] = True
        start = time.time()
        logits = engine.forward(**kwargs)
        #print(f"Engine execution time: {time.time() - start:.3f}s")

        if logits is None:
            batch.nopad_max_len_in_batch += 1
            batch.nopad_b_seq_len += 1
            self.cache[batch.batch_id] = batch
            return None

        next_token_ids, next_token_probs = sample(logits, batch)
        next_token_ids = next_token_ids.detach().cpu().numpy()
        next_token_logprobs = torch.log(next_token_probs).detach().cpu().numpy()
        output_dict = {}
        new_input_ids = []        
        for i, (r, all_input_ids, next_token_id, next_token_logprob) in enumerate(zip(batch.requests, batch.all_input_ids, next_token_ids, next_token_logprobs)):
            # all_input_ids_tensor = torch.tensor(all_input_ids, dtype=torch.long, device="cuda")
            all_input_ids.append(int(next_token_id))
            # all_input_ids_tensor = None
            new_input_ids.append(next_token_id)
            batch.all_input_ids[i] = all_input_ids
            batch.input_lengths[i] += 1
            batch.out_token_id_counts[i][next_token_id] += 1
            metadata = {
                'id': int(next_token_id),
                'logprob': float(next_token_logprob),
            }
            output_dict[r['request_id']] = (int(next_token_id), metadata)
        
        batch.input_ids = torch.tensor(new_input_ids, dtype=torch.long).cuda()
        batch.nopad_b_start_loc = batch.nopad_b_start_loc + torch.arange(0, len(batch), dtype=torch.int32, device="cuda")
        batch.nopad_total_token_num += len(batch)
        batch.nopad_max_len_in_batch += 1
        batch.nopad_b_seq_len += 1
        self.cache[batch.batch_id] = batch
        return output_dict


    def compare_lora_drift_per_layer(self):
        """
        Compares current LoRA weights to the originally saved weights,
        and prints per-layer total drift (sum of L2 norms across q/k/v/o).
        """
        print("=== LoRA Parameter Drift by Layer ===")
        for layer_id, layer in enumerate(self.finetuning_adapter.layers):
            total_drift = 0.0
            for name in [
                "w_combined_home",
            ]:
                key = f"layer_{layer_id}.{name}"
                original = self.original_weights[key].to(layer.__dict__[name].device)
                current = getattr(layer, name)
                
                drift = torch.norm(current - original).item()
                total_drift += drift
                weight_norm = original.norm().item()
                ratio = total_drift / weight_norm

            if total_drift!=0:
                print(f"Layer {layer_id:02d}: total ∆W = {total_drift:.6f} |  step/weight ratio = {ratio:.4e}")


    def check_adapter_update(self):
        for idx, layer in enumerate(self.finetuning_adapter.layers):
            w_fp32 = layer.w_combined_home_fp32       # Tensor on GPU, dtype=float32
            w = layer.w_combined_home          # Tensor on GPU, dtype=float16
            if torch.isnan(w).any() or torch.isinf(w).any():
                print(f"[non-finite] layer {idx} → w_combined_home has NaN/Inf")
                if torch.isnan(w_fp32).any() or torch.isinf(w_fp32).any():
                    print(f"And w_combined_home_fp32 has NaN/Inf")
                else:
                    print(f"but w_combined_home_fp32 does not has NaN/Inf")
                break   
            
    def exposed_back_batch(self, current_epoch):
        self.current_epoch = current_epoch
        result = self.backward()
        if current_epoch == self.total_epochs -1 and self.input_params.finetuning_params.finetuning_type == "Alignment":
            self.model.backward_engine.print_reset_log()
        return result
    
    def backward(self):
        start = time.time()
        if self.backward_service is not None:
            requests_info_dict = self.model.alt_mem_manager.export_requests_info()
            requests_info_dict["current_epoch"] = self.current_epoch
            self.rpc_send.send(requests_info_dict)
            finished, loss, total_token_processed = self.rpc_recv.recv()
        else:
            finished, loss, total_token_processed = self.model.backward_engine._context_backward(self.model, self.finetuning_adapter)
        if finished:
            if self.enable_unified_mem_manager:
                print("reset activation pool in alt mem manager")
                self.model.alt_mem_manager.reset_activation_pool()
            else:
                self.model.mem_manager.reset_activation_pool()
            return True, loss, total_token_processed
        else:
            return False, None, None

    def exposed_pause_backward(self):
        self.bwd_pause_event.clear()

    def exposed_resume_backward(self):
        self.bwd_pause_event.set()
    
    def exposed_unmerge_adapter(self):
        print("len adapters:", len(self.infer_adapter.adapter_dirs))
        assert len(self.infer_adapter.adapter_dirs) == 1
        print("unmerge:", self.infer_adapter.adapter_dirs)
        engine = LoraPEFTBatchInfer(self.model, infer_adapter=self.infer_adapter)
        engine.unmerge_adapter()

    def exposed_merge_adapter(self):
        print("len adapters:", len(self.infer_adapter.adapter_dirs))
        assert len(self.infer_adapter.adapter_dirs) == 1
        print("merge:", self.infer_adapter.adapter_dirs)
        engine = LoraPEFTBatchInfer(self.model, infer_adapter=self.infer_adapter)
        engine.merge_adapter()


class ModelRpcClient:
    def __init__(self, model_rpc, world_size, rpc_server_process=None):
        self.model: ModelRpcServer = model_rpc
        self.world_size = world_size
        self.rpc_server_process = rpc_server_process
        self.use_rpc = self.world_size != 1
        if self.use_rpc:
            def async_wrap(f):
                f = rpyc.async_(f)
                async def _func(*args, **kwargs):
                    ans = f(*args, **kwargs)
                    await asyncio.to_thread(ans.wait)
                    # raise if exception
                    return ans.value
                return _func
            self._init_model = async_wrap(self.model.init_model)
            self._load_adapters = rpyc.async_(self.model.load_adapters)
            self._offload_adapters = rpyc.async_(self.model.offload_adapters)
            self._unmerge_adapter = rpyc.async_(self.model.unmerge_adapter)
            self._merge_adapter = rpyc.async_(self.model.merge_adapter)
            self._add_batch = async_wrap(self.model.add_batch)
            self._prefill_batch = async_wrap(self.model.prefill_batch)

            self._back_batch = async_wrap(self.model.back_batch)
            self._pause_backward = async_wrap(self.model.pause_backward)
            self._resume_backward = async_wrap(self.model.resume_backward)

            self._decode_batch = async_wrap(self.model.decode_batch)
            self._filter_batch = async_wrap(self.model.filter_batch)
            self._merge_batch = async_wrap(self.model.merge_batch)
            self._remove_batch = async_wrap(self.model.remove_batch)
        else:
            self._init_model = self.model.exposed_init_model
            self._load_adapters = self.model.exposed_load_adapters
            self._offload_adapters = self.model.exposed_offload_adapters
            self._merge_adapter = self.model.exposed_merge_adapter
            self._unmerge_adapter = self.model.exposed_unmerge_adapter
            self._add_batch = self.model.exposed_add_batch
            self._prefill_batch = self.model.exposed_prefill_batch

            self._back_batch = self.model.exposed_back_batch
            self._pause_backward = self.model.exposed_pause_backward
            self._resume_backward = self.model.exposed_resume_backward
            
            self._decode_batch = self.model.exposed_decode_batch
            self._filter_batch = self.model.exposed_filter_batch
            self._merge_batch = self.model.exposed_merge_batch
            self._remove_batch = self.model.exposed_remove_batch
        return

    async def init_model(self, rank_id, world_size, weight_dir, adapter_dirs,
                         max_total_token_num, load_way, mode, input_params,
			                prefetch_stream, half_model=False, mem_manager_log_path=None,
                            enable_unified_mem_manager=False, unified_mem_manager_max_size=0,
                            gpu_profiler=None):
        ans : rpyc.AsyncResult = self._init_model(rank_id, world_size, weight_dir, adapter_dirs,
                                                  max_total_token_num, load_way, mode, input_params,
						                            prefetch_stream, half_model, mem_manager_log_path, enable_unified_mem_manager,
                                                    gpu_profiler, unified_mem_manager_max_size)
        if self.use_rpc:
            await ans
            return
        else:
            return

    async def load_adapters(self, reqs, prefetch=False):
        self._load_adapters(reqs, prefetch=prefetch)


    async def offload_adapters(self, reserved_reqs=None, prefetch=False):
        self._offload_adapters(reserved_reqs, prefetch=prefetch)
    
    async def unmerge_adapter(self):
        self._unmerge_adapter()
    
    async def merge_adapter(self):
        self._merge_adapter()


    async def init_batch(self, batch_id, reqs):
        ans = self._add_batch(batch_id, reqs, "fp16")
        if self.use_rpc:
            await ans
            return
        else:
            return

    # async def prefill_batch(self, batch_id, prefill_interrupt_event=None):
    #     ans = self._prefill_batch(batch_id, prefill_interrupt_event)
    #     if self.use_rpc:
    #         return await ans
    #     else:
    #         return ans
    
    async def prefill_batch(self, batch_id, prefill_interrupt_event=None):
        if self.use_rpc:
            ans = self._prefill_batch(batch_id, prefill_interrupt_event)
            return await ans
        else:
            # run blocking call in a thread
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, self._prefill_batch, batch_id, prefill_interrupt_event)

    # async def back_batch(self):
    #     ans = self._back_batch()
    #     if self.use_rpc:
    #         return await ans
    #     else:
    #         return ans
        
    async def back_batch(self, current_epoch):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._back_batch, current_epoch)
    
    def back_batch_threading(self, current_epoch):
        loop = asyncio.get_running_loop()
        return loop.run_in_executor(None, self._back_batch, current_epoch)

    async def pause_backward(self):
        if self.use_rpc:
            ans = self._pause_backward()
            return await ans
            
        else:
            return self._pause_backward()

    async def resume_backward(self):
        if self.use_rpc:
            ans = self._resume_backward()
            return await ans
        else:
            return self._resume_backward()
    
    def reset_activation_pool(self):
        if self.model.enable_unified_mem_manager:
            self.model.model.alt_mem_manager.reset_activation_pool()
        else:
            self.model.model.mem_manager.reset_activation_pool()
        return

    async def decode_batch(self, batch_id, decode_count=-1):
        ans = self._decode_batch(batch_id, decode_count=decode_count)
        if self.use_rpc:
            return await ans
        else:
            return ans

    async def filter_batch(self, batch_id, req_id_list):
        ans = self._filter_batch(batch_id, req_id_list)
        if self.use_rpc:
            await ans
            return
        else:
            return 

    async def merge_batch(self, batch_id1, batch_id2):
        ans = self._merge_batch(batch_id1, batch_id2)
        if self.use_rpc:
            await ans
            return
        else:
            return

    async def remove_batch(self, batch_id):
        ans = self._remove_batch(batch_id)
        if self.use_rpc:
            await ans
            return
        else:
            return
    

def _init_env(port):
    from rpyc.utils.server import ThreadedServer
    t = ThreadedServer(ModelRpcServer(), port=port, protocol_config={"allow_pickle": True})
    t.start()
    return


async def start_model_process(port, world_size):
    # 单卡时不使用 rpc
    if world_size == 1:
        return ModelRpcClient(ModelRpcServer(), world_size)
    
    import multiprocessing
    proc = multiprocessing.Process(target=_init_env, args=(port,))
    proc.start()
    await asyncio.sleep(2)
    repeat_count = 0
    while repeat_count < 20:
        try:
            con = rpyc.connect("localhost", port, config={"allow_pickle": True})
            break
        except BaseException:
            await asyncio.sleep(1)
        repeat_count += 1
    if repeat_count == 20:
        raise Exception("init rpc env error!")

    assert proc.is_alive()
    return ModelRpcClient(con.root, world_size, rpc_server_process=proc)