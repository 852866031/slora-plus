import copy
from functools import partial
from slora.server.router.profile_req_queue import Profile_ReqQueue
from slora.server.router.tracker import BatchExecutionTracker, BatchExecutionType, DecodeExecutionEstimator, PrefillExecutionEstimator
from slora.server.router.profiling_batch_generator import ProfilingBatchGenerator
import uvloop
import asyncio
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
import os
import pickle
import time
import torch
import zmq
import zmq.asyncio
from typing import Dict, List, Optional

from ..sampling_params import SamplingParams
from ..io_struct import FinetuneReq, FinetuneStatusReq, Req, Batch, BatchAbortReq
from .model_infer.model_rpc import start_model_process, ModelRpcClient
from .req_queue import ReqQueue
from .mixed_req_queue import Mixed_ReqQueue
from rpyc.utils.classic import obtain
from slora.utils.infer_utils import calculate_time
from ..io_struct import BatchTokenIdOut, AbortReq
from .stats import Stats

from slora.server.input_params import InputParams
from slora.models.peft.lora_adapter import get_lora_config
from slora.server.router.profiler import AlphaModel, BetaModel
from slora.server.router.abort_req_queue import AbortReqQueue
from slora.server.router.cluster_req_queue import ClusterReqQueue
from slora.server.router.vtc_req_queue import VTCReqQueue
from slora.server.router.pets_req_queue import PETSReqQueue
from slora.server.router.peft_req_queue import PEFTReqQueue

def get_scheduler(input_params, adapter_dirs):
    if input_params.scheduler == "vtc_fair":
        return VTCReqQueue(input_params.max_total_token_num, input_params.batch_max_tokens,
                           input_params.running_max_req_size, adapter_dirs, input_params.fair_weights)
    elif input_params.scheduler == "pets":
        return PETSReqQueue(input_params.max_total_token_num, input_params.batch_max_tokens,
                            input_params.running_max_req_size)
    elif input_params.scheduler == "peft":
        return PEFTReqQueue(input_params.max_total_token_num, input_params.batch_max_tokens,
                            input_params.running_max_req_size)
    elif input_params.batch_num_adapters is not None:
        return ClusterReqQueue(input_params.max_total_token_num, input_params.batch_max_tokens,
                               input_params.running_max_req_size, input_params.batch_num_adapters)
    elif input_params.enable_abort:
        return AbortReqQueue(input_params.max_total_token_num, input_params.batch_max_tokens,
                             input_params.running_max_req_size)
    elif input_params.scheduler == "slora":
        return ReqQueue(input_params.max_total_token_num, input_params.batch_max_tokens,
                        input_params.running_max_req_size)
    elif input_params.scheduler == "slora_plus":
        if input_params.finetuning_params.finetuning_type == "SFT":
            return Mixed_ReqQueue(input_params.max_total_token_num, input_params.batch_max_tokens,
                                input_params.running_max_req_size, input_params.finetuning_params, input_params.slo_params, input_params.bwd_log_index)
        elif input_params.finetuning_params.finetuning_type == "SFT Profile":
            return Profile_ReqQueue(input_params.max_total_token_num, input_params.batch_max_tokens,
                                input_params.running_max_req_size, input_params.finetuning_params, input_params.slo_params, adapter_dirs[0])
    else:
        raise Exception("unrecognized scheduler")

from pprint import pprint

from enum import Enum

class RouterStatus(Enum):
    IDLE = 0
    PREFILL = 1
    DECODE = 2
    BWD = 3

import threading

def router_print(*args, sep=' ', end='\n', in_red=False):
    GREEN = "\033[92m"
    RED = "\033[1;31m"
    RESET = "\033[0m"

    color = RED if in_red else GREEN
    text = sep.join(str(arg) for arg in args)
    print(f"{color}[Router]: {text}{RESET}", end=end)

class RouterManager:
    def __init__(self, weightdir, adapter_dirs, load_way, world_size, eos_id,
                 router_port, detokenization_port, model_rpc_ports,
                 input_params,
                 mode=[], log_stats=True, log_stats_interval=10, half_model=False, 
                 mem_manager_log_path=None, enable_unified_mem_manager=False, unified_mem_manager_max_size=0,
                 enable_gpu_profile=False, rank_id=0):
        self.model_weightdir = weightdir
        self.adapter_dirs = adapter_dirs
        self.world_size = world_size
        self.load_way = load_way
        self.rank_id = rank_id
        self.mode = mode
        self.input_params = input_params
        self.finetuning_params = input_params.finetuning_params
        self.no_inference_since = time.time()
        self.half_model = half_model
        self.mem_manager_log_path = mem_manager_log_path
        self.enable_unified_mem_manager = enable_unified_mem_manager
        self.unified_mem_manager_max_size = unified_mem_manager_max_size
        self.update_sent = False

        if self.input_params.prefetch:
            self.prefetch_stream = torch.cuda.Stream()
        else:
            self.prefetch_stream = None

        pprint(input_params.__dict__)
        pprint(input_params.finetuning_params.__dict__)
        # get adapter rank
        self.lora_ranks = {}
        for lora_dir in adapter_dirs:
            config, _ = get_lora_config(lora_dir, input_params.dummy)
            self.lora_ranks[lora_dir] = config["r"]
        self.lora_ranks[input_params.finetuning_params.finetuning_lora_path] = get_lora_config(adapter_dirs[-1], input_params.dummy)[0]["r"]
        self.lora_ranks[None] = 0       
        print(self.lora_ranks)
        self.running_batch: Batch = None
        self.eos_id = eos_id
        
        context = zmq.asyncio.Context(3)
        self.recv_from_httpserver = context.socket(zmq.PULL)
        self.recv_from_httpserver.bind(f"tcp://127.0.0.1:{router_port}")
        self.send_to_detokenization = context.socket(zmq.PUSH)
        self.send_to_detokenization.connect(f"tcp://127.0.0.1:{detokenization_port}")
        self.req_queue = get_scheduler(input_params, adapter_dirs)
        self.profiling_batch_generator = ProfilingBatchGenerator(input_params.finetuning_params, adapter_dirs[0])
        self.profiling_batch_generator.prepare()
        self.prefill_estimator = PrefillExecutionEstimator()
        self.decode_estimator = DecodeExecutionEstimator()
        self.batch_exec_tracker = BatchExecutionTracker()
        if isinstance(self.req_queue, Mixed_ReqQueue):
            self.req_queue.set_estimators(self.prefill_estimator, self.decode_estimator)

        self.model_rpc_ports = model_rpc_ports

        self.stats_tool = Stats(log_stats, log_stats_interval)
        self.gpu_profiler = None
        
        self.decode_step_count = 0
        self.backward_is_running = False
        self.prefill_interrupt_event = None
        self.pause_printing = True

    def is_backward_running(self):
        return self.backward_is_running

    async def wait_to_model_ready(self):
        self.model_rpcs: List[ModelRpcClient] = []
        for rank_id in range(self.rank_id, self.world_size):
            rpc_model = await start_model_process(port=self.model_rpc_ports[rank_id], 
                                                  world_size=self.world_size)
            self.model_rpcs.append(rpc_model)

        init_model_ret = []
        for rank_id in range(self.rank_id, self.world_size):  # async init model process
            init_model_ret.append(
                self.model_rpcs[rank_id].init_model(
                    rank_id,
                    self.world_size,
                    self.model_weightdir,
                    self.adapter_dirs,
                    self.input_params.max_total_token_num,
                    self.load_way,
                    self.mode,
                    input_params=self.input_params,
                    prefetch_stream=self.prefetch_stream,
                    half_model=self.half_model,
                    mem_manager_log_path=self.mem_manager_log_path,
                    enable_unified_mem_manager=self.enable_unified_mem_manager,
                    unified_mem_manager_max_size=self.unified_mem_manager_max_size,
                    gpu_profiler=self.gpu_profiler
                ))

        await asyncio.gather(*init_model_ret)
        return

    def add_req(
        self,
        adapter_dir: str,
        prompt_ids: List[int],
        sampling_params: SamplingParams,
        request_id: str
    ):
        req = Req(adapter_dir, request_id, prompt_ids, sampling_params)
        req.arrival_time = time.time()
        self.req_queue.append(req)
        self.send_to_detokenization.send_pyobj(req.to_req_detokenization_state())
        return

    async def abort(self, request_id):
        if self.running_batch is not None:
            for req in self.running_batch.reqs:
                if req.request_id == request_id:
                    req.has_generate_finished = True
                    req.aborted = True
        for req in self.req_queue.waiting_req_list:
            if req.request_id == request_id:
                req.has_generate_finished = True
                req.aborted = True
        return

    async def loop_for_fwd(self,):
        counter_count = 0
        while True:
            #await self._step()
            await self._co_serving_step()
            counter_count += 1
            if self.running_batch is not None:
                if counter_count % 50 == 0:
                    pass
                self.stats_tool.print_stats()
                
            if self.running_batch is None:
                await asyncio.sleep(0.01)  # 10ms
    
    def _check_if_finetuning_scheduler(self):
        return isinstance(self.req_queue, Mixed_ReqQueue) or isinstance(self.req_queue, Profile_ReqQueue)

    def _check_backward_condition(self, printing=False):
        if self._check_if_finetuning_scheduler() \
            and not self.req_queue.finetuning_is_finished() \
            and self.req_queue.ready_for_bwd():
            return True
        return False
    
    def _clear_abort_reqs(self):
        if self.input_params.enable_abort and len(self.req_queue.abort_req_list) > 0:
            self.send_to_detokenization.send_pyobj(BatchAbortReq(self.req_queue.abort_req_list))
            self.req_queue.reset_abort_list()

    async def _co_serving_step(self):
        if self.prefill_estimator.fit_rmse > 0.1 and self.batch_exec_tracker.check_refit():
            self.prefill_estimator.data_fit(self.batch_exec_tracker)
            self.decode_estimator.data_fit(self.batch_exec_tracker)
            router_print(f"Error for prefill estimator: {self.prefill_estimator.fit_rmse}")
            router_print(f"Error for decode estimator: {self.decode_estimator.fit_rmse}")
        if self.running_batch is None:
            # Prefill new batch
            self._clear_abort_reqs()
            new_batch = self.req_queue.generate_new_batch(self.running_batch, self.lora_ranks, self.is_backward_running())
            if new_batch is not None:
                for req in new_batch.reqs:
                    if req.needs_to_notify_detokenize:
                        self.send_to_detokenization.send_pyobj(req.to_req_detokenization_state())
                self.running_batch = new_batch
                rets = [self.model_rpcs[tp_rank].load_adapters(self.running_batch.adapter_dirs) for tp_rank in range(self.world_size)]
                await asyncio.gather(*rets)
                if new_batch.get_inference_token_num() == 0:
                    self.prefill_interrupt_event = threading.Event()
                await self._prefill_batch(self.running_batch)
                self.decode_step_count = 0
                self.prefill_interrupt_event = None
                await self._filter_runing_batch()
                return
            else:
                await self.resume_backward()
        elif await self.req_queue.check_will_starve(self.running_batch, self.lora_ranks):
            router_print(f"Incoming request will starve, prefill new batch after {self.decode_step_count} decoding steps.")
            # Prefill and merge batch
            self._clear_abort_reqs()
            new_mini_batch = self.req_queue.generate_new_batch(self.running_batch, self.lora_ranks, self.is_backward_running())
            if new_mini_batch is not None:
                rets = [self.model_rpcs[tp_rank].load_adapters(new_mini_batch.adapter_dirs) for tp_rank in range(self.world_size)]
                await asyncio.gather(*rets)
                await self._prefill_batch(new_mini_batch, minibatch=True)
                self.decode_step_count = 0
                if not new_mini_batch.is_clear():
                    await self._merge_batch(self.running_batch, new_mini_batch)
                    self.running_batch.merge(new_mini_batch)
            return
        else:
            # Decode existing batch
            await self._decode_batch(self.running_batch)
            await self._filter_runing_batch()
            self.decode_step_count += 1
            # if self.running_batch is None and len(self.req_queue.waiting_req_list)==0:
            #     await self.resume_backward()
            return

            
    async def _init_batch(self, batch: Batch):
        reqs = [r.to_rpc_obj() for r in batch.reqs]
        rets = [self.model_rpcs[tp_rank].init_batch(batch.batch_id, reqs) for tp_rank in range(self.world_size)]
        await asyncio.gather(*rets)
        return

    def backward_callback(self, combined_future, job_id=None):
        results = combined_future.result()
        loss_list = []
        processed_tokens =0
        success = True
        for finished, loss, tokens in results:
            success = success and finished
            if loss!=None: loss_list.append(loss)
            if tokens!= None: processed_tokens += tokens
        if success: 
            self.req_queue.update_finetuning_status_after_bwd(loss_list, processed_tokens)
        self.backward_is_running = False
        if job_id is not None:
            self.gpu_profiler.stop_annotation(job_id)

    def _start_back_batch_threading(self, current_epoch=None):
        self.backward_is_running = True
        if self.gpu_profiler is not None:
            job_id = self.gpu_profiler.start_annotation("backward")
        futures = []
        for tp_rank in range(self.world_size):
            fut = self.model_rpcs[tp_rank].back_batch_threading(current_epoch=current_epoch)
            if isinstance(fut, asyncio.Future):
                futures.append(fut)
        if futures == []:
            return
        combined = asyncio.gather(*futures)
        if self.gpu_profiler is not None:
            combined.add_done_callback(partial(self.backward_callback, job_id=job_id))
        else:
            combined.add_done_callback(self.backward_callback)

    def predict_inference_tokens(self, batch: Batch):
        token_count = 0
        for req in batch.reqs:
            if not req.is_finetuning:
                token_count += req.max_output_len
        return token_count

    async def _prefill_batch(self, batch, minibatch=True):
        inference_tokens_list, finetuning_tokens_list = batch.export_batch_info()
        await self.pause_backward()
        if self.gpu_profiler is not None:
            job_id = self.gpu_profiler.start_annotation(f"prefill #tokens: {batch.input_tokens()}")
        await self._init_batch(batch)
        prefill_start_time = time.time()
        rets = [self.model_rpcs[tp_rank].prefill_batch(batch.batch_id, self.prefill_interrupt_event) for tp_rank in range(self.world_size)]
        ans = await asyncio.gather(*rets)
        if None not in ans:
            if self.world_size != 1:
                req_to_out_token_id = obtain(ans[0])
            else:
                req_to_out_token_id = ans[0]
            self._add_token_id_to_req(batch, req_to_out_token_id)
        if self._check_if_finetuning_scheduler() and None not in ans:
            self.req_queue.update_finetuning_status_after_fwd(batch)
        has_new_finished_req = batch.mark_finished_req(self.eos_id)
       
        if None not in ans:
            self._send_to_detokenization_proc(batch, req_to_out_token_id)
            earliest_arrival_time = batch.get_earliest_arrival_time()
            prefill_end_time = time.time()
            duration = prefill_end_time - prefill_start_time
            self.batch_exec_tracker.add_batch_stats(
                inference_tokens=inference_tokens_list,
                finetuning_tokens=finetuning_tokens_list,
                execution_type=BatchExecutionType.PREFILL,
                execution_duration=duration)
            batch.record_time_to_first_token(prefill_end_time)
            if earliest_arrival_time is not None:
                if prefill_end_time - earliest_arrival_time <= self.req_queue.ttft_slo * 1.05:
                    router_print(f"Prefill Duration: {duration:.3f}, worst TTFT: {prefill_end_time - earliest_arrival_time:.3f}")
                else:
                    router_print(f"Prefill Duration: {duration:.3f}, worst TTFT: {prefill_end_time - earliest_arrival_time:.3f}", in_red=True)
        await self._handle_finish_req(batch, has_new_finished_req, minibatch=minibatch, has_ft_reqs=finetuning_tokens_list!=[])

        if self._check_if_finetuning_scheduler() and None not in ans:
            if self.is_backward_running():
                await self.resume_backward()
            elif self._check_backward_condition():
                    await self.resume_backward()
                    self._start_back_batch_threading(current_epoch=self.req_queue.finetuning_manager.current_epoch)

        if self.gpu_profiler is not None:
            self.gpu_profiler.stop_annotation(job_id)
        if None in ans:
            return False
        return True

    async def _decode_batch(self, batch:Batch):
        inference_tokens_list, finetuning_tokens_list = batch.export_batch_info()
        num_inf_tokens = sum(inference_tokens_list)
        c1 = num_inf_tokens < 200 and len(finetuning_tokens_list) !=0
        c2 = num_inf_tokens > 5000 
        c3 = self.decode_step_count > 25
        if c1 or c2 or c3:
            if self.pause_printing:
                if c1:
                    router_print(f"Pause due to small inference tokens {num_inf_tokens}")
                if c2:
                    router_print(f"Pause due to large inference tokens {num_inf_tokens}")
                if c3:
                    router_print(f"Pause due to long decode steps {self.decode_step_count}")
                self.pause_printing = False
            await self.pause_backward()
        else:
            await self.resume_backward()
        if self.gpu_profiler is not None:
            job_id = self.gpu_profiler.start_annotation(f"decode #tokens: {batch.input_tokens()}")
        self.req_queue.update_counter(batch)
        start_time = time.time()
        rets = [self.model_rpcs[tp_rank].decode_batch(batch.batch_id, self.decode_step_count) for tp_rank in range(self.world_size)]
        try:
            ans = await asyncio.gather(*rets)
        except Exception as e:
            for adapter_dir in batch.adapter_dirs:
                router_print(f"{adapter_dir}", in_red=True)
            print("Current decode step count:", self.decode_step_count)
            raise e
        if isinstance(self.req_queue, Profile_ReqQueue):
            print(f"Decode Step {self.decode_step_count} Duration: {(time.time() - start_time):.4f}s")  
            self.req_queue.add_to_decode_time_queue(time.time() - start_time)
        # if self.decode_step_count<=4:
        #     duration = time.time() - start_time
        #     print(f"Decode Step {self.decode_step_count} Duration: {duration:.4f}s, Decode tokens: {num_inf_tokens}, Decode requests: {num_inf_reqs}")  
        decode_end_time = time.time()
        if self.world_size != 1:
            req_to_out_token_id = obtain(ans[0])
        else:
            req_to_out_token_id = ans[0]
        self._add_token_id_to_req(batch, req_to_out_token_id)
        has_new_finished_req = batch.mark_finished_req(self.eos_id)
        self._send_to_detokenization_proc(batch, req_to_out_token_id)
        duration = decode_end_time - start_time
        if self.decode_step_count < 8:
            self.batch_exec_tracker.add_batch_stats(
                inference_tokens=inference_tokens_list,
                finetuning_tokens=finetuning_tokens_list,
                execution_type=BatchExecutionType.DECODE,
                execution_duration=duration)
        batch.record_token_time(decode_end_time)
        await self._handle_finish_req(batch, has_new_finished_req)
        if self.gpu_profiler is not None:
            self.gpu_profiler.stop_annotation(job_id)
        return

    async def _filter_batch(self, batch: Batch):
        req_id_list = [r.request_id for r in batch.reqs]
        rets = [self.model_rpcs[tp_rank].filter_batch(batch.batch_id, req_id_list) for tp_rank in range(self.world_size)]
        await asyncio.gather(*rets)
        return

    async def _merge_batch(self, batch1, batch2):
        rets = [self.model_rpcs[tp_rank].merge_batch(batch1.batch_id, batch2.batch_id) for tp_rank in range(self.world_size)]
        await asyncio.gather(*rets)
        return

    async def _remove_batch(self, batch):
        rets = [self.model_rpcs[tp_rank].remove_batch(batch.batch_id) for tp_rank in range(self.world_size)]
        await asyncio.gather(*rets)
        return

    async def _handle_finish_req(self, batch: Batch, has_new_finished_req, minibatch=False, has_ft_reqs=False):
        adapter_to_keep = set()
        if has_ft_reqs:
            for req in batch.reqs:
                if req.adapter_dir!=self.req_queue.finetuning_lora_path:
                    adapter_to_keep.add(req.adapter_dir)
        if has_new_finished_req:
            batch.filter_finished()
            # unmerge adapter from base model
            if self.input_params.scheduler == "peft" and batch.is_clear():
                ret = []
                for tp_rank in range(self.world_size):
                    ret.append(self.model_rpcs[tp_rank].unmerge_adapter())
                await asyncio.gather(*ret)

            if not minibatch and not self.input_params.no_lora and not has_ft_reqs:
                #print(" _handle_finish_req Offloading adapters...")
                ret = []
                for tp_rank in range(self.world_size):
                    ret.append(self.model_rpcs[tp_rank].offload_adapters(batch.adapter_dirs))
                await asyncio.gather(*ret)
            elif has_ft_reqs:
                ret = []
                for tp_rank in range(self.world_size):
                    ret.append(self.model_rpcs[tp_rank].offload_adapters(adapter_to_keep))
                await asyncio.gather(*ret)
            if batch.is_clear():
                await self._remove_batch(batch)
            else:
                await self._filter_batch(batch)
        return

    async def _filter_runing_batch(self):
        if self.running_batch is not None and self.running_batch.is_clear():
            if not self.input_params.no_lora:
                # offload model and adapters
                #print(" _filter_runing_batch Offloading adapters...")
                ret = []
                for tp_rank in range(self.world_size):
                    ret.append(self.model_rpcs[tp_rank].offload_adapters())
                await asyncio.gather(*ret)

            self.running_batch = None
            return
    
    def _add_token_id_to_req(self, batch: Batch, req_ans):
        for req_id, (new_token_id, new_gen_metadata) in req_ans.items():
            req = batch.id_to_reqs[req_id]
            req.output_ids.append(new_token_id)
            req.output_metadata_list.append(new_gen_metadata)
        return
        
    def _send_to_detokenization_proc(self, batch: Batch, req_ans):
        batch_out = BatchTokenIdOut()
        for req_id, (new_token_id, new_gen_metadata) in req_ans.items():
            req = batch.id_to_reqs[req_id]
            if req.is_finetuning or req.is_reference:
                continue
            if req.has_generate_finished:
                batch_out.reqs_infs.append((req_id, new_token_id, new_gen_metadata, req.has_generate_finished, req.aborted, req.export_perf_metrics()))
            else:
                batch_out.reqs_infs.append((req_id, new_token_id, new_gen_metadata, req.has_generate_finished, req.aborted, None))
        self.send_to_detokenization.send_pyobj(batch_out)
        return

    async def loop_for_netio_req(self):
        req_counter = 0
        while True:
            recv_req = await self.recv_from_httpserver.recv_pyobj()
            if isinstance(recv_req, tuple) and len(recv_req) == 4:
                adapter_dir, prompt_ids, sampling_params, request_id = recv_req
                if self.prefill_interrupt_event is not None:
                    self.prefill_interrupt_event.set()
                self.add_req(adapter_dir, prompt_ids, sampling_params, request_id)
                req_counter += 1
            elif isinstance(recv_req, FinetuneReq):
                #self.req_queue.start_finetuning()
                if recv_req.exit_finetuning:
                    router_print("Received exit finetuning request.")
                    self.req_queue.stop_finetuning()
                else:
                    router_print("Received start finetuning request.")
                    self.req_queue.start_finetuning()
            elif isinstance(recv_req, FinetuneStatusReq):
                recv_req.finished = self.req_queue.finetuning_is_finished()
                self.send_to_detokenization.send_pyobj(recv_req)
            elif isinstance(recv_req, AbortReq):
                abort_req = recv_req
                request_id = abort_req.req_id
                await self.abort(request_id)
                self.send_to_detokenization.send_pyobj(abort_req)
            else:
                assert False, f"Error Req Inf {recv_req}"

    def clean_up(self):
        for model_rpc in self.model_rpcs:
            model_rpc.rpc_server_process.kill()
        for model_rpc in self.model_rpcs:
            model_rpc.rpc_server_process.join()
        return
    
    async def pause_backward(self):
        if not self.is_backward_running():
            return
        rets = [self.model_rpcs[tp_rank].pause_backward() for tp_rank in range(self.world_size)]
        await asyncio.gather(*rets)
    
    async def resume_backward(self):
        self.pause_printing = True
        if not self.is_backward_running():
            return
        rets = [self.model_rpcs[tp_rank].resume_backward() for tp_rank in range(self.world_size)]
        await asyncio.gather(*rets)
    
    async def shutdown_backward(self):
        rets = [self.model_rpcs[tp_rank].shutdown_backward() for tp_rank in range(self.world_size)]
        await asyncio.gather(*rets)

    async def isolated_prefill(self, batch: Batch):
        start_time = time.time()
        self.running_batch = batch
        await self.model_rpcs[0].load_adapters(batch.adapter_dirs)
        await self._init_batch(batch)
        rets = [self.model_rpcs[tp_rank].prefill_batch(batch.batch_id) for tp_rank in range(self.world_size)]
        ans = await asyncio.gather(*rets)
        if None not in ans:
            if self.world_size != 1:
                req_to_out_token_id = obtain(ans[0])
            else:
                req_to_out_token_id = ans[0]
            self._add_token_id_to_req(batch, req_to_out_token_id)
        has_new_finished_req = batch.mark_finished_req(self.eos_id)
        await self._handle_finish_req(batch, has_new_finished_req, minibatch=False)
        await self._filter_runing_batch()
        duration = time.time() - start_time
        return duration
    
    async def isolated_decode(self, batch: Batch):
        timer_start = time.time()
        rets = [self.model_rpcs[tp_rank].decode_batch(batch.batch_id) for tp_rank in range(self.world_size)]
        ans = await asyncio.gather(*rets)
        if None not in ans:
            if self.world_size != 1:
                req_to_out_token_id = obtain(ans[0])
            else:
                req_to_out_token_id = ans[0]
            self._add_token_id_to_req(batch, req_to_out_token_id)
        has_new_finished_req = batch.mark_finished_req(self.eos_id)
        await self._handle_finish_req(batch, has_new_finished_req, minibatch=False)
        duration = time.time() - timer_start
        await self._filter_runing_batch()
        return duration

    async def estimate_finetuning_overhead(self):
        if not self._check_if_finetuning_scheduler():
            return
        print("Start modeling prefill/decode time")
        inf_batches = self.profiling_batch_generator.inference_batches
        co_batches = self.profiling_batch_generator.coserving_batches
        # Run inference-only batches
        for idx, batch in enumerate(inf_batches):
            inference_tokens_list, finetuning_tokens_list = batch.export_batch_info()
            prefill_time = await self.isolated_prefill(batch)
            # Log inference-only PREFILL
            self.batch_exec_tracker.add_batch_stats(
                inference_tokens=inference_tokens_list,
                finetuning_tokens=finetuning_tokens_list,
                execution_type=BatchExecutionType.PREFILL,
                execution_duration=prefill_time,
            )
            inference_tokens_list, finetuning_tokens_list = batch.export_batch_info()
            decode_time = await self.isolated_decode(batch)
            self.batch_exec_tracker.add_batch_stats(
                inference_tokens=inference_tokens_list,
                finetuning_tokens=finetuning_tokens_list,
                execution_type=BatchExecutionType.DECODE,
                execution_duration=decode_time,
            )
        # Run co-serving batches
        for idx, batch in enumerate(co_batches):
            inference_tokens_list, finetuning_tokens_list = batch.export_batch_info()
            prefill_time = await self.isolated_prefill(batch)
            self.batch_exec_tracker.add_batch_stats(
                inference_tokens=inference_tokens_list,
                finetuning_tokens=finetuning_tokens_list,
                execution_type=BatchExecutionType.PREFILL,
                execution_duration=prefill_time,
            )
            # Reset activations
            [self.model_rpcs[tp_rank].reset_activation_pool() for tp_rank in range(self.world_size)]
        self.batch_exec_tracker.drop_batch_stats(0)
        self.batch_exec_tracker.drop_batch_stats(1)
        #self.batch_exec_tracker.print_batch_stats()
        self.prefill_estimator.data_fit(self.batch_exec_tracker)
        self.decode_estimator.data_fit(self.batch_exec_tracker)
        print(f"Error for prefill estimator: {self.prefill_estimator.fit_rmse}")
        print(f"Error for decode estimator: {self.decode_estimator.fit_rmse}")

def start_router_process(args, router_port, detokenization_port, model_rpc_ports, mode, pipe_writer):
    input_params = InputParams(max_req_total_len=args.max_req_total_len,
                               # kv cache manager parameters
                               max_total_token_num=args.max_total_token_num,
                               pool_size_lora=args.pool_size_lora,
                               batch_max_tokens=args.batch_max_tokens,
                               running_max_req_size=args.running_max_req_size,
                               # heuristic
                               swap=args.swap,
                               prefetch=args.prefetch,
                               prefetch_size=args.prefetch_size,
                               scheduler=args.scheduler,
                               profile=args.profile,
                               batch_num_adapters=args.batch_num_adapters,
                               enable_abort=args.enable_abort,
                               # mem_ratio=args.mem_ratio,
                               dummy=args.dummy,
                               no_lora_swap=args.no_lora_swap,
                               no_lora_compute=args.no_lora_compute,
                               no_kernel=args.no_kernel,
                               no_mem_pool=args.no_mem_pool,
                               bmm=args.bmm,
                               no_lora=args.no_lora,
                               fair_weights=args.fair_weights,
                               model_weightdir=args.model_dir,
                               tokenizer_mode=args.tokenizer_mode,
                               trust_remote_code=args.trust_remote_code,
                               finetuning_config=args.finetuning_config,
                               bwd_log_index = args.bwd_log_index,
                              )

    try:
        router = RouterManager(
            args.model_dir,
            args.lora_dirs,
            load_way="HF",
            world_size=args.tp,
            eos_id=args.eos_id,
            router_port=router_port,
            detokenization_port=detokenization_port,
            model_rpc_ports=model_rpc_ports,
            input_params=input_params,
            mode=mode,
            log_stats = not args.disable_log_stats,
            log_stats_interval = args.log_stats_interval,
            half_model=args.half_model,
            mem_manager_log_path=args.mem_manager_log_path,
            enable_unified_mem_manager=args.enable_unified_mem_manager,
            unified_mem_manager_max_size=args.unified_mem_manager_max_size,
            enable_gpu_profile=args.enable_gpu_profile
        )
    
        asyncio.run(router.wait_to_model_ready())
        if input_params.scheduler == "pets" and input_params.profile:
            router.req_queue.alpha = router.alpha_model
            router.req_queue.beta = router.beta_model
        elif input_params.scheduler == "pets":
            # loading from file
            cache_dir = os.path.expanduser("~/.cache/slora")
            router.req_queue.alpha = AlphaModel.from_file(cache_dir+"/profile_results.pkl")
            router.req_queue.beta = BetaModel.from_file(cache_dir+"/profile_results.pkl")
    
    except Exception as e:
        import traceback
        err_str = '\n'.join(traceback.format_exception(e))
        pipe_writer.send(err_str)
        router.clean_up()
        raise

    asyncio.run(router.estimate_finetuning_overhead())
    pipe_writer.send('init ok')
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.create_task(router.loop_for_fwd())
    loop.run_until_complete(router.loop_for_netio_req())
    return
