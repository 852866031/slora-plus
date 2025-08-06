from slora.server.router.live_alignment_req_queue import LiveAlignment_ReqQueue
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
from ..io_struct import Req, Batch, BatchAbortReq
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
from slora.server.router.alignment_req_queue import Alignment_ReqQueue
from .mixed_req_queue import rprint

def get_scheduler(input_params, adapter_dirs, finetuning_adapters_tracker):
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
                                input_params.running_max_req_size, input_params.finetuning_params, finetuning_adapters_tracker)
        elif input_params.finetuning_params.finetuning_type == "Alignment":
            return Alignment_ReqQueue(input_params.max_total_token_num, input_params.batch_max_tokens,
                                input_params.running_max_req_size, input_params.finetuning_params, finetuning_adapters_tracker)
        elif input_params.finetuning_params.finetuning_type == "Alignment Live":
            return LiveAlignment_ReqQueue(input_params.max_total_token_num, input_params.batch_max_tokens,
                                input_params.running_max_req_size, input_params.finetuning_params, finetuning_adapters_tracker)
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

class FinetuningAdaptersTracker:
    def __init__(self):
        self.finetuning_adapters_status = {}
        self._lock = threading.RLock()

    def get(self, adapter_path):
        with self._lock:
            return self.finetuning_adapters_status.get(adapter_path, None)

    def set(self, adapter_path, value) -> None:
        with self._lock:
            self.finetuning_adapters_status[adapter_path] = value
    
    def all_adapters_available(self) -> bool:
        with self._lock:
            result = all(self.finetuning_adapters_status.values())
            return result

class RouterManager:
    def __init__(self, weightdir, adapter_dirs, load_way, world_size, eos_id,
                 router_port, detokenization_port, model_rpc_ports,
                 input_params,
                 mode=[], log_stats=True, log_stats_interval=10, half_model=False, mem_manager_log_path=None, enable_unified_mem_manager=False):
        self.model_weightdir = weightdir
        self.adapter_dirs = adapter_dirs
        self.world_size = world_size
        self.load_way = load_way
        self.mode = mode
        self.input_params = input_params
        self.finetuning_params = input_params.finetuning_params
        self.no_inference_since = time.time()
        self.decay_timeout = 2
        self.half_model = half_model
        self.mem_manager_log_path = mem_manager_log_path
        self.enable_unified_mem_manager = enable_unified_mem_manager

        if self.input_params.prefetch:
            self.prefetch_stream = torch.cuda.Stream()
        else:
            self.prefetch_stream = None

        pprint(input_params.__dict__)
        pprint(input_params.finetuning_params.__dict__)
        self.finetuning_adapters_tracker = FinetuningAdaptersTracker()
        self.req_queue = get_scheduler(input_params, adapter_dirs, self.finetuning_adapters_tracker)
        rprint("router: req_queue created, type", input_params.scheduler)
        # get adapter rank
        self.lora_ranks = {}
        for lora_dir in adapter_dirs:
            config, _ = get_lora_config(lora_dir, input_params.dummy)
            self.lora_ranks[lora_dir] = config["r"]
        self.lora_ranks[input_params.finetuning_params.finetuning_lora_path] = get_lora_config(adapter_dirs[-1], input_params.dummy)[0]["r"]
        self.lora_ranks[None] = 0       
        self.running_batch: Batch = None
        self.eos_id = eos_id
        self.has_wait_tokens = 0
        self.max_wait_tokens = 10
        
        context = zmq.asyncio.Context(2)
        self.recv_from_httpserver = context.socket(zmq.PULL)
        self.recv_from_httpserver.bind(f"tcp://127.0.0.1:{router_port}")
        
        self.send_to_detokenization = context.socket(zmq.PUSH)
        self.send_to_detokenization.connect(f"tcp://127.0.0.1:{detokenization_port}")
        self.model_rpc_ports = model_rpc_ports

        self.stats_tool = Stats(log_stats, log_stats_interval)
        self.backwarding = False
        self.router_status = RouterStatus.IDLE



    async def wait_to_model_ready(self):
        self.model_rpcs: List[ModelRpcClient] = []
        for rank_id in range(self.world_size):
            rpc_model = await start_model_process(port=self.model_rpc_ports[rank_id], 
                                                  world_size=self.world_size)
            self.model_rpcs.append(rpc_model)

        init_model_ret = []
        for rank_id in range(self.world_size):  # async init model process
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
                    finetuning_adapters_tracker=self.finetuning_adapters_tracker,
                    half_model=self.half_model,
                    mem_manager_log_path=self.mem_manager_log_path,
                    enable_unified_mem_manager=self.enable_unified_mem_manager
                ))

        await asyncio.gather(*init_model_ret)
        return
    
    async def profile_prefill(self):
        res = []
        for rank_id in range(self.world_size):  # async init model process
            res.append(
                self.model_rpcs[rank_id].profile_prefill())

        results = await asyncio.gather(*res)
        self.alpha_model = AlphaModel(results[0])
        self.beta_model = BetaModel(results[0])
        # check if the path exists else create it
        cache_dir = os.path.expanduser("~/.cache/slora")
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        with open(cache_dir+"/profile_results.pkl", "wb") as f:
            pickle.dump(results[0], f)
        return


    def add_req(
        self,
        adapter_dir: str,
        prompt_ids: List[int],
        sampling_params: SamplingParams,
        request_id: str
    ):
        rprint("router: add_req called")
        req = Req(adapter_dir, request_id, prompt_ids, sampling_params)
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
            await self._step(counter_count)
            counter_count += 1
            if self.running_batch is not None:
                if counter_count % 50 == 0:
                    rprint("\ncurrent batch size:", len(self.running_batch.reqs), "token used ratio:", self.running_batch.calcu_used_tokens() / self.input_params.max_total_token_num)
                    pass
                self.stats_tool.print_stats()
                
            if self.running_batch is None:
                await asyncio.sleep(0.01)  # 10ms

    async def _step(self, counter_count):
        """
        事件处理循环
        """
        if self.running_batch is None:
            new_batch = self.req_queue.generate_new_batch(self.running_batch, self.lora_ranks)
            if new_batch is None \
                and (isinstance(self.req_queue, Mixed_ReqQueue) or \
                     isinstance(self.req_queue, Alignment_ReqQueue) or isinstance(self.req_queue, LiveAlignment_ReqQueue)) \
                and not self.req_queue.finetuning_is_finished() \
                and self.req_queue.pending_bwd_tokens>0: # here we decide if it is needed to run the backward pass
                if self.decay_timeout < 0.5:
                    print(f"[router] Confident for no incoming inference, do not wait")
                    success, loss_list, procssed_tokens = await self._start_back_batch(separate_steps=False, current_epoch=self.req_queue.current_epoch)
                    if success: self.req_queue.update_finetuning_status_after_bwd(loss_list, procssed_tokens)
                elif time.time() - self.no_inference_since > self.decay_timeout:
                    print(f"[router] No inference in batch for {self.decay_timeout:.2f}s → triggering backward")
                    success, loss_list, procssed_tokens = await self._start_back_batch(separate_steps=True, current_epoch=self.req_queue.current_epoch)
                    if success: self.req_queue.update_finetuning_status_after_bwd(loss_list, procssed_tokens)
                    self.decay_timeout = self.decay_timeout * 0.75
                    return
                else:
                    return
            elif new_batch is None and isinstance(self.req_queue, LiveAlignment_ReqQueue) and \
                self.req_queue.finetuning_is_finished() and \
                (self.decay_timeout < 0.5 or time.time() - self.no_inference_since > self.decay_timeout):
                self.req_queue.check_dataset_and_load()
            elif new_batch is not None:
                if not new_batch.has_inference(): # here we check if there is any inference in the batch
                    self.no_inference_since = time.time()  # start tracking
                else:
                    self.decay_timeout = 2.0
                for req in new_batch.reqs:
                    if req.needs_to_notify_detokenize:
                        self.send_to_detokenization.send_pyobj(req.to_req_detokenization_state())
            if self.input_params.enable_abort and len(self.req_queue.abort_req_list) > 0:
                self.send_to_detokenization.send_pyobj(BatchAbortReq(self.req_queue.abort_req_list))
                self.req_queue.reset_abort_list()
                
            if new_batch is not None:
                rprint("router: new_batch formed")
                self.stats_tool.count_prompt_tokens(new_batch)
                self.running_batch = new_batch

                if not self.input_params.no_lora:
                    # load adapters
                    ret = []
                    for tp_rank in range(self.world_size):
                        ret.append(self.model_rpcs[tp_rank].load_adapters(new_batch.adapter_dirs))
                    await asyncio.gather(*ret)

                # merge adapter to base model
                if self.input_params.scheduler == "peft":
                    torch.cuda.synchronize()
                    ret = []
                    for tp_rank in range(self.world_size):
                        ret.append(self.model_rpcs[tp_rank].merge_adapter())
                    await asyncio.gather(*ret)
            
                torch.cuda.synchronize()
                result = await self._prefill_batch(self.running_batch)
                if not result:
                    self.req_queue.sample_index = self.req_queue.last_index
                await self._filter_runing_batch()
                self.has_wait_tokens = 0
            return        

        if self.has_wait_tokens < self.max_wait_tokens:
            self.stats_tool.count_output_tokens(self.running_batch)
            # prefetch
            if (not self.input_params.no_lora and
                self.input_params.prefetch and (self.has_wait_tokens == self.max_wait_tokens // 2 or
                self.has_wait_tokens == self.max_wait_tokens - 3) and self.input_params.scheduler != "peft"):
                next_batch = self.req_queue.next_batch()
                if next_batch is not None:
                    ret = []
                    for tp_rank in range(self.world_size):
                        ret.append(self.model_rpcs[tp_rank].load_adapters(
                            next_batch.adapter_dirs, prefetch=True))
                    await asyncio.gather(*ret)
            await self._decode_batch(self.running_batch)
            await self._filter_runing_batch()

            self.has_wait_tokens += 1
            return
        else:
            new_mini_batch = self.req_queue.generate_new_batch(self.running_batch, self.lora_ranks)
            if self.input_params.enable_abort and len(self.req_queue.abort_req_list) > 0:
                self.send_to_detokenization.send_pyobj(BatchAbortReq(self.req_queue.abort_req_list))
                self.req_queue.reset_abort_list()
            if new_mini_batch is not None:
                self.stats_tool.count_prompt_tokens(new_mini_batch)

                if not self.input_params.no_lora:
                    ret = []
                    for tp_rank in range(self.world_size):
                        ret.append(self.model_rpcs[tp_rank].load_adapters(new_mini_batch.adapter_dirs))
                    await asyncio.gather(*ret)

                await self._prefill_batch(new_mini_batch, minibatch=True)
                if not new_mini_batch.is_clear():
                    await self._merge_batch(self.running_batch, new_mini_batch)
                    self.running_batch.merge(new_mini_batch)
                self.has_wait_tokens = 0
            else:
                self.stats_tool.count_output_tokens(self.running_batch)
                await self._decode_batch(self.running_batch)
                await self._filter_runing_batch()
        

    async def _init_batch(self, batch: Batch):
        rprint("router: calling rpc to init batch")
        reqs = [r.to_rpc_obj() for r in batch.reqs]
        rets = [self.model_rpcs[tp_rank].init_batch(batch.batch_id, reqs) for tp_rank in range(self.world_size)]
        await asyncio.gather(*rets)
        return
        
    async def _start_back_batch(self, separate_steps=False, current_epoch=None):
        self.router_status = RouterStatus.BWD
        rets = [self.model_rpcs[tp_rank].back_batch(separate_steps, current_epoch=current_epoch) for tp_rank in range(self.world_size)]
        results = await asyncio.gather(*rets)
        self.backwarding = False
        success = True
        loss_list = []
        processed_tokens =0
        for result in results:
            finished, loss, tokens = result
            success = success and finished
            if loss!=None: loss_list.append(loss)
            if tokens!= None: processed_tokens += tokens
        self.router_status = RouterStatus.IDLE
        return success, loss_list, processed_tokens

    async def _prefill_batch(self, batch, minibatch=True):
        self.router_status = RouterStatus.PREFILL
        rprint("router: prefill batch called")
        await self._init_batch(batch)
        rets = [self.model_rpcs[tp_rank].prefill_batch(batch.batch_id) for tp_rank in range(self.world_size)]
        ans = await asyncio.gather(*rets)
        if None not in ans:
            if self.world_size != 1:
                req_to_out_token_id = obtain(ans[0])
            else:
                req_to_out_token_id = ans[0]
            self._add_token_id_to_req(batch, req_to_out_token_id)
        rprint("router: force finetuning tokens to be eos")
        count = 0
        for req in batch.reqs:
            if getattr(req, 'is_finetuning', False):
                count +=1
        if (isinstance(self.req_queue, Mixed_ReqQueue) or isinstance(self.req_queue, Alignment_ReqQueue) or isinstance(self.req_queue, LiveAlignment_ReqQueue)) \
            and None not in ans:
            self.req_queue.update_finetuning_status_after_fwd(batch)
        has_new_finished_req = batch.mark_finished_req(self.eos_id)
       
        if None not in ans:
            self._send_to_detokenization_proc(batch, req_to_out_token_id)
        await self._handle_finish_req(batch, has_new_finished_req, minibatch=True)
        self.router_status = RouterStatus.IDLE
        if None in ans:
            return False
        return True

    async def _decode_batch(self, batch:Batch):
        self.router_status = RouterStatus.DECODE
        self.req_queue.update_counter(batch)
        rets = [self.model_rpcs[tp_rank].decode_batch(batch.batch_id) for tp_rank in range(self.world_size)]
        ans = await asyncio.gather(*rets)
        if self.world_size != 1:
            req_to_out_token_id = obtain(ans[0])
        else:
            req_to_out_token_id = ans[0]
        self._add_token_id_to_req(batch, req_to_out_token_id)
        has_new_finished_req = batch.mark_finished_req(self.eos_id)
        self._send_to_detokenization_proc(batch, req_to_out_token_id)
        await self._handle_finish_req(batch, has_new_finished_req)
        self.router_status = RouterStatus.IDLE
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

    async def _handle_finish_req(self, batch: Batch, has_new_finished_req, minibatch=False):
        if has_new_finished_req:
            batch.filter_finished()

            # unmerge adapter from base model
            if self.input_params.scheduler == "peft" and batch.is_clear():
                ret = []
                for tp_rank in range(self.world_size):
                    ret.append(self.model_rpcs[tp_rank].unmerge_adapter())
                await asyncio.gather(*ret)

            if not minibatch and not self.input_params.no_lora:
                ret = []
                for tp_rank in range(self.world_size):
                    ret.append(self.model_rpcs[tp_rank].offload_adapters(batch.adapter_dirs))
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
            batch_out.reqs_infs.append((req_id, new_token_id, new_gen_metadata, req.has_generate_finished, req.aborted))
        rprint("router: outbatch size", len(batch_out.reqs_infs))
        self.send_to_detokenization.send_pyobj(batch_out)
        return

    async def loop_for_netio_req(self):
        req_counter = 0
        while True:
            recv_req = await self.recv_from_httpserver.recv_pyobj()
            if isinstance(recv_req, tuple) and len(recv_req) == 4:
                adapter_dir, prompt_ids, sampling_params, request_id = recv_req
                self.add_req(adapter_dir, prompt_ids, sampling_params, request_id)
                print(f"\033[33mrouter: received req, number {req_counter}\033[0m")
                req_counter += 1
                print("router status", self.router_status)
                if self.router_status == RouterStatus.BWD:
                    print("\033[91mBackwarding, send interrupt\033[0m")
                    for rpc in self.model_rpcs:
                        rpc.interrupt()
                elif self.router_status == RouterStatus.PREFILL:
                    print("\033[91mFinetune Forward, send interrupt\033[0m")
                    for rpc in self.model_rpcs:
                        rpc.interrupt()

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
            enable_unified_mem_manager=args.enable_unified_mem_manager
        )
    
        asyncio.run(router.wait_to_model_ready())
        if input_params.profile:
            asyncio.run(router.profile_prefill())
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

    pipe_writer.send('init ok')
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.create_task(router.loop_for_fwd())
    loop.run_until_complete(router.loop_for_netio_req())
    return
