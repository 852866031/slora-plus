import asyncio
import time
import uuid
import numpy as np
from typing import List, Optional

from slora.server.router.finetuning_store import FinetuningManager

# Example import if you have a local definition:
# from ..io_struct import Batch, Req
from ..io_struct import Batch, Req
from ..tokenizer import get_tokenizer
from ..input_params import FinetuneParams, SLOParams
from ..sampling_params import SamplingParams

# If using the original time calculation decorator
# from slora.utils.infer_utils import calculate_time

def get_finetuning_sampling_params() -> SamplingParams:
    """
    Return a 'dummy' sampling params object suitable for fine-tuning requests.
    By default, no sampling or advanced penalties are used.
    """
    sp = SamplingParams(
        do_sample=False,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        temperature=1.0,
        top_p=1.0,
        top_k=1,
        ignore_eos=False,     # Typically handle EOS in training
        max_new_tokens=1,   # Arbitrary placeholder
        stop_sequences=[]
    )
    return sp


def rprint(*args):
    return
    RED = "\033[91m"
    RESET = "\033[0m"
    print(RED + " ".join(str(arg) for arg in args) + RESET)

class Mixed_ReqQueue:
    """
    A queue that handles both inference requests and fine-tuning requests.
    Key differences from ReqQueue:
    - We store inference requests in `waiting_req_list`.
    - We store fine-tuning requests in `finetuning_req_list`.
    - We only add fine-tuning requests if (and only if) there are no inference
      requests waiting.
    """

    def __init__(self,
                 max_total_tokens: int,
                 batch_max_tokens: int,
                 running_max_req_size: int,
                 finetune_params: FinetuneParams,
                 slo_params: SLOParams):
        self.max_total_tokens = max_total_tokens #1024
        self.batch_max_tokens = batch_max_tokens
        self.running_max_req_size = running_max_req_size
        # config parameters
        self.finetuning_data_path = finetune_params.finetuning_data_path
        self.finetuning_prepare_size = finetune_params.finetuning_prepare_size
        self.finetuning_lora_path = finetune_params.finetuning_lora_path  
        self.max_saved_finetuning_tokens = finetune_params.max_saved_finetuning_tokens  #max size of saved activations in memory
        self.total_epoch = finetune_params.num_epochs
        self.start_task= finetune_params.start_on_launch
        self.ttft_slo = slo_params.ttft_slo
        self.avg_tbt_slo = slo_params.avg_tbt_slo
        self.max_tbt_slo = slo_params.max_tbt_slo
        print(f"\033[34m[Forward Batch Constructor]: ttft_slo={self.ttft_slo}, avg_tbt_slo={self.avg_tbt_slo}, max_tbt_slo={self.max_tbt_slo}\033[0m")

        try: 
            self.tokenizer = get_tokenizer(finetune_params.model_weightdir, 
                                           finetune_params.tokenizor_mode, 
                                           trust_remote_code=finetune_params.trust_remote_code) 
        except:
            print("Could not load tokenizer. Using default.")
            self.tokenizer = get_tokenizer("huggyllama/llama-7b", finetune_params.tokenizor_mode) 
        
        self.finetuning_manager = FinetuningManager(
            data_path=self.finetuning_data_path,
            tokenizer=self.tokenizer,
            adapter_dir=self.finetuning_lora_path,
            total_epochs=self.total_epoch,
            max_prepare=self.finetuning_prepare_size,
            trust_remote_code=finetune_params.trust_remote_code,
            max_saved_finetuning_tokens=self.max_saved_finetuning_tokens
        )
        self.finetuning_manager.load()
        self.waiting_req_list: List[Req] = []
        self.cache_len_list = []
        self.adapters = set()
        self.adapter_size = 0
        #self.prepare_finetuning_requests()
        self.prefill_estimator = None
        self.decode_estimator = None
        self.check_iter = 0
        self.last_batch_time = None

    def append(self, req: Req):
        self.waiting_req_list.append(req)
    
    def start_finetuning(self):
        self.start_task = True

    def _init_cache_list(self, current_batch: Optional[Batch], lora_ranks: dict[str, int]):
        if current_batch is not None:
            self.cache_len_list = []
            self.adapters = set()
            self.adapter_size = 0
            for req in current_batch.reqs:
                used_len = req.input_len + len(req.output_ids)
                left_len = req.max_output_len - len(req.output_ids) - 1
                self.cache_len_list.append((used_len, left_len))

                if req.adapter_dir not in self.adapters and req.adapter_dir is not None:
                    self.adapter_size += lora_ranks[req.adapter_dir] * 4
                    self.adapters.add(req.adapter_dir)
        else:
            self.cache_len_list = []
            self.adapters = set()
            self.adapter_size = 0

    async def check_will_starve(self, current_batch, decode_step_count) -> bool:
        if len(self.waiting_req_list) == 0:
            self.check_iter=0
            await asyncio.sleep(0.000000001)  # Yield to the request handler loop
        if len(self.waiting_req_list) > 0:
            if decode_step_count >20:
                return True
            predicted_next_decode_time = self.decode_estimator.predict(current_batch.input_tokens(), len(current_batch.reqs))
            predicted_next_checking_time = time.time() + 1.6 * predicted_next_decode_time
            total_pending_inf_tokens = 0
            for req in self.waiting_req_list:
                total_pending_inf_tokens += req.input_len
            predicted_next_prefill_time = self.prefill_estimator.predict_inference(total_pending_inf_tokens, len(self.waiting_req_list))
            time_left = self.get_earliest_req_time() + self.ttft_slo - (predicted_next_checking_time + predicted_next_prefill_time)
            # if time_left < 0 and self.last_batch_time is not None:
            #     print(f"Time elapsed from last batch: {time.time() - self.last_batch_time:.3f}s")
            #print(f"Num pending inf reqs: {len(self.waiting_req_list)}")
            return time_left < 0
        return False


    def _can_add_new_req(self, req: Req, lora_ranks: dict[str, int]) -> bool:
        self.cache_len_list.append((req.input_len + 1, req.max_output_len - 1))
        self.cache_len_list.sort(key=lambda x: -x[1])
    
        if req.adapter_dir not in self.adapters and req.adapter_dir is not None:
            self.adapter_size += lora_ranks[req.adapter_dir] * 4
            self.adapters.add(req.adapter_dir)
        
        left_out_len_array = np.array([e[1] for e in self.cache_len_list])
        has_run_len_array  = np.array([e[0] for e in self.cache_len_list])
        cum_run_len_array  = np.cumsum(has_run_len_array)
        size_array         = np.arange(1, len(self.cache_len_list) + 1)

        need_max_token_num = (left_out_len_array * size_array + cum_run_len_array).max()

        if (need_max_token_num < (self.max_total_tokens - self.adapter_size) and
            len(self.cache_len_list) <= self.running_max_req_size):
            return True
        else:
            return False
    
    def generate_new_batch_1(self, current_batch: Optional[Batch], lora_ranks: dict[str, int], is_backward_running: bool) -> Optional[Batch]:
        if current_batch is not None and len(current_batch.reqs) >= self.running_max_req_size:
            return None
        self._init_cache_list(current_batch, lora_ranks)
        new_batch_total_tokens = 0
        can_run_list = []
        aborted_count = 0
        earliest_inf_arrival_time = self.waiting_req_list[0].arrival_time if len(self.waiting_req_list) > 0 else time.time()
        if len(self.waiting_req_list) > 0:
            for req in self.waiting_req_list:
                if req.aborted:
                    aborted_count += 1
                    continue
                if (self._can_add_new_req(req, lora_ranks) and
                    (new_batch_total_tokens + req.input_len) <= self.batch_max_tokens):
                    can_run_list.append(req)
                    new_batch_total_tokens += req.input_len
                else:
                    break
        if len(can_run_list) > 0:
            self.waiting_req_list = self.waiting_req_list[len(can_run_list) + aborted_count:]

        infer_tokens = new_batch_total_tokens
        ft_tokens = 0
        if self.start_task and not self.finetuning_is_finished() and not is_backward_running:
            ft_list = []
            while self.finetuning_manager.has_next():
                ft_req = self.finetuning_manager.pop_next()
                if ft_req is None:
                    break
                elif not self._can_add_new_req(ft_req, lora_ranks):
                    break
                elif new_batch_total_tokens + ft_req.input_len > self.batch_max_tokens:
                    break
                elif self.finetuning_manager.pending_bwd_tokens + ft_tokens  + ft_req.input_len > self.max_saved_finetuning_tokens:
                    break
                elif not self.prefill_estimator.can_add_ft(infer_tokens, 
                                                         ft_tokens+ft_req.input_len, 
                                                         len(can_run_list)+len(ft_list), 
                                                         earliest_inf_arrival_time, 
                                                         ft_req.input_len, self.ttft_slo):
                    break
                elif current_batch is not None:
                    # check avg tbt slo
                    worst_req_last_batch = current_batch.get_req_with_worst_avg_tbt()
                    predicted_next_prefill_time = self.prefill_estimator.predict_coserving(infer_tokens, ft_tokens+req.input_len, len(can_run_list)+1)
                    predicted_next_decode_time = self.decode_estimator.predict(current_batch.input_tokens()+req.input_len+new_batch_total_tokens, len(can_run_list)+1)
                    next_token_time = predicted_next_prefill_time + predicted_next_decode_time
                    if next_token_time > self.max_tbt_slo:
                        print(f"next predicted token time {next_token_time:.4f} > {self.max_tbt_slo}, stop adding finetuning reqs")
                        break
                    worst_avg_tbt_predicted = worst_req_last_batch.avg_tbt_if_next_token(next_token_time)
                    if worst_avg_tbt_predicted > self.avg_tbt_slo:
                        print(f"worst avg tbt {worst_avg_tbt_predicted:.4f} > {self.avg_tbt_slo}, stop adding finetuning reqs")
                        break
                else:
                    ft_list.append(ft_req)
                    new_batch_total_tokens += ft_req.input_len
                    ft_tokens += ft_req.input_len
            can_run_list.extend(ft_list)
        if len(can_run_list) > 0:
            new_batch = Batch(uuid.uuid4().hex, can_run_list)
            self.print_batch_layout(infer_tokens, ft_tokens, new_batch)
            self.last_batch_time = time.time()
            return new_batch
        else:
            return None
    
    
    def print_batch_layout(self, infer_tokens, ft_tokens, new_batch):
        unused = self.batch_max_tokens - (infer_tokens + ft_tokens)
        predicted_duration = self.prefill_estimator.predict_coserving(N_inf=infer_tokens, N_ft=ft_tokens, B=len(new_batch.reqs))
        earliest_arrival_time = new_batch.get_earliest_arrival_time()   
        text = "\033[34m[Forward Batch Constructor]: "
        text += f"[{infer_tokens} Infer | {ft_tokens} FT | {unused} unused] "
        text += f"\n\tT(Predicted Prefill) = {predicted_duration:.3f}s"
        if earliest_arrival_time is not None:
            predicted_longest_ttft = time.time() + predicted_duration - earliest_arrival_time
            text += f" | T(Predicted Worst TTFT) = {predicted_longest_ttft:.3f}s"
        if self.finetuning_manager.pending_bwd_tokens > 0:
            text += f"\n\tPending BWD tokens: {self.finetuning_manager.pending_bwd_tokens}"
        text += "\033[0m"
        print(text)

    def get_req_timestamps(self, can_run_list):
        out = []
        for req in can_run_list:
            out.append(req.arrival_time)
        return out

    def get_earliest_req_time(self):
        if len(self.waiting_req_list) == 0:
            return time.time()
        return self.waiting_req_list[0].arrival_time
    
    def ready_for_bwd(self):
        return self.finetuning_manager.ready_for_bwd()

    def set_estimators(self, prefill_estimator, decode_estimator):
        self.prefill_estimator = prefill_estimator
        self.decode_estimator = decode_estimator

    def update_finetuning_status_after_fwd(self, batch: Batch):
        return self.finetuning_manager.update_finetuning_status_after_fwd(batch)

    def update_finetuning_status_after_bwd(self, loss_list, num_processed_tokens):
        return self.finetuning_manager.update_finetuning_status_after_bwd(loss_list, num_processed_tokens)

    def finetuning_is_finished(self):
        if self.start_task == False:
            return True
        return self.finetuning_manager.finetuning_is_finished()        

    def update_counter(self, req: Req):
        pass

    def generate_new_batch(self, current_batch: Optional[Batch], lora_ranks: dict[str, int], is_backward_running: bool) -> Optional[Batch]:
        if current_batch is not None and len(current_batch.reqs) >= self.running_max_req_size:
            return None
        self._init_cache_list(current_batch, lora_ranks)
        new_batch_total_tokens = 0
        can_run_list = []
        aborted_count = 0
        earliest_inf_arrival_time = self.waiting_req_list[0].arrival_time if len(self.waiting_req_list) > 0 else time.time()
        if len(self.waiting_req_list) > 0:
            for req in self.waiting_req_list:
                if req.aborted:
                    aborted_count += 1
                    continue
                if (self._can_add_new_req(req, lora_ranks) and
                    (new_batch_total_tokens + req.input_len) <= self.batch_max_tokens):
                    can_run_list.append(req)
                    new_batch_total_tokens += req.input_len
                else:
                    break
        if len(can_run_list) > 0:
            self.waiting_req_list = self.waiting_req_list[len(can_run_list) + aborted_count:]
        if len(self.waiting_req_list)>0:
            print(f"\033[34m[Forward Batch Constructor]: {len(self.waiting_req_list)} inference requests are waiting in the queue.\033[0m")
        infer_tokens = new_batch_total_tokens
        ft_tokens = 0
        if self.start_task and not self.finetuning_is_finished() and not is_backward_running:
            ft_list = []
            while self.finetuning_manager.has_next():
                restrain_batch_max_tokens = self.batch_max_tokens - new_batch_total_tokens
                restrain_backward_batch_size = self.max_saved_finetuning_tokens - self.finetuning_manager.pending_bwd_tokens - ft_tokens
                restrain_ttft_slo = self.prefill_estimator.max_next_ft_tokens(
                    infer_tokens, ft_tokens, len(can_run_list)+len(ft_list), earliest_inf_arrival_time, self.ttft_slo*0.9)
                restrain = min(restrain_batch_max_tokens, restrain_backward_batch_size, restrain_ttft_slo)
                req = self.finetuning_manager.pop_best_under(restrain, exclude=ft_list)
                if req is not None and current_batch is not None:
                    # check avg tbt slo
                    worst_req_last_batch = current_batch.get_req_with_worst_avg_tbt()
                    predicted_next_prefill_time = self.prefill_estimator.predict_coserving(infer_tokens, ft_tokens+req.input_len, len(can_run_list)+1)
                    predicted_next_decode_time = self.decode_estimator.predict(current_batch.input_tokens()+req.input_len+new_batch_total_tokens, len(can_run_list)+1)
                    next_token_time = predicted_next_prefill_time + predicted_next_decode_time
                    if next_token_time > self.max_tbt_slo:
                        print(f"next predicted token time {next_token_time:.4f} > {self.max_tbt_slo}, stop adding finetuning reqs")
                        break
                    worst_avg_tbt_predicted = worst_req_last_batch.avg_tbt_if_next_token(next_token_time)
                    if worst_avg_tbt_predicted > self.avg_tbt_slo:
                        print(f"worst avg tbt {worst_avg_tbt_predicted:.4f} > {self.avg_tbt_slo}, stop adding finetuning reqs")
                        break
                if req is None:
                    break
                else:
                    ft_list.append(req)
                    new_batch_total_tokens += req.input_len
                    ft_tokens += req.input_len
            can_run_list.extend(ft_list)
        if len(can_run_list) > 0:
            new_batch = Batch(uuid.uuid4().hex, can_run_list)
            self.print_batch_layout(infer_tokens, ft_tokens, new_batch)
            self.last_batch_time = time.time()
            return new_batch
        else:
            return None