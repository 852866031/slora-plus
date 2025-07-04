import copy
import csv
import uuid
import numpy as np
from typing import List, Optional

# Example import if you have a local definition:
# from ..io_struct import Batch, Req
from ..io_struct import Batch, Req
from ..tokenizer import get_tokenizer
from ..input_params import FinetuneParams
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

class Alignment_ReqQueue:
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
                 finetuning_adapters_tracker):
        self.max_total_tokens = max_total_tokens #1024
        self.batch_max_tokens = batch_max_tokens
        self.running_max_req_size = running_max_req_size
        # config parameters
        self.finetuning_data_path = finetune_params.finetuning_data_path
        self.finetuning_prepare_size = finetune_params.finetuning_prepare_size
        self.finetuning_lora_path = finetune_params.finetuning_lora_path
        self.reference_lora_path = finetune_params.reference_lora_path
        self.max_saved_finetuning_tokens = finetune_params.max_saved_finetuning_tokens  #max size of saved activations in memory
        self.max_finetuning_tokens_in_batch = finetune_params.max_finetuning_tokens_in_batch #max size of finetuning tokens in a forward batch
        self.total_epoch = finetune_params.num_epochs
        self.finetuning_adapters_tracker = finetuning_adapters_tracker
        # tracker variables
        self.finetuning_tokens_in_memory = 0 #
        self.finetuning_tokens_processed = 0
        self.pending_bwd_tokens = 0
        self.epoch_avg_kto_loss_list = []
        self.epoch_avg_completion_loss_list = []
        self.kto_loss_list =[]
        self.completion_loss_list = []
        self.current_epoch = 0
        self.sample_index = 0
        self.last_index = 0
        self.flag = True
        # prepare size is the number of finetuning requests to load into cpu
        # finetuning processed size is the number of requests being forwarded and waiting for backward 
        
        try: 
            self.tokenizer = get_tokenizer(finetune_params.model_weightdir, 
                                           finetune_params.tokenizor_mode, 
                                           trust_remote_code=finetune_params.trust_remote_code) 
        except:
            print("Could not load tokenizer. Using default.")
            self.tokenizer = get_tokenizer("huggyllama/llama-7b", finetune_params.tokenizor_mode) 

        # Holds waiting inference requests
        self.waiting_req_list: List[Req] = []
        # Holds waiting fine-tuning requests
        self.finetuning_req_list: List[Req] = []
        self.reference_req_list: List[Req] = []
        
        # Used internally to track concurrency usage when building a batch
        self.cache_len_list = []
        self.adapters = set()
        self.adapter_size = 0
        print("Initializing Alignment_ReqQueue")
        print("Initializing Finetuning Settings")
        print(f"Finetuning data path: {self.finetuning_data_path}")
        print(f"Finetuning lora path: {self.finetuning_lora_path}")
        print(f"Finetuning prepare size: {self.finetuning_prepare_size}")
        print(f"Max saved finetuning tokens: {self.max_saved_finetuning_tokens}")
        print(f"batch max tokens: {self.batch_max_tokens}")
        self.prepare_finetuning_requests()

        
    def update_finetuning_status_after_fwd(self, batch: Batch):
        for req in batch.reqs:
            if req.is_finetuning:
                self.pending_bwd_tokens += req.input_len


    def update_finetuning_status_after_bwd(self, loss_list, num_processed_tokens):
        for losses in loss_list:
            self.kto_loss_list.append(losses[0])
            self.completion_loss_list.append(losses[1])
        self.finetuning_tokens_processed += num_processed_tokens
        self.pending_bwd_tokens -= num_processed_tokens
        #print bar
        bar_width = 50
        ratio = self.finetuning_tokens_processed / max(self.finetuning_tokens_in_memory, 1)
        filled_len = int(bar_width * ratio)
        empty_len = bar_width - filled_len
        grey = "*"
        white = " "
        bar = grey * filled_len + white * empty_len
        print(f"Epoch: {self.current_epoch+1}/{self.total_epoch} [{bar}] {self.finetuning_tokens_processed}/ {self.finetuning_tokens_in_memory} {ratio:.1%} processed")
        if self.finetuning_tokens_processed >= self.finetuning_tokens_in_memory:
            self.epoch_avg_kto_loss_list.append(np.mean(self.kto_loss_list))
            self.epoch_avg_completion_loss_list.append(np.mean(self.completion_loss_list))
            print(f"Epoch {self.current_epoch} finished. Average KTO Loss: {self.epoch_avg_kto_loss_list[-1]:.6f}, Average Completion Loss: {self.epoch_avg_completion_loss_list[-1]:.6f}\n")
            for i, loss in enumerate(self.kto_loss_list):
                kto_loss = loss
                completion_loss = self.completion_loss_list[i]
                print(f"Epoch {self.current_epoch} Batch {i}: KTO Loss = {kto_loss:.6f}, Completion Loss = {completion_loss:.6f}")
            self.kto_loss_list = []
            self.completion_loss_list = []
            self.current_epoch += 1
            if self.current_epoch >= self.total_epoch:
                print("=== Loss List ===")
                for i, loss in enumerate(self.epoch_avg_kto_loss_list):
                    print(f"Backward Epoch {i}: KTO Loss = {loss:.6f}, Completion Loss = {self.epoch_avg_completion_loss_list[i]:.6f}")
                print("=== End of Loss List ===")
            else:
                self.sample_index = 0
                self.finetuning_tokens_processed = 0
        else:
            print()

    def finetuning_is_finished(self):
        if self.current_epoch >= self.total_epoch:
            return True
        else:
            return False
        

    def append(self, req: Req):
        self.waiting_req_list.append(req)

    def prepare_finetuning_requests(self):
        loaded_count = 0
        try:
            with open(self.finetuning_data_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    prompt      = row["prompt"].strip()
                    completion  = row["completion"].strip()
                    label       = int(row["label"])

                    if not prompt or not completion:
                        continue
                    prompt_enc      = self.tokenizer(prompt, add_special_tokens=False)
                    completion_enc  = self.tokenizer(completion, add_special_tokens=False)
                    prompt_ids      = prompt_enc.get("input_ids", [])
                    completion_ids  = completion_enc.get("input_ids", [])
                    full_ids        = prompt_ids + completion_ids     # concat
                    completion_mask = [False] * (len(prompt_ids)-1) + [True] * len(completion_ids)
                    text = prompt + " " + completion
                    finetune_req = Req(
                        adapter_dir=self.finetuning_lora_path,
                        request_id=uuid.uuid4().hex,
                        prompt_ids=full_ids,
                        sample_params=get_finetuning_sampling_params(),
                        is_finetuning=True,
                        text=text,
                    )
                    self.finetuning_req_list.append(finetune_req)
                    self.finetuning_tokens_in_memory += finetune_req.input_len
                    loaded_count += 1

                    reference_req = Req(
                        adapter_dir=self.reference_lora_path,
                        request_id=uuid.uuid4().hex,
                        prompt_ids=copy.deepcopy(full_ids),
                        completion_mask=completion_mask,
                        label=label,
                        sample_params=get_finetuning_sampling_params(),
                        is_finetuning=False,
                        is_reference=True,
                        text=str(text),
                    )
                    self.reference_req_list.append(reference_req)
                    if loaded_count >= self.finetuning_prepare_size:
                        break

            print(f"Loaded {loaded_count} fine-tuning requests from CSV.")
        except FileNotFoundError:
            print(f"finetuning_data_path not found: {self.finetuning_data_path}")
        except Exception as e:
            print(f"Error while reading or tokenising fine-tuning data: {e}")
        
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

    def update_counter(self, req: Req):
        pass

    def generate_new_batch(self, current_batch: Optional[Batch], lora_ranks: dict[str, int]) -> Optional[Batch]:
        """
        Generates a new batch. Priority:
          1) If inference requests exist in `self.waiting_req_list`, only fill them.
          2) Otherwise, fill with fine-tuning requests from `self.finetuning_req_list`.
        """
        if current_batch is not None and len(current_batch.reqs) >= self.running_max_req_size:
            return None

        self._init_cache_list(current_batch, lora_ranks)
        new_batch_total_tokens = 0
        can_run_list = []
        aborted_count = 0
        # 1) If we have more than one inference requests, try them first
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
        
        new_batch_inference_tokens = new_batch_total_tokens
        #print(f"Adapter avaiable: {self.finetuning_adapters_tracker.all_adapters_available()}")
        if new_batch_total_tokens * 1.2 < self.batch_max_tokens and len(self.finetuning_req_list)> 0 and self.finetuning_adapters_tracker.all_adapters_available():
            new_batch_finetuning_tokens = 0
            self.last_index = self.sample_index
            for i in range(self.sample_index, len(self.finetuning_req_list)):
                req = self.finetuning_req_list[i]
                ref_req = self.reference_req_list[i]
                if new_batch_total_tokens + req.input_len + ref_req.input_len > self.batch_max_tokens:
                    #print("1 Cannot add more finetuning requests, breaking")
                    break
                elif new_batch_finetuning_tokens + req.input_len  + ref_req.input_len > self.max_finetuning_tokens_in_batch:
                    #print("2 Cannot add more finetuning requests, breaking")
                    self.flag = False
                    break
                elif self.pending_bwd_tokens + req.input_len > self.max_saved_finetuning_tokens:
                    #print("3 Cannot add more finetuning requests, breaking")
                    self.flag = False
                    break
                elif new_batch_inference_tokens!= 0 and new_batch_finetuning_tokens + req.input_len + ref_req.input_len > new_batch_inference_tokens*2:
                    #print("4 Cannot add more finetuning requests, breaking")
                    self.flag = False
                    break
                else:
                    can_run_list.append(req)
                    can_run_list.append(ref_req)
                    new_batch_total_tokens += req.input_len + ref_req.input_len
                    new_batch_finetuning_tokens += req.input_len + ref_req.input_len
                    self.sample_index += 1
                    self.flag = True

        if len(can_run_list) > 0:
            infer_tokens = 0
            finetune_tokens = 0
            ref_tokens = 0
            for req in can_run_list:
                if req.is_finetuning:
                    finetune_tokens += req.input_len
                elif req.is_reference:
                    ref_tokens += req.input_len
                else:
                    infer_tokens += req.input_len
            unused = self.batch_max_tokens - (infer_tokens + finetune_tokens)
            print(f"\033[34mForward Batch Token Layout: [{infer_tokens} infer tokens | {finetune_tokens} finetune + {ref_tokens} reference (Sum Max: {self.max_finetuning_tokens_in_batch}) | {unused} unused] \033[0m")
            new_batch = Batch(uuid.uuid4().hex, can_run_list)
            self.waiting_req_list = self.waiting_req_list[len(can_run_list) + aborted_count:]
            return new_batch
        else:
            return None


    def next_batch(self) -> Optional[Batch]:
        if len(self.waiting_req_list) > 0:
            next_batch_reqs = []
            new_batch_total_tokens = 0
            aborted_count = 0

            for req in self.waiting_req_list:
                if req.aborted:
                    aborted_count += 1
                    continue
                if new_batch_total_tokens + req.input_len <= self.batch_max_tokens:
                    next_batch_reqs.append(req)
                    new_batch_total_tokens += req.input_len
                else:
                    break

            if len(next_batch_reqs) > 0:
                new_batch = Batch(uuid.uuid4().hex, next_batch_reqs)
                self.waiting_req_list = self.waiting_req_list[len(next_batch_reqs) + aborted_count:]
                return new_batch
            else:
                return None
        else:
            next_batch_reqs = []
            new_batch_total_tokens = 0
            aborted_count = 0

            for req in self.finetuning_req_list:
                if req.aborted:
                    aborted_count += 1
                    continue
                if new_batch_total_tokens + req.input_len <= self.batch_max_tokens:
                    next_batch_reqs.append(req)
                    new_batch_total_tokens += req.input_len
                else:
                    break

            if len(next_batch_reqs) > 0:
                new_batch = Batch(uuid.uuid4().hex, next_batch_reqs)
                self.finetuning_req_list = self.finetuning_req_list[len(next_batch_reqs) + aborted_count:]
                return new_batch
            else:
                return None