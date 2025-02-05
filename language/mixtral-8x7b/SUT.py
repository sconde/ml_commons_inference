import os
import time
import numpy as np
import array
import torch
from torch.nn.functional import pad
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessor,
    LogitsProcessorList,
)
from transformers.generation.streamers import BaseStreamer

import pickle
import time
import threading
import tqdm
import queue

import logging
from typing import TYPE_CHECKING, Optional, List
from pathlib import Path

import mlperf_loadgen as lg
from dataset import Dataset

import re

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("Mixtral-8x7B-Instruct-v0.1")

gen_kwargs = {
    "early_stopping": True,
    "max_new_tokens": 1024,
    "min_new_tokens": 2,
    "num_beams": 1,
    "do_sample": False,
}


class StopAfterSequence(LogitsProcessor):
    """Logits processor (to use with HuggingFace `generate()` method :
    https://huggingface.co/docs/transformers/v4.24.0/en/main_classes/
    text_generation#transformers.generation_utils.GenerationMixin).

    This logits processor makes that when the model generates a specified
    stopping sequence, it stops generating new tokens

    Args:
        stop_seq (List[int]): ID of the space token.
        eos_token_id (int): ID of the EOS token.
        device (str): Device that the model is running
    """

    def __init__(
        self,
        eos_token_id: int,
        stop_seq: List[int] = [13, 13940, 28832, 13],
        device="cpu",
    ):
        super().__init__()
        assert len(stop_seq) >= 1
        self.device = device
        self.stop_seq = torch.tensor(stop_seq, dtype=torch.long).to(device)
        self.stop_seq_length = len(stop_seq)
        self.eos_token_id = eos_token_id

    def check_stop_condition(self, input_ids: torch.LongTensor):
        stop_condition_met = (
            input_ids[:, -self.stop_seq_length:] == self.stop_seq
        ).all(dim=1)
        return stop_condition_met

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        if input_ids.size(1) > self.stop_seq_length:
            forced_eos = torch.full(
                (scores.size(1),), -float("inf")).to(self.device)
            forced_eos[self.eos_token_id] = 0
            scores[self.check_stop_condition(input_ids)] = forced_eos
        return scores


class FirstTokenStreamer(BaseStreamer):
    """Streams first tokens to a 'holder'"""

    def __init__(
        self, first_token, tokens_cache=[], is_first_token=True, response_ids=[]
    ):
        """Response ids added to 'sign' the first token"""

        self.first_token = first_token  # Queue for first token
        self.is_first_token = is_first_token

        # Cache for subsequent generated tokens
        self.tokens_cache = tokens_cache

        self.response_ids = response_ids

        # The first tokens sent to the streamer are actually the input prompts
        self.is_prompt = True

    def put(self, value):
        """Caches the tokens as they're generated. Assumes bs=1"""

        # Prompts are streamed first so we need to skip the first time value
        # that arrives
        if self.is_prompt:
            self.is_prompt = False
            return

        value = value.item()
        if self.is_first_token:

            # Add generated first token together with its query response_id to
            # first tokens queue
            self.first_token.put((value, self.response_ids[0]))

            self.is_first_token = False
            return

        self.tokens_cache.append(value)

    def end(self):
        pass

    def get_out_tokens(self):
        return self.tokens_cache


class SUT:
    def __init__(
        self,
        model_path=None,
        dtype="bfloat16",
        device="cpu",
        batch_size=None,
        total_sample_count=24576,
        dataset_path=None,
        use_cached_outputs=False,
        # Set this to True *only for test accuracy runs* in case your
        # prior session was killed partway through
        workers=1,
    ):

        self.model_path = model_path or "mistralai/Mixtral-8x7B-Instruct-v0.1"
        self.device = device

        if not batch_size:
            if device == "cpu":
                batch_size = 1
            else:
                batch_size = 32  # Reduce to 8 if using 4 GPUs, 16 for 8.
        self.batch_size = batch_size

        # dtype
        if dtype == "bfloat16":
            self.amp_enabled = True
            self.amp_dtype = torch.bfloat16
        elif dtype == "float16":
            self.amp_enabled = True
            self.amp_dtype = torch.float16
        else:
            self.amp_enabled = False
            self.amp_dtype = torch.float32

        if "cuda" in self.device:
            assert torch.cuda.is_available(), "torch gpu is not available, exiting..."

        self.dataset_path = dataset_path
        self.data_object = Dataset(
            self.model_path,
            dataset_path=self.dataset_path,
            total_sample_count=total_sample_count,
            device=self.device,
        )
        self.qsl = lg.ConstructQSL(
            self.data_object.total_sample_count,
            self.data_object.perf_count,
            self.data_object.LoadSamplesToRam,
            self.data_object.UnloadSamplesFromRam,
        )

        self.load_model()

        self.num_workers = workers
        self.worker_threads = [None] * self.num_workers
        # This event is set when any thread encounters an exception.
        self.exception_event = threading.Event()
        self.query_queue = queue.Queue()

        self.use_cached_outputs = use_cached_outputs
        self.sample_counter = 0
        self.sample_counter_lock = threading.Lock()

    def start(self):
        # Create worker threads
        for j in range(self.num_workers):
            # worker = threading.Thread(target=self.process_queries)
            worker = threading.Thread(target=self.thread_wrapper, name=f"worker-{j}")
            worker.start()
            self.worker_threads[j] = worker

    def stop(self):
        for _ in range(self.num_workers):
            self.query_queue.put(None)

        for worker in self.worker_threads:
            worker.join()

    def thread_wrapper(self):
        try:
            self.process_queries()
        except Exception as e:
            print(f"Exception in thread {threading.current_thread().name}: {e}")
            # Signal all threads to stop.
            self.exception_event.set()
            # Optionally, you can re-raise if you want to crash immediately.
            # raise

    def process_queries(self):
        """Processor of the queued queries. User may choose to add batching logic"""

        while True:
            qitem = self.query_queue.get()
            if qitem is None:
                break

            query_ids = [q.index for q in qitem]

            fname = "q" + "_".join([str(i) for i in query_ids])
            fname = f"run_outputs/{fname}.pkl"
            _p = Path(fname)
            if self.use_cached_outputs and _p.exists():
                # Read cache
                with _p.open(mode="rb") as f:
                    d = pickle.load(f)
                processed_output = d["outputs"]
                tik1 = None
                tik2 = None
                tik3 = None
                tok = None
            else:
                # Construct / collate batch
                max_seq_len = 1024

                tik1 = time.time()

                input_ids_tensor = []
                input_masks_tensor = []
                input_len = []
                input_dataset = []
                for q in qitem:
                    input_ids_tensor.append(
                        pad(
                            self.data_object.input_ids[q.index],
                            (
                                max_seq_len -
                                self.data_object.input_lens[q.index],
                                0,
                                0,
                                0,
                            ),
                            value=self.tokenizer.pad_token_id,
                        )
                    )
                    input_masks_tensor.append(
                        pad(
                            self.data_object.attention_masks[q.index],
                            (
                                max_seq_len -
                                self.data_object.input_lens[q.index],
                                0,
                                0,
                                0,
                            ),
                            value=0,
                        )
                    )
                    input_len.append(self.data_object.input_lens[q.index])

                    # In case we predict code generation, we can specify an
                    # additional stop sequence
                    input_dataset.append(
                        self.data_object.dataset_names[q.index])
                input_ids_tensor = torch.cat(input_ids_tensor)
                input_masks_tensor = torch.cat(input_masks_tensor)

                assert input_ids_tensor.shape == input_masks_tensor.shape
                assert input_ids_tensor.shape[0] <= self.batch_size

                tik2 = time.time()
                logits_processor = LogitsProcessorList(
                    [StopAfterSequence(
                        self.tokenizer.eos_token_id, device=self.device)]
                )
                for i in range(len(input_ids_tensor)):
                    ids, masks, dataset = (
                        input_ids_tensor[i: i + 1],
                        input_masks_tensor[i: i + 1],
                        input_dataset[i],
                    )
                    pred_output_tokens = []
                    if dataset == "MBXP":
                        out = self.model.generate(
                            input_ids=ids,
                            attention_mask=masks,
                            pad_token_id=self.tokenizer.pad_token_id,
                            logits_processor=logits_processor,
                            **gen_kwargs,
                        )
                    else:
                        out = self.model.generate(
                            input_ids=ids,
                            attention_mask=masks,
                            pad_token_id=self.tokenizer.pad_token_id,
                            **gen_kwargs,
                        )
                    pred_output_tokens.append(out)
                pred_output_tokens = torch.cat(pred_output_tokens)
                tik3 = time.time()

                processed_output = self.data_object.postProcess(
                    pred_output_tokens,
                    input_seq_lens=input_len,
                    query_id_list=query_ids,
                )

            for i in range(len(qitem)):
                n_tokens = processed_output[i].shape[0]
                response_array = array.array(
                    "B", processed_output[i].tobytes())
                bi = response_array.buffer_info()
                response = [
                    lg.QuerySampleResponse(
                        qitem[i].id,
                        bi[0],
                        bi[1],
                        n_tokens)]
                lg.QuerySamplesComplete(response)

            tok = time.time()

            with self.sample_counter_lock:
                self.sample_counter += len(qitem)
                print(f"Samples run: {self.sample_counter}")
                if tik1:
                    print(f"\tBatchMaker time: {tik2 - tik1}")
                    print(f"\tInference time: {tik3 - tik2}")
                    print(f"\tPostprocess time: {tok - tik3}")
                    print(f"\t==== Total time: {tok - tik1}")
                else:
                    print(f"\tLoaded from cache: {_p}")

    def print_model_size(self):
        # Calculate the total number of parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Total number of parameters: {total_params:,}")

        # Determine bytes per parameter based on the dtype of the first parameter.
        # This is a rough estimate; if your model uses mixed dtypes, you might want to iterate over all.
        first_param_dtype = next(self.model.parameters()).dtype
        if first_param_dtype == torch.float32:
            bytes_per_param = 4
        elif first_param_dtype == torch.float16 or first_param_dtype == torch.bfloat16:
            bytes_per_param = 2
        elif first_param_dtype == torch.int8:
            bytes_per_param = 1
        else:
            # Default to 4 bytes if unknown
            bytes_per_param = 4

        # Calculate total size in bytes
        total_bytes = total_params * bytes_per_param
        # Convert to gigabytes (1 GB = 1024^3 bytes)
        total_gb = total_bytes / (1024 ** 3)
        print(f"Approximate model size (parameters only): {total_gb:.4f} GB")

    def print_model_device_sizes(self):
        # Dictionary to hold total size per device (in bytes)
        device_sizes = {}
        # Dictionary to record which expert IDs are found on each device.
        # Keys are device strings, values are sets of expert identifiers.
        expert_devices = {}

        def add_to_device_sizes(name, tensor):
            dev = str(tensor.device)
            size_bytes = tensor.numel() * tensor.element_size()
            # Accumulate the size per device
            device_sizes[dev] = device_sizes.get(dev, 0) + size_bytes

            # Check if this tensor belongs to an expert module.
            # This regex looks for "expert.<id>" or "experts.<id>" in the parameter name.
            match = re.search(r'expert(?:s)?\.(\d+)', name, re.IGNORECASE)
            if match:
                expert_id = match.group(1)
                # Add the expert id to the set for this device.
                expert_devices.setdefault(dev, set()).add(expert_id)

            # Print the individual tensor information in MB.
            size_mb = size_bytes / (1024 ** 2)
            print(f"{name:50s} on {dev:8s} | {size_mb:6.2f} MB")

        print("Parameters:")
        for name, param in self.model.named_parameters():
            add_to_device_sizes(name, param)

        print("\nBuffers:")
        for name, buf in self.model.named_buffers():
            add_to_device_sizes(name, buf)

        print("\nTotal memory per device:")
        for dev, size in device_sizes.items():
            size_gb = size / (1024 ** 3)
            print(f"Device {dev:8s}: {size_gb:.4f} GB")

        print("\nExpert summary:")
        if expert_devices:
            for dev, experts in expert_devices.items():
                experts_list = sorted(experts, key=lambda x: int(x))
                print(f"Device {dev:8s}: {len(experts)} experts -> {experts_list}")
        else:
            print("No experts found in the model parameters or buffers.")


    def load_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map="auto",
            low_cpu_mem_usage=True,
            torch_dtype=self.amp_dtype,
        )
        print("Loaded model")

        # Usage example, assuming self.model is already loaded:
        self.print_model_size()

        # Example usage (assuming model is already loaded):
        self.print_model_device_sizes()

        self.device = torch.device(self.device)
        if self.device == "cpu":
            # Force CPU if your system has GPU and you specifically want
            # CPU-only run
            self.model = self.model.to(self.device)

        self.model.eval()
        try:  # for systems with low ram, the below command gives error as some part is offloaded to disk
            self.model = self.model.to(memory_format=torch.channels_last)
        except Exception as e:
            # Catch the exception, print what it gives, and continue.
            print(f"Warning: Failed to set memory format to torch.channels_last: {e}")
            pass

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            model_max_length=1024,
            padding_side="left",
            use_fast=False,
        )

        self.tokenizer.pad_token = self.tokenizer.eos_token
        print("Loaded tokenizer")

    def get_sut(self):
        self.sut = lg.ConstructSUT(self.issue_queries, self.flush_queries)
        return self.sut

    def get_qsl(self):
        return self.qsl

    def predict(self, **kwargs):
        raise NotImplementedError

    def issue_queries(self, query_samples):
        """Receives samples from loadgen and adds them to queue. Users may choose to batch here"""

        list_prompts_tokens = []
        list_prompts_attn_masks = []

        print(f"IssueQuery started with {len(query_samples)} samples")
        while len(query_samples) > 0:
            self.query_queue.put(query_samples[: self.batch_size])
            query_samples = query_samples[self.batch_size:]
        print(f"IssueQuery done")

    def flush_queries(self):
        pass

    def __del__(self):
        pass


class SUTServer(SUT):
    def __init__(
        self,
        model_path=None,
        dtype="bfloat16",
        device="cpu",
        total_sample_count=24576,
        dataset_path=None,
        workers=1,
    ):

        super().__init__(
            model_path=model_path,
            dtype=dtype,
            device=device,
            total_sample_count=total_sample_count,
            dataset_path=dataset_path,
            workers=workers,
        )

        self.first_token_queue = queue.Queue()

    def start(self):

        # Create worker threads
        for j in range(self.num_workers):
            worker = threading.Thread(target=self.process_queries)
            worker.start()
            self.worker_threads[j] = worker

        # Create first token response thread
        self.ft_response_thread = threading.Thread(
            target=self.process_first_tokens)
        self.ft_response_thread.start()

    def process_first_tokens(self):

        while True:
            first_token_item = self.first_token_queue.get()

            if first_token_item is None:
                log.info("Exiting First token response thread")
                break

            first_tokens, response_id = first_token_item

            response_data = array.array(
                "B", np.array(
                    first_tokens, np.int32).tobytes())
            bi = response_data.buffer_info()
            response = [lg.QuerySampleResponse(response_id, bi[0], bi[1])]
            lg.FirstTokenComplete(response)

    def process_queries(self):
        """Processor of the queued queries. User may choose to add batching logic"""
        while True:

            qitem = self.query_queue.get()
            if qitem is None:
                break

            input_ids_tensor = self.data_object.input_ids[qitem.index]
            input_masks_tensor = self.data_object.attention_masks[qitem.index]
            dataset = self.data_object.dataset_names[qitem.index]

            # TODO: This PoC is super slow with significant overhead. Best to
            # create a patch to `generate`
            tokens_cache = []
            tokens_streamer = FirstTokenStreamer(
                self.first_token_queue,
                tokens_cache=tokens_cache,
                is_first_token=True,
                response_ids=[qitem.id],
            )

            logits_processor = LogitsProcessorList(
                [StopAfterSequence(
                    self.tokenizer.eos_token_id, device=self.device)]
            )
            if dataset == "MBXP":
                _ = self.model.generate(
                    input_ids=input_ids_tensor,
                    attention_mask=input_masks_tensor,
                    pad_token_id=self.tokenizer.pad_token_id,
                    streamer=tokens_streamer,
                    logits_processor=logits_processor,
                    **gen_kwargs,
                )
            else:
                _ = self.model.generate(
                    input_ids=input_ids_tensor,
                    attention_mask=input_masks_tensor,
                    pad_token_id=self.tokenizer.pad_token_id,
                    streamer=tokens_streamer,
                    **gen_kwargs,
                )

            output_tokens = tokens_streamer.get_out_tokens()
            n_tokens = len(output_tokens)
            response_array = array.array(
                "B", np.array(output_tokens, np.int32).tobytes()
            )
            bi = response_array.buffer_info()
            response = [
                lg.QuerySampleResponse(
                    qitem.id,
                    bi[0],
                    bi[1],
                    n_tokens)]
            lg.QuerySamplesComplete(response)

    def issue_queries(self, query_samples):

        self.query_queue.put(query_samples[0])

    def stop(self):
        for _ in range(self.num_workers):
            self.query_queue.put(None)

        for worker in self.worker_threads:
            worker.join()

        self.first_token_queue.put(None)
        self.ft_response_thread.join()
