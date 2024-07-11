# -*- coding: utf-8 -*-
import gc
import os
import re
import torch
import openai
import torch.nn.functional as F

from peft import PeftModel
from tqdm.auto import tqdm
from functools import partial
from vllm import LLM as VLLM
from vllm import SamplingParams
from argparse import ArgumentParser
from peft.config import PeftConfigMixin
from typing import Any, Dict, List, Tuple
from datasets import load_dataset, Value, Dataset, Features
from vllm.distributed.parallel_state import destroy_model_parallel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer)
from time import sleep


def load_model(checkpoint) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:

  # get base_model_name
  base_model_name = "vaiv/GeM2-Llamion-14B-Chat"

  # load
  model = AutoModelForCausalLM.from_pretrained(
      base_model_name,
      torch_dtype="auto",
      device_map="auto")
  model = PeftModel.from_pretrained(model, checkpoint)  # TODO

  model = model.merge_and_unload()
  tokenizer = AutoTokenizer.from_pretrained(base_model_name)

  return model, tokenizer

def prompt_fn(
    examples:Dict[str, List[Any]],
    tokenizer:PreTrainedTokenizer
    ) -> Dict[str, List[Any]]:

  prompts = []
  for context, question in zip(examples["context"], examples["question"]):
    prompt = tokenizer.apply_chat_template([
        {"role": "context", "content": context},
        {"role": "question", "content": question}],
        tokenize=False)
    prompt += "<|start_header_id|>answer<|end_header_id|>\n\n"
    prompts.append(prompt)

  return {
      "prompt": prompts,
      "chosen": examples["chosen"],
      "rejected": examples["rejected"]}

def evaluate_free_generation(
    model_name_or_path:str,
    dataset:Dataset,
    chatgpt_model:str,
    chatgpt_api_key:str,
    args,
    ) -> float:

  # generation
  llm = VLLM(
      tokenizer=model_name_or_path,
      model=model_name_or_path,
      dtype="auto",
      gpu_memory_utilization=args.gpu_utils,  # TODO
      tensor_parallel_size=args.num_gpus  
      )
  # NOTE: One of the solutions against OOM when using the LongChat model is
  # to decrease the max_position_embeddings in config.json. (e.g., 200000 to 4096)

  sampling_params = SamplingParams(
      max_tokens=4096)

  outputs = llm.generate(
      dataset["prompt"],
      sampling_params)

  responses = [output.outputs[0].text for output in outputs]

  # flush vram
  del llm
  gc.collect()
  destroy_model_parallel()
  torch.distributed.destroy_process_group()
  torch.cuda.empty_cache()

  # save responses
  dataset = dataset.add_column("response", responses)
  dataset.save_to_disk(f"{model_name_or_path}/outputs")

  # g-eval
  openai.api_key = chatgpt_api_key
  gpt_scores = []

  for prompt, response in tqdm(zip(dataset["prompt"], responses), desc="G-Eval"):

    max_num_retry = args.gpt_steps
    for i_try in range(max_num_retry):

      dialog = prompt + response

      instruct = \
          "The conversation consists of a response generated by AI " \
          "based on documents retrieved in response to a user's query. " \
          "Evaluate the quality of the response in the conversation " \
          "based on factuality, helpfulness, and naturalness.\n" \
          "Factuality refers to whether the response contains " \
          "any incorrect information, conflict with the content of " \
          "the retrieved documents, or include fabricated details. " \
          "Helpfulness concerns whether the response provides " \
          "an appropriate answer to the question and is useful to the user, " \
          "without including irrelevant information. Naturalness assesses " \
          "whether there are any awkward expressions, redundant phrases, or " \
          "unnatural language use in the responses.\n" \
          "Rate the total score on a scale from 0 to 10, " \
          "considering the criterions." \
          "If the response is incomprehensible, assign the scores of 0. " \
          "Provide the evaluation in the format: 'Score: x/10' " \
          "without any additional explanation."

      messages = [
          {
              "role": "system",
              "content": "You are an objective and coherent judgement system."},
          {
              "role": "user",
              "content": dialog + "\n\n" + instruct}]
      def call(model, message):
            try:
                return openai.ChatCompletion.create(model=model,messages=message)
            except openai.error.RateLimitError as e:
                sleep(0.5)
                call(model, message)
                    
      response = call(chatgpt_model, messages)
      response = response.choices[0].message["content"]

      pattern = r'Score: \d+(\.\d+)?/\d+'
      match = re.search(pattern, response)
      if match:
        score = float(match.group().replace("Score: ", "").split("/")[0])
        gpt_scores.append(score)
        break
      elif i_try == max_num_retry - 1:
        gpt_scores.append(0)
        break
      else:
        continue # print("No match found. Retry.")

  gpt_score = sum(gpt_scores) / len(gpt_scores)

  return gpt_score

def evaluate_multiple_choice(
    model:PreTrainedModel,
    tokenizer:PreTrainedTokenizer,
    dataset:Dataset
    ) -> float:

  corrects = []

  # bsz = 1 to ensure consistency of the output logits
  for sample in tqdm(dataset, desc="eval_mc"):

    # encode
    def encode_fn(text: str) -> torch.Tensor:
      return tokenizer([text], return_tensors="pt").input_ids

    prompt_ids = encode_fn(sample["prompt"])
    chosen_ids = encode_fn(sample["chosen"] + tokenizer.eos_token)
    rejected_ids = encode_fn(sample["rejected"] + tokenizer.eos_token)

    # concat
    prompt_chosen_ids = torch.cat([prompt_ids, chosen_ids], dim=-1)
    prompt_rejected_ids = torch.cat([prompt_ids, rejected_ids], dim=-1)

    # forward to get logits
    with torch.inference_mode():
      pc_logits = model(prompt_chosen_ids.to(device=model.device)).logits
      pr_logits = model(prompt_rejected_ids.to(device=model.device)).logits
    c_logits = pc_logits[:, prompt_ids.size(-1):]
    r_logits = pr_logits[:, prompt_ids.size(-1):]

    # calculate the average negative surprisal
    def calculate_avg_neg_surprisal(
        logits:torch.Tensor, # (bsz, len, vsz)
        input_ids:torch.Tensor # (bsz, len)
        ) -> List[float]:

      target_ids = input_ids[:, 1:].contiguous() # (bsz, len - 1)

      logits = logits[:, :-1, :].contiguous() # (bsz, len - 1, vsz)
      log_probs = F.log_softmax(logits, dim=-1)
      log_probs = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1) # (bsz, len - 1)

      sum_neg_surprisal = log_probs.sum(dim=-1) # (bsz,) # surprisal = -1 * log_prob
      avg_neg_surprisal = sum_neg_surprisal / target_ids.size(-1) # divided by the maximum length, so strongly recommend to use bsz = 1

      return avg_neg_surprisal

    chosen_ans = calculate_avg_neg_surprisal(
        logits=c_logits, input_ids=chosen_ids.to(device=c_logits.device))
    rejected_ans = calculate_avg_neg_surprisal(
        logits=r_logits, input_ids=rejected_ids.to(device=r_logits.device))

    # choice and assessment
    for c_ans, r_ans in zip(chosen_ans, rejected_ans):
      if c_ans > r_ans: # the higher neg_surprisal, the less surprised
        corrects.append(1)
      else:
        corrects.append(0)

  accuracy = sum(corrects) / len(corrects)

  return accuracy

def evaluate_adapter(args, checkpoint):

  # save and load the merged model
  model_path = f".merged/{checkpoint}"
  if not os.path.exists(model_path):
      model, tokenizer = load_model(checkpoint)
      print(f"Save {model_path}")
      model.save_pretrained(model_path)
      tokenizer.save_pretrained(model_path)
  else:
      print(f"Load {model_path}")
      model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", device_map='auto')
      tokenizer = AutoTokenizer.from_pretrained(model_path)
  

  # dataset
  dataset = load_dataset(args.dataset_name, split=args.dataset_split)

  # prompt
  dataset = dataset.map(
      function=partial(prompt_fn, tokenizer=tokenizer),
      batched=True,
      batch_size=32,
      load_from_cache_file=False,
      drop_last_batch=False,
      num_proc=os.cpu_count() // 2,
      features=Features({
          "prompt": Value("string"),
          "chosen": Value("string"),
          "rejected": Value("string")}),
      remove_columns=dataset.features.keys(),
      desc="Prompting")

  # eval generation
  gpt_score = evaluate_free_generation(
      model_name_or_path=model_path,
      dataset=dataset,
      chatgpt_model=args.chatgpt_model,
      chatgpt_api_key=args.chatgpt_api_key,
      args=args)
  print(f"G-Eval = {gpt_score * 10:.2f}")
  
  # eval multiple-choice
  # model.to(device="cuda")
  accuracy = evaluate_multiple_choice(
      model=model,
      tokenizer=tokenizer,
      dataset=dataset)
  print(f"Accuracy = {accuracy * 100:.2f}")
  
  from json import dump
  
  with open('log.txt', 'a') as f:
    f.write(checkpoint + '\n')
    f.write(f'gpt: {gpt_score}\n')
    f.write(f'acc: {accuracy}\n\n')


if __name__ == "__main__":
    
  parser = ArgumentParser(add_help=False) 
  parser.add_argument("--checkpoint", type=str, required=True)
  parser.add_argument("--dataset_name", default="vaiv/ko-rag-preference")  # vaiv/ko-rag-preference
  parser.add_argument("--dataset_split", default="validation")  # validation
  parser.add_argument("--chatgpt_model", default="gpt-3.5-turbo")  # gpt-4
  parser.add_argument("--chatgpt_api_key", required=True)  # CHATGPT_API_K
  parser.add_argument("--gpt_steps", type=int, required=True, help='gpt 평가할 때, 몇번 반복해서 평균낼건지')  
  parser.add_argument("--num_gpus", type=int, required=True, help='gpu 몇개 쓸건지')  
  parser.add_argument("--gpu_utils", type=float, required=True, help='가속화 문제, 높을수록 빨라짐,  OOM뜨면 줄이셈') 
  args = parser.parse_args()
  assert args.gpu_utils < 1 and args.gpu_utils > 0, '0~1 사이 값만 가능'
  checkpoint =  'checkpoint-' + args.checkpoint
  print(checkpoint)
  print('===' * 10)
  evaluate_adapter(args, checkpoint)
