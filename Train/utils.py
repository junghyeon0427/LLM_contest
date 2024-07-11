def causal_prompt_fn(
    examples,
    tokenizer
    ):
    prompts = []  
    for context, question, answer in zip(examples["context"], examples["question"], examples['answer']):
        prompt = tokenizer.apply_chat_template([
            {"role": "context", "content": context},
            {"role": "question", "content": question},
            {"role": "answer", "content": answer}],
            tokenize=False)
        prompt += tokenizer.eos_token
        prompts.append(prompt)

    return {
        "prompt": prompts}


def dpo_prompt_fn(
    examples,
    tokenizer
    ):
    chosens = []
    rejecteds = []
    for answer, response in zip(examples['answer'], examples['response']):
        '''
        answer: GPT
        response: Chat(baseline)
        '''
        # TODO eotid
        chosens.append(answer+tokenizer.eos_token)
        rejecteds.append(response+tokenizer.eos_token)

    return {
        "chosen": chosens,
        "rejected": rejecteds}


def orpo_prompt_fn(
    examples,
    tokenizer
    ):
    chosens = []
    rejecteds = []
    prompts = []
    for context, question, chosen, rejected in zip(examples['context'], examples['question'], examples['chosen'], examples['rejected']):
        prompt = tokenizer.apply_chat_template([
            {"role": "context", "content": context},
            {"role": "question", "content": question}],
        tokenize=False)
        prompt += "<|start_header_id|>answer<|end_header_id|>\n\n"
        prompts.append(prompt)
        chosens.append(chosen+tokenizer.eos_token)
        rejecteds.append(rejected+tokenizer.eos_token)

    return {
        "prompt": prompts,
        "chosen": chosens,
        "rejected": rejecteds}


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )
