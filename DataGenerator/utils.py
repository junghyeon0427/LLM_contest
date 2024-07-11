# -*- coding: utf-8 -*-
import openai
import json
import time

f = open('openai_token.txt', 'r') 
OPENAI_API_KEY = f.read().strip()
openai.api_key = OPENAI_API_KEY
easy_sys_message = "MISSION: You are an AI designed to provide expert-level search, summarization, and answers in Korean. Your role is to read the context thoroughly and generate detailed and accurate responses based on the provided text. You must focus on understanding the given information deeply and generate questions that require comprehensive and elaborate answers.\n\n\
GUIDELINES:\n \
1. Answer questions strictly based on the provided text without incorporating external knowledge.\n \
2. Ensure that responses are detailed, structured, and in complete sentence form.\n \
3. Generate questions and answers in JSON format.\n \
4. Formulate questions and answers in Korean.\n \
5. Generate only one pair of question and answer per interaction.\n \
6. The answer should be kind and include grounds.\n\
FORMAT:\n \
{\n \
  'question': 'Generated question'\n \
  'answer': 'Generated answer'\n \
}"
hard_sys_message = "MISSION: You are an AI designed to provide expert-level search, summarization, and answers in Korean. Your role is to read the context thoroughly and generate detailed and accurate responses based on the provided text. You must focus on understanding the given information deeply and generate questions that require comprehensive and elaborate answers.\n\n\
GUIDELINES:\n \
1. Answer questions strictly based on the provided text without incorporating external knowledge.\n \
2. Ensure that responses are detailed, structured, and in complete sentence form.\n \
3. Generate questions and answers in JSON format.\n \
4. Formulate questions and answers in Korean.\n \
5. Generate only one pair of question and answer per interaction.\n \
6. Please make an answer to the sentence more than 1000 characters.\n \
7. Ensure that questions require detailed explanations and are not in a short-answer format.\n\
8. The answer should be kind and include grounds.\n\
9. Create a question that requires complex reasoning. The longer the answer, the better, but it shouldn't contain unnecessary content.n\n\
FORMAT:\n \
{\n \
  'question': 'Generated question'\n \
  'answer': 'Generated answer'\n \
}"


# def remove_item_by_value(data, value):
#     try:
#         # 값의 인덱스를 찾기
#         index = data.index(value)
#         # 해당 인덱스의 요소를 제거
#         data.pop(index)
#         return True
#     except:
#         return False


class Dataset:
    def __init__(self, dataset:list, num:int):
        assert 0<=num and num<8
        self.dataset = dataset
        # self.filtered = 0
        
        # # 이미 생성했던 context와의 중복 제거
        # with open('generated_context.json', 'r') as f:
        #     data = json.load(f)
        #     assert isinstance(data, list)
        #     for context in data:
        #         self.filtered += remove_item_by_value(self.dataset, context)
                
        print(f'Original Dataset Size: {len(self.dataset)}')
        from random import shuffle, seed
        seed(1008)  # multi-processing을 위해 seed 고정
        self.dataset.sort()  
        shuffle(self.dataset)  # domian 다양화를 위해 shuffle
        chunk_size = len(self.dataset) // 8
        chunks = [self.dataset[i:i + chunk_size] for i in range(0, len(self.dataset), chunk_size)]
        
        assert len(chunks) > num
        self.dataset = chunks[num]
        self.len = len(self.dataset)
        print(f'Current Dataset Size: {self.len}')
        print(f'Start Context:')
        print(self.dataset[0])
        self.index = 0
        
    def next(self):
        if self.index >= self.len: 
            return None
        self.index += 1
        return self.dataset[self.index]


def gpt_call(context:str, flag:bool):
    model = "gpt-3.5-turbo"
    if flag: sys_message = hard_sys_message
    else: sys_message = easy_sys_message
    messages = [{
        "role": "system",
        "content": sys_message
    }, {
        "role": "user",
        "content": context
    }]
    
    try:
        response = openai.ChatCompletion.create(model=model, messages=messages)
    except openai.error.APIError as e:
        print(f"APIError encountered: {e}. Skipping this request.")
        return None, False
    except openai.error.RateLimitError as e:
        # RateLimitError 발생 시 일정 시간 대기 후 다시 요청
        print(f"RateLimitError encountered: {e}. Waiting and retrying...")
        time.sleep(3)  # 3초 대기 후 다시 시도
        return gpt_call(context, flag)  # 재귀 호출로 다시 시도
    except Exception as e:
        # 그 외 OpenAIError 처리 (선택 사항)
        print(f"An error occurred: {e}.")
        return None, False

    answer = response['choices'][0]['message']['content']
    try:
        answer = json.loads(answer)
        tmp = dict()
        tmp['question'] = answer['question']
        tmp['answer'] = answer['answer']
        tmp['context'] = context
        return tmp, True
    except:
        # TODO ipynb랑 비교
        tmp = [context, answer]
        return tmp, False
