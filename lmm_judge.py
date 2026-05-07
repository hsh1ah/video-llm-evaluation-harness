import os, re
import time
import json, argparse
from load_longvideobench import LongVideoBenchDataset
from concurrent.futures import ProcessPoolExecutor, as_completed
from call_gpt4o import request
from utils import dump_jsonl


# Global variable for video_data
video_data = LongVideoBenchDataset(os.getenv('LVB_PATH'), "lvb_test_wo_gt.json", max_num_frames=128)


PROMPTS = {
    "role": """**Remember: You are watching a Video.**

A user, characterized by a specific persona, is interacting with two AI assistant models (A and B) to better understand video content using the same question. Here is the user's persona:
```persona
{persona}
```

The user's question is:
```question
{question}
```

The response from Model A is:
```model_a
{answer_a}
```

The response from Model B is:
```model_b
{answer_b}
```

Please act as an impartial judge and carefully evaluate the responses of Model A and Model B to determine which one is better. Use the following standards:

1. [Instruction Following]: The response should closely adhere to the user's instructions, ensuring it directly addresses the specified task.
2. [Accuracy]: The response must accurately utilize information from the video, avoiding fabrication or misquotation. It should maintain factual correctness, avoid hallucinations, and demonstrate contextual coherence with precise terminology and knowledge.
3. [Relevance]: The response should consider the user's background information and needs, providing a comprehensive, detailed answer that addresses the question directly without straying off-topic. Responses should be thorough, offering multiple perspectives where relevant.
4. [Helpfulness]: The response should provide valuable information to aid the user in understanding or solving their issue, avoiding irrelevant or vague content.

If the responses from Model A and Model B are of similar quality (whether both are good or both are bad), you may declare a tie.

**Please follow these steps for your judgment:**

- Step 1: Analyze which model provides a better response for the [Instruction Following] standard.
- Step 2: Analyze which model provides a better response for the [Accuracy] standard.
- Step 3: Analyze which model provides a better response for the [Relevance] standard.
- Step 4: Analyze which model provides a better response for the [Helpfulness] standard.
- Step 5: Based on the results from Steps 1-4, determine the overall outcome: Model A, Model B, Tie (both are good), or Tie (both are bad).

Please respond strictly in the following format:

```[Instruction Following]
[Your Analysis]
```

```[Accuracy]
[Your Analysis]
```

```[Relevance]
[Your Analysis]
```

```[Helpfulness]
[Your Analysis]
```

```[Overall Judge]
A/B/Tie
```"""
}

def response_parse(text):
    """Parse the response text into a dictionary."""
    pattern = re.compile(r'\[(.*?)\](.*?)(?=\[|$)', re.DOTALL)
    content_dict = {}
    
    for match in pattern.finditer(text):
        key = match.group(1).strip()
        value = match.group(2).strip()
        value = re.sub(r'^```\n?|\n?```$', '', value)
        content_dict[key] = value
    
    return content_dict

def run_one_prompt(paths):
    """Process a single prompt and save the result."""
    idx, sample, output_dir = paths
    video_id = sample["video_id"]
    qid = sample["qid"]
    persona = sample["persona"]
    question = sample["question"]
    model_a_answer = sample["model a answer"]
    model_b_answer = sample["model b answer"]
    output_path = os.path.join(output_dir, f'{qid}.jsonl')
    
    if os.path.exists(output_path):
        print(f'{output_path} already exists, skipping...')
        return

    prompt = PROMPTS["role"].format(
        persona=persona,
        question=question,
        answer_a=model_a_answer,
        answer_b=model_b_answer
    )

    response = request(prompt)
    parsed = response_parse(response)
    
    result = {
        'qid': qid,
        'video_id': video_id,
        'persona': persona,
        'question': question,
        'model_a_answer': model_a_answer,
        'model_b_answer': model_b_answer,
        'raw_response': response,
        'parsed': parsed
    }
    
    dump_jsonl(output_path, [result])
    return result


def main(args):
    with open(args.battle_path, 'r') as f:
        battles = json.load(f)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    tasks = [(idx, sample, args.output_dir) for idx, sample in enumerate(battles)]
    
    with ProcessPoolExecutor(max_workers=args.worker_num) as executor:
        futures = [executor.submit(run_one_prompt, task) for task in tasks]
        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as e:
                print(f"Error processing task: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--battle_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--worker_num', type=int, default=8)
    args = parser.parse_args()
    main(args)