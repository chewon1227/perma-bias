from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import json

model_name = "Qwen/Qwen2.5-72B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def gen(prompt):
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=8192
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

riasec_list = [
    ["I am good at working independently.", "A"],
    ["I like to read about art and music.", "A"],
    ["I enjoy creative writing.", "A"],
    ["I am a creative person.", "A"],
    ["I like to play instruments or sing.", "A"],
    ["I like acting in plays.", "A"],
    ["I like to draw.", "A"],
    ["I like to organize things, (files, desks/offices).", "C"],
    ["I like to have clear instructions to follow.", "C"],
    ["I wouldn't mind working 8 hours per day in an office.", "C"],
    ["I pay attention to details.", "C"],
    ["I like to do filing or typing.", "C"],
    ["I am good at keeping records of my work.", "C"],
    ["I would like to work in an office.", "C"],
    ["I am an ambitious person, I set goals for myself.", "E"],
    ["I like to try to influence or persuade people.", "E"],
    ["I like selling things.", "E"],
    ["I am quick to take on new responsibilities.", "E"],
    ["I would like to start my own business.", "E"],
    ["I like to lead.", "E"],
    ["I like to give speeches.", "E"],
    ["I like to do puzzles.", "I"],
    ["I like to do experiments.", "I"],
    ["I enjoy science.", "I"],
    ["I enjoy trying to figure out how things work.", "I"],
    ["I like to analyze things (problems/situations).", "I"],
    ["I like working with numbers or charts.", "I"],
    ["I'm good at math.", "I"],
    ["I like to work on cars.", "R"],
    ["I like to build things.", "R"],
    ["I like to take care of animals.", "R"],
    ["I like putting things together or assembling things.", "R"],
    ["I like to cook.", "R"],
    ["I am a practical person.", "R"],
    ["I like working outdoors.", "R"],
    ["I like to work in teams.", "S"],
    ["I like to teach or train people.", "S"],
    ["I like trying to help people solve their problems.", "S"],
    ["I am interested in healing people.", "S"],
    ["I enjoy learning about other cultures.", "S"],
    ["I like to get into discussions about issues.", "S"],
    ["I like helping people.", "S"],
]

augmented_riasec_list = []
for riasec in tqdm(riasec_list):
    prompt = "Paraphrase the given sentence into 10 sentences each. Do not say anything else. Do not number them.\n"
    prompt += f"{riasec[0]}\n"
    response = gen(prompt)
    for sentence in response.split("\n"):
        augmented_riasec_list.append([sentence, riasec[0], riasec[1]])

with open('data/riasec-augmented-job-data.json','w',encoding='utf-8') as f:
    json.dump(augmented_riasec_list,f,indent=2,ensure_ascii=False)
