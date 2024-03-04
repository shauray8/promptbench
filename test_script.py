import promptbench as pb
import torch
import json 

model = pb.LLMModel(model='migtissera/Tess-M-v1.3', max_new_tokens=1, device="cuda", dtype=torch.float16, api_key="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")

data = "mmlu"
prompts = pb.Prompt(dataset_name=data)
dataset = pb.DatasetLoader.load_dataset(data)


from tqdm import tqdm
failure = {}
res = []
for prompt in prompts[:-2]:
    preds = []
    labels = []
    for data in tqdm(dataset):
        # process input
        input_text = pb.InputProcess.basic_format(prompt, data)
        label = data['target']
        raw_pred = model(input_text, pad_token_id=2)

        # process output
        pred = list(raw_pred.split(" "))
        try:
            for ans in pred:
                if ans[0] in ["A","B","C","D"]:
                    pred = ans[0]
                    break
        except:
            pass
        if label != pred:
            if data["task"] not in failure:
                failure[data["task"]] = 1
            else:
                failure[data["task"]] += 1
        preds.append(pred)
        labels.append(label)

    # evaluate
    score = pb.Eval.compute_cls_accuracy(preds, labels)
    print(f"{score:.3f}, {prompt}, {failure.items()}")
    res.append(f"{score:.3f}, {prompt}, {failure.items()}")
    
    with open("results_mmlu.json", 'w') as f:
        json.dump(res, f, ensure_ascii=False, indent=2)


