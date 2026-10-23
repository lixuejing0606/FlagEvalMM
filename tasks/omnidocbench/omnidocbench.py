config = dict(
    dataset_path="opendatalab/OmniDocBench",
    split="train",
    processed_dataset_path="OmniDocBench",
    processor="process.py",
)

dataset = dict(
    type="VqaBaseDataset",
    config=config,
    prompt_template=dict(
        type="PromptTemplate",
        post_prompt="",
    ),
    name="omnidocbench",
)

evaluator = dict(type="BaseEvaluator", eval_func="evaluate.py")
