## Eval Guide

The evaluation script is available in `egonormia/src/evaluate.py`.

Use the following command to run the script from the `egonormia/src` directory:
```bash
python3 evaluate.py --modelname [modelname] --jsonfile [jsonfile].json (--blind) (--description)
```

Include the `--blind` flag to run the evaluation without the ground truth, and the `--description` flag to include the description in the evaluation.
`--blind` and `--description` flags are mutually exclusive.

`--modelname` should be set to the model you are evaluating. The model should be compatible with the OpenAI API, the Gemini API, or the AnthropicVertex API,
and you should have the relevant API key populated in the SECRETS.env file (AnthropicVertex uses a GCP account key with permissions to use the 
models you want to evaluate).

`--jsonfile` should be set to the name of the file in `egonormia/src/final_dataset` containing the data you are evaluating.


### eval_api.py
The API for running eval. No changes should be made to this file, unless you want to change the embeddings you will use for evaluation of RAG-based models.

### custom_eval_api.py
To run using a custom VLM, replace self.modelname in `egonormia/src/eval/custom_eval_api.py` with the model name of your OpenAI API- or 
AnthropicVertex-compatible VLM.

### context_indexing.py
Helper file for the RAG-based model. No changes should be made to this file.