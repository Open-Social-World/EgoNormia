# EgoNormia: Benchmarking Physical Social Norm Understanding

[Leaderboard](https://egonormia.org) | [Blog](https://opensocial.world/articles/egonormia)

EgoNormia is a comprehensive benchmark evaluating agentic VLM capabilities in grounded reasoning scenarios.

## Features

- Comprehensive evaluation of grounded agentic abilities
- Support for onboarding and evaluation on custom dataset
- Support for both reasoning and vision-language models
- Integration with popular AI APIs
- Easy integration for custom agents

## Installation

Use conda or venv for installation. (Or install it all locally if you're brave.)

```bash
conda create -n egonormia python=3.10 -y
conda activate egonormia

git clone https://github.com/Open-Social-World/EgoNormia
cd EgoNormia
pip install -e .
```

## Evaluate locally using HuggingFace VLM

To run using a HuggingFace VLM, specify the `-hf` flag when calling the evaluation API, and give the modelname as
`--modelname [org/modelname]`, where org/modelname is the organization and name of the model as specified in HuggingFace.

## Evaluate locally using vLLM

To run using vLLM, first specify the endpoint of your vLLM server as OPENAI_API_BASE in SECRETS.env.

To tunnel a connection from a remote server to your own machine, use the following command on the vLLM host machine, replacing relevant fields as needed:
`bash ssh -i /path/to/private_key -L 8000:localhost:8000 user@remote_machine `

Then, specify the `-vl` flag when calling the evaluation API, and give the modelname as
`--modelname [org/modelname]`, where org/modelname is the organization and name of the model as specified in HuggingFace.

## Evaluate using custom VLM (if openai API compatible)

To run using a custom VLM, replace self.modelname in `eval/custom_eval_api.py` with the model name of your openai API-compatible VLM,
and fill in any remaining fields as necessary.

Then, simply evaluate using API the evaluation API with `--modelname custom`.

## Evaluate using API

To run only eval scripts, you can provide either an OpenAI API key or a Gemini API key (depending on the model you intend to run)

(To run all scripts related to EgoNormia, you need to populate *both* an OpenAI key and a Gemini API key)

This can be directly exported:

```bash
export OPENAI_API_KEY=<KEY>
export GEMINI_API_KEY=<KEY>
export GOOGLE_APPLICATION_CREDENTIALS=<PATH_TO_GCP_CREDENTIALS>
export LOCATION=<GCP_PROJECT_LOCATION>
export PROJECT_ID=<GCP_PROJECT_ID>
export AZURE_KEY=<AZURE_KEY>
export AZURE_ENDPOINT=<AZURE_ENDPOINT>
export OPENAI_API_BASE=<OPENAI_API_BASE>
```

Or you can modify the `SECRETS.env` file, adding your api keys.

You can then run the evaluation from the `egonormia/src` directory with the following command:

```bash
python3 evaluate.py --modelname [modelname] --jsonfile [jsonfile].json (--blind) (--description) (--azure) (--workers) (--number)

```
Include the `--blind` flag to run the evaluation without the ground truth, and the `--description` flag to include the description in the evaluation.
`--blind` and `--description` flags are mutually exclusive. The `--azure` flag should be set when you want to use the Azure OpenAI API.

The `--workers` flag specifies the number of workers to use for the evaluation. The `--number` flag specifies the number of samples to evaluate.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use EgoNormia in any of your work, please cite:

```
@misc{rezaei2025egonormiabenchmarkingphysicalsocial,
      title={EgoNormia: Benchmarking Physical Social Norm Understanding},
      author={MohammadHossein Rezaei and Yicheng Fu and Phil Cuvin and Caleb Ziems and Yanzhe Zhang and Hao Zhu and Diyi Yang},
      year={2025},
      eprint={2502.20490},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2502.20490},
}
```
