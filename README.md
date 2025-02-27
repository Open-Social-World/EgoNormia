# EgoNormia: Benchmarking Physical Social Norm Understanding

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

## Evaluate using custom VLM (if openai API compatible)

To run using a custom VLM, replace self.modelname in `eval/custom_eval_api.py` with the model name of your openai API-compatible VLM,
and fill in any remaining fields as necessary.

Then, simply evaluate using API the evaluation API with `--modelname custom`.
```bash

## Evaluate using API

To run only eval scripts, you can provide either an OpenAI API key or a Gemini API key (depending on the model you intend to run)

(To run all scripts related to EgoNormia, you need to populate *both* an OpenAI key and a Gemini API key)

This can be directly exported:

```bash
export OPENAI_API_KEY=<KEY>
export ANTHROPIC_API_KEY=<KEY>
```

Or you can modify the `SECRETS.env` file, adding your api keys.

You can then run the evaluation from the `egonormia/src` directory with the following command:

```bash
python3 evaluate.py --modelname gemini-1.5-flash-002 --jsonfile final_data.json (--blind) (--description)

```
Include the `--blind` flag to run the evaluation without the ground truth, and the `--description` flag to include the description in the evaluation.
`--blind` and `--description` flags are mutually exclusive.

## License

This project is licensed under the Apache License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use EgoNormia in any of your work, please cite:

#TODO: Update citation
```
@article{example.2023.001,
  title     = {EgoNormia},
  author    = {MohammadHossein Rezaei* and Yicheng Fu* and Philippe Cuvin* and Caleb Ziems and Yanzhe Zhang and Hao Zhu and Diyi Yang},
  journal   = {Open Social World},
  year      = {2025},
  doi       = {10.1234/example.2023.001},
  publisher = {Open Social World}
}
```
