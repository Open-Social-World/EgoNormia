## EgoNormia Generation Guide

Each script in this directory performs one step in the pipeline to generate data for the EgoNormia dataset. The scripts should
all be run in order, as they depend on the output of the previous script.

### 01_cluster_samples
Follow instructions in the script. Used to collect samples from Ego4D.

### 02_desc_gen
Run directly from within the `egonormia/src/gen` directory. Used to generate descriptions for the samples.

### 03_gen_questions
Run directly from within the `egonormia/src/gen` directory. Both Gemini and OpenAI API keys are required to be populated in the `SECRETS.env` file.

Used to generate actions, justifications, and taxonomy details for the samples.

### 04_filtering
Run directly from within the `egonormia/src/gen` directory. Used to perform normativity and blind filtering on the generated data.

### samples
The `egonormia/src/gen/samples` directory contains the generated samples, descriptions, and questions. Should be left alone.

### Prompts and Extra Prompts
Files for prompts are stored in the `egonormia/src/gen/prompts.py` file.