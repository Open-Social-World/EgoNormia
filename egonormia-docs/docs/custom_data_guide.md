## Custom Data Guide
You can use the EgoNormia pipeline to process custom data and videos.

Place all videos in mp4 format in the `egonormia/src/raw` directory. The videos should be named in the format `<number>.mp4`.

Optionally, populate the `egonormia/src/raw/custom_data.json` with hand-built behaviors and justifications for each video, in the format specified by the template.

If you do not, or populate fewer than the number of videos in the `egonormia/src/raw` directory, the pipeline will instead output a blank file to `egonormia/src/final_dataset/custom_data.json`, which can then be filled in manually or populated with the generation scripts.

You can then run custom evals from evaluate.py using the `--modelname custom` flag.

### load_custom_data.py
This script is used to load custom data into the EgoNormia pipeline. The script is used to load the data into the pipeline, and the data is then processed by the pipeline scripts.

If you do not have any video data in `egonormia/src/raw/custom_data.json`, the evaluation script will not run if specified to run with the `--modelname custom` flag.
