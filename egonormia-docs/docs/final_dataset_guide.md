## Dataset Guide

Final data (for EgoNormia, the out-of-domain robotics dataset, and your custom dataset) should be stored in the final_dataset directory. 
 
Use the following command to run the score computation script from the `egonormia/src/final_dataset` directory:
```bash
python3 compute_scores_final.py --file final_data.json
```

`--file` is a required argument to specify the file you are evaluating. The file should be the name (not path) of a json in the final_dataset directory.


### egonormia/final_dataset/final_data|eval.json
The EgoNormia datapoints and evaluation results are stored in this file. This file is used to evaluate the performance of the models on the EgoNormia dataset.

### egonormia/final_dataset/ood_robot|eval.json
The out-of-domain robotics datapoints and evaluation results are stored in this file. This file is used to evaluate the performance of the models on the out-of-domain robotics dataset.

### egonormia/final_dataset/compute_scores_final.py
The script for computing the scores for the final dataset. No changes should be made to this file.