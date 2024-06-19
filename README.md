# GTCC_CVPR2024_NEU
Code for paper titled, "Learning to Predict Activity Progress by Self-Supervised Video Alignment" by Gerard Donahue and Ehsan Elhamifar, published at CVPR 2024.

# Paper Details
"Learning to Predict Activity Progress by Self-Supervised Video Alignment" offers a method to predict activity progress for both monotonic and in-the-wild activity-based procedural videos. The paper can be accessed (here.)[https://openaccess.thecvf.com/content/CVPR2024/html/Donahue_Learning_to_Predict_Activity_Progress_by_Self-Supervised_Video_Alignment_CVPR_2024_paper.html]

The primary tool we use to predict progress is video alignment (our proposed method for video alignment is called Generalized Temporal Cycle-Consistency, or GTCC), where progress prediction comes as an extremely useful byproduct of a multi-cycleback consistent embedding space amongst same-activity videos that are alignable. This github page is to increase the accessibility of our method for research or industrial purposes. 

To understand the technical mechanisms in GTCC, please read (our paper)[https://openaccess.thecvf.com/content/CVPR2024/html/Donahue_Learning_to_Predict_Activity_Progress_by_Self-Supervised_Video_Alignment_CVPR_2024_paper.html]. 

# Environment Setup
We use python version 3.8.10. Additionally, we use Ubuntu 18.04. Please set up a virtual environment with this python version and ideally the same linux version and run the following command:
- pip install -r ./requirements-py3.8.10.txt

This should set up your environment properly to run our open-source code. 

## environment variables
We use 3 environment variables for you to customize your data location, output path, and json data:
- export DATASET_PATH=#PATH TO YOUR DATASET FOLDER. THE FEATURES OR VIDEO FRAMES SHOULD BE STORE IN A SUBFOLDER (please check ./utils/train_util.py:get_data_subfolder_and_extension function)
- export OUTPUT_PATH='./output' # PATH TO WHERE YOU WOULD LIKE CODE TO OUTPUT FILES
- export JSON_DPATH='./dset_jsons' # PATH TO WHERE YOU STORE YOUR JSON FILES WITH DATASET METADATA.

Please put the above environment variable declarations in your ~/.bashrc file for easy loading of variables. 

# Dataset Organization
We understand that there are many different formats for dataset download. As such, we use a JSON file to save data filenames, as well as action sequences and temporal annotations for evaluation. We encourage you to explore the JSON file for the (egoprocel dataset)[https://sid2697.github.io/egoprocel/] that we included in ./dset_jsons. 

The json is a list of dictionaries, where each dictionary is the data-specific information for a task. It contains the keys of: "handles","hdl_actions", "hdl_start_times", and "hdl_end_times".
- "handles" is a list of the filename handles of the data files. 
- "hdl_actions" is a list of action sequences corresponding to the handles. 
- "hdl_start_times" and "hdl_end_times" are thee corresponding start and end frames of the actions in "hdl_actions" for each handle in "handles".

Implementors of our code are encouraged to use this method for dataloading which works out of the box, however if implementors would like to make their own dataloader, be sure to follow the pytorch dataloader format which is outlined in ./models/json_dataset.py:JSONDataset for the torch Dataset component of task videos. 

# Command line and arguments. 

## singletask setting
To do training for one task at a time for all tasks specified in the dataset json in ./dset_jsons, please run the following command:
- python singletask_train.py 1 --gtcc --ego --resnet
- python singletask_train.py 1 --vava --ego --resnet
- python singletask_train.py 1 --lav --ego --resnet
- python singletask_train.py 1 --tcc --ego --resnet

You can check ./utils/parser_util to explore all the args that we use for baseline and GTCC comparison. 

## cross-task setting
To use one model to align all tasks, please use the following commands:
- python multitask_train.py 1 --gtcc --ego --resnet --mcn
- python multitask_train.py 1 --vava --ego --resnet --mcn
- python multitask_train.py 1 --lav --ego --resnet --mcn
- python multitask_train.py 1 --tcc --ego --resnet --mcn

Adding the flag "--mcn" tells the code to use the multi-head crosstask network proposed in our paper.

## evaluation
After your code finishes executing, there will be an output folder in your $OUTPUT_PATH folder. To run evaluation on the last checkpoint in this folder, please run the following command (may differ depending on your output path):
- python eval.py -f output/single-task-setting/V1___GTCC_egoprocel
- python eval.py -f output/multi-task-setting/V1___GTCC_egoprocel

# Citation.
If you find this work useful in your project and you would like to cite your results, be sure to use the following citation for our paper:
```Citing
@InProceedings{Donahue_2024_CVPR,
    author    = {Donahue, Gerard and Elhamifar, Ehsan},
    title     = {Learning to Predict Activity Progress by Self-Supervised Video Alignment},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {18667-18677}
}
```

# Thank you!
If you have any questions, please feel free to email the first author, Gerard Donahue, at the following email:
- donahue [DOT] g [AT] northeastern [DOT] edu