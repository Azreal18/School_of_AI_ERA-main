# TSAI-ERAv2-S23

# Open CLIP APP
## Objective
- Building a CLIP application on gradio/spaces using open-source models.


### Gradio App in HuggingFace Spaces
1. Created app.py that can read the model artifacts from HuggingFace Model Hub and launch the app
2. Pushed the app.py and requirements.txt to HuggingFace spaces using huggingface API from this [notebook](https://github.com/Azreal18/TSAI-ERAv2-S23/blob/main/notebook.ipynb)

## The HuggingFace Spaces Gradio App

The app is available [here](https://huggingface.co/spaces/Azreal18/Open_Clip)

![image](https://github.com/Azreal18/TSAI-ERAv2-S23/blob/main/imgs/spaces_snap.png)

- The app provides a set of images to search from. These images are sourced from scikit-learn image
- It takes free text as input for image search
- Displays the most suitable image from the given image as output
