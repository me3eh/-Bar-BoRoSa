# -Bar-BoRoSa
Project for AI for software Engineering

## Setting Env
If you are using Conda : 
```bash
conda env create -f conda.yaml
conda activate news_prediction 
```
Also you have to make your own `.env` file according to the `.env.example` file.



## Datasets
- [Fake News Detection](https://www.kaggle.com/vishakhdapat/fake-news-detection)
- [WELFake Fake News Detection Dataset](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification)
- TODO There was another one

## Project Structure
- `/model/data/` - Contains the datasets
- `/model/notebooks/` - Contains the notebooks
- `/model/notebooks/transformers` - Contains the notebooks for the transformers
- `/report_directory` - Contains our report 

## Chrome plugin
You need to install Google Chrome to get chrome_plugin to run. You will need to activate Developer mode, click Load Unpacked and point to folder with project of chrome_plugin.
Link to whole tutorial: https://medium.com/@meet30997/getting-started-with-chrome-extension-in-2023-how-to-create-your-own-chrome-extension-f5716770e8bb

## Python Flask Server
Best way to start the server is to invoke a command (from main directory)
```sh
cd server
flask run -p 5001
```
Additionally you will need to have mlflow installed 
It will run mlflow 

## Python dependency
When working with that project, you might like to create virtual environment.
It's mainly used to separate libraries between different projects.
To do it (1 time action), 
```sh
python -m venv venv
```
After that, every time from console, do:
```sh
source venv/bin/activate
```
That way you'll be in virtual environment of python

To get all needed libraries, to the following:
```sh
pip install -r requirements.txt
```
If you add any new library, delete requirements.txt and do:
```sh
pip freeze > requirements.txt
```
That way all needed libraries will be automatically added.

## Link to repository with site for articles
https://github.com/me3eh/BarBoRoSa_articles