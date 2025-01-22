[//]: # (Image References)


## links 

Github : https://github.com/carlosjimenez88M/Building-an-ML-Pipeline-for-Short-Term-Rental-Prices

WandB: https://wandb.ai/danieljimenez88m/nyc_airbnb?nw=nwuserdanieljimenez88m


---
[image1]: ./images/MLOps-process_fromNeptuneAI.PNG "ML workflow:"
[image2]: ./images/mlops_nyc_airbnb_projectstructure.PNG "Project structure:"
[image3]: ./images/nyc_airbnb_model-artifact_rf-regression_prod.PNG "End-to-end pipeline:"

# Build an ML Pipeline for Short-Term Rental Prices in NYC
## Business Use Case
We are working for a property management company renting rooms and properties for short periods of 
time on various rental platforms. We need to estimate the typical price for a given property based 
on the price of similar properties. Our company receives new data in bulk every week. The model needs 
to be retrained with the same cadence, necessitating an end-to-end pipeline that can be reused.

This task is part of the second project of the [MLOps](https://www.udacity.com/course/machine-learning-dev-ops-engineer-nanodegree--nd0821) Udacity course.

## Technical Information
Here are some general information given by the project settings.
- The experiment results and artifacts are created with _Python 3_, _Mlflow_ and _Hydra_ and stored on _Weights&Biases_ in the following [project](https://wandb.ai/ilo/nyc_airbnb/overview).<br>
There the best found model is tagged for production. Its hyperparameters are used as default values of the overall configuration file ``config.yaml`` which is stored in the root directory and read by the ``main.py`` file only. We will use Hydra to manage this configuration file via CLI.
- The basic dependencies given by the Udacity fork are not usable, because MLflow and Hydra need newer library versions to be usable. The associated files have been changed accordingly.
- We are using the former _pandas profiling_ library for exploratory data analysis (EDA) which is not up-to-date.
- We create an end-to-end pipeline with simple data preprocessing, doing some cleaning, and are using a random forest regression model with simple hyperparameter tuning. Both could be improved, getting better prediction results. For example, other models can be used for prediction, like<br>
-- Ridge regressor: Linear least squares with l2 regularization<br>
-- SVM regressor: Support Vector regression with RBF kernel<br>
-- LightGBM regressor: Constructing a gradient boosting model using tree-based learning algorithms.
- Additionally, production ready model resp. pipeline deployment is not implemented. In our case, getting a batch of newly collected data frequently every week, we need the model performing batch processing. For example, we can run its docker container to predict the new batch data.<br>
As mentioned in the [mlflow documentation](https://www.mlflow.org/docs/latest/models.html#built-in-deployment-tools) " MLflow can package models as self-contained Docker images with the REST API endpoint. The image can be used to safely deploy the model to various environments such as Kubernetes."<br>
So, regarding our XOps activities, we can containerize our model resp. project pipeline for an official CI/CD pipeline doing our code review, automated testing and pushing the container image to other repositories. In general, a container image is a kind of lightweight system image including our model, the needed coding and files regarding all dependencies (docker image, docker file as description how to build the docker image including all dependency information, docker compose file to run the docker container). To organise everything container orchestration can be done e.g. with Kubernetes. Infrastructure-as-code tools and frameworks help to create a deployment strategy.

On his [blog](https://neptune.ai/blog/mlops-architecture-guide) post Stephen Oladele showed a simple workflow diagram regarding the specific parts to do.

![ML workflow:][image1]

## Prerequisites

### Create environment
We expect you have at least Python 3.9 e.g. via conda installed and furthermore having cloned and stored this project repo locally. Then create a new environment using the ``environment.yml`` file provided in the root of your local repository and activate it via:

```bash
> conda env create -f environment.yml
> conda activate nyc_airbnb_dev
```

### Get API key for Weights and Biases
Let's make sure we are logged in to Weights & Biases. Get your API key from W&B by going to 
[https://wandb.ai/authorize](https://wandb.ai/authorize) and click on the + icon (copy to clipboard), 
then paste your key into this command:

```bash
> wandb login [your API key]
```

You should see a message similar to:
```
wandb: Appending key for api.wandb.ai to your netrc file: /home/[your username]/.netrc
```

## Project Structure
Main coding is stored in the ``src`` and in the ``components`` root subdirectories. To build the general project structure the cookie cutter template is used via CLI command _> cookiecutter cookie-mlflow-step -o src_, so, all modules include the file ``conda.yml`` to configure the conda environment and the file ``MLproject`` for the MLflow configuration of the specific module.<br>

Regarding the ML part, the modeling should be considered as baseline, because focus of the project are the MLOps process steps. In the Python coding of the main.py file such workflow steps are listed after the import block. So, the end-to-end pipeline is defined in this ``main.py`` file of the root directory.

```
_steps = [
    "download",
    "basic_cleaning",
    "data_check",
    "data_split",
    "train_random_forest",
    # NOTE: We do not include this in the steps so it is not run by mistake.
    # You first need to promote a model export to "prod" before you can run this,
    # then you need to run this step explicitly
    #"test_regression_model"
]
```

 All relevant steps are desribed now:

- **step 1** downloads a sample of the data. The pipeline stores the data in W&B as the artifact named ``sample.csv``.
- **step 2** is dealing with the exploratory data analysis (EDA). A simple cleaning happens: price outlier handling, last review date format, log scaling and drop out of invalid geolocation. Before and after such preprocessing a profiling report is created and stored in the reports subdirectory. Such kind of EDA is implemented as a jupyter notebook _EDA.ipynb_ and as Python script in ./src/basic_cleaning/run.py. Note: Imputation of missing values will be done in the inference pipeline for training and production.
- **step 3** includes some testing of the cleaned dataset content compared to a reference set, like e.g. dataset size or price range. The tests are implemented in ``src/data_check/test_data.py``.
- **step 4** handels the data splitting
- **step 5** has to deal with several substeps: the initial training of the random forest regression model, selecting best found model for production and its test against a test set.
 
The component ``train_val_test_split`` is used to extract and segregate the test set. We are uploading 2 new datasets, called ``trainval_data.csv`` and ``test_data.csv``. Then, the script ``src/train_random_forest/run.py`` includes the fit and prediction part together with the inference pipeline and plotting of feature importance. We run the entire pipeline varying the hyperparameters of the Random Forest Regression model by using Hydra's multi-run feature (adding the `-m` option at the end of the `hydra_options` specification). Example:<br>
```bash
> mlflow run . \
  -P steps=train_random_forest \
  -P hydra_options="modeling.random_forest.max_depth=10,50,100 modeling.random_forest.n_estimators=100,200,500 -m"
```
Then we select the best performing model for production, which is in W&B the one having the lowest Mean Absolute Error (MAE). In other words, the Mean Absolute Error is considered as our target metric. The 'lowest value' model is tagged with ``prod`` to mark it as "production ready".
- **step 6** finally tests the best performing RF regression model for production against a test set.

### Running the entire pipeline or just a selection of steps
In order to run the pipeline, you need to be in the root of the starter kit. The following command will run the entire pipeline:

```bash
>  mlflow run .
```
or by using the specific etl file on your local repository CLI

```bash
>  git pull origin master
>  mlflow run . -P hydra_options="etl.sample='sample2.csv'"
```

When developing it is useful to be able to run one step at the time. Say you want to run only
the ``download`` step. This step can be selected by using the `steps` parameter on the command line:

```bash
> mlflow run . -P steps=download
```
If you want to run the ``download`` and the ``basic_cleaning`` steps, you can similarly do:
```bash
> mlflow run . -P steps=download,basic_cleaning
```
You can override any other parameter in the configuration file using the Hydra syntax, by
providing it as a ``hydra_options`` parameter. For example, say that we want to set the parameter
modeling -> random_forest -> n_estimators to 10 and etl->min_price to 50:

```bash
> mlflow run . \
  -P steps=download,basic_cleaning \
  -P hydra_options="modeling.random_forest.n_estimators=10 etl.min_price=50"
```
The parameters that the steps require are delivered in their `MLproject` file.

### Structure details
So, finally the following structure exists:

![Project structure:][image2]

### Visualize the pipeline
In W&B in the Artifacts section, we get the following representation of our pipeline.

![End-to-end pipeline:][image3]

### Project releases
The initial release 1.0.0 provides a first basic model, but includes an issue regarding the geolocation of the airbnb. Its fix is part of the release 1.0.1. There boundaries have been taken into account.

## License
The original [License](LICENSE.txt) of the fork is delivered.
