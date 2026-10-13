\# Lab 2 - Model Training and Experiment Tracking with MLflow



\## Question 1: What changed in pyproject.toml and uv.lock?



After running `uv add mlflow torch torchvision scikit-learn`, both files were updated.



\*\*pyproject.toml:\*\*

\- Added MLflow for experiment tracking.

\- Added PyTorch and torchvision for deep learning and image processing.

\- Added scikit-learn for machine learning evaluation.

\- Configured PyTorch to use CPU-only packages through the `pytorch-cpu` index.



\*\*uv.lock:\*\*

\- Recorded the exact versions of the installed libraries and their dependencies.

\- Ensures that the project environment can be reproduced consistently.



The installed versions included MLflow 3.17.1, PyTorch 2.14.1+cpu, torchvision 0.29.1+cpu, and scikit-learn 1.9.1.



\---



\## Question 2: What are backend-store-uri and default-artifact-root used for?



The MLflow tracking server was started using:



`uv run mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns`



\*\*backend-store-uri:\*\*



Specifies where MLflow stores experiment metadata. In our implementation, a local SQLite database named `mlflow.db` is used to store experiment information, run IDs, parameters, metrics, and run status.



\*\*default-artifact-root:\*\*



Specifies the default location where experiment artifacts are stored. We configured `./mlruns` as our local artifact storage location.



\*\*Difference:\*\*



Metadata describes the experiment, including its parameters and results.



Artifacts are actual files produced during training, such as saved machine learning models and environment configuration files.



\---



\## Question 3: Why should mlflow.db and mlruns/ not be tracked by Git or DVC?



The MLflow database and artifact directory are generated during experiment execution.



They should not be tracked by Git because:

\- They may contain large binary files.

\- They change frequently between training runs.

\- Git is mainly intended for versioning source code and configuration files.



They should not be tracked by DVC for this lab because:

\- They are local experiment tracking outputs, rather than the Food-11 datasets being versioned.

\- MLflow already manages the experiment metadata and artifacts.

\- Tracking these generated outputs again with DVC would introduce unnecessary duplication.



We added `mlflow.db\*` and `mlruns/` to `.gitignore`.



\---



\## Question 4: What happens when set\_experiment is called with a new name?



The following commands were used:



`mlflow.set\_tracking\_uri("http://127.0.0.1:5000")`



`mlflow.set\_experiment("food11")`



When `set\_experiment` is called with a name that does not already exist, MLflow automatically creates a new experiment.



During the first training execution, MLflow displayed a message indicating that the `food11` experiment did not exist and was being created.



The new experiment appeared in the MLflow web interface.



Subsequent training executions were recorded as separate runs under the same experiment.



\---



\## Question 5: What is the difference between log\_param and log\_metric?



\*\*mlflow.log\_param():\*\*



Records a configuration value that is fixed for a training run.



Examples:

\- Learning rate

\- Batch size

\- Number of epochs

\- Model architecture



\*\*mlflow.log\_metric():\*\*



Records numerical results produced during or after training.



Examples:

\- Training loss

\- Validation loss

\- Validation accuracy

\- Test accuracy



The `step` argument allows MLflow to record a metric at a particular training epoch or iteration.



For example, validation accuracy changes during training, so it is logged at each epoch using `step=epoch`.



Parameters do not require a step because they remain constant throughout the run.



\---



\## Question 6: Where are the parameters, metrics, and model artifact stored?



After successfully training ResNet18, we opened the run in the MLflow web interface.



The run displayed seven parameters, including:

\- Dataset: mini

\- Epochs: 5

\- Learning rate: 0.001

\- Batch size: 32

\- Architecture: resnet18

\- Pretrained: True

\- Trainable layers: fc



Five metrics were recorded:

\- train\_loss

\- val\_loss

\- val\_accuracy

\- test\_loss

\- test\_accuracy



Training and validation metrics were logged at each epoch and can be inspected in the MLflow interface.



The trained PyTorch model was logged using `mlflow.pytorch.log\_model()`.



The logged model contained:

\- `data/model.pth`

\- `pickle\_module\_info.txt`

\- `MLmodel`

\- `conda.yaml`

\- `python\_env.yaml`

\- `requirements.txt`



The model artifact is stored on the local machine in MLflow's artifact storage, configured through `./mlruns`, under the corresponding MLflow-managed model artifact directory.



The model was logged successfully but was not registered in the Model Registry, which was not required for this lab.



\---



\## Question 7: Which learning rate produced the best validation accuracy? Is higher always better?



We trained ResNet18 on the Food-11 mini dataset using different learning rates while keeping the number of epochs at 5.



| Learning Rate | Batch Size | Final Validation Accuracy |

|---|---|---|

| 0.001 | 32 | 58.12% |

| 0.01 | 32 | 58.30% |

| 0.0001 | 32 | 26.73% |

| 0.001 | 64 | 57.48% |



The learning rate \*\*0.01\*\*, with batch size 32, produced the highest final validation accuracy of \*\*58.30%\*\*.



However, higher learning rates are not always better.



During the experiment with learning rate 0.01, validation accuracy reached 60.04% at epoch 3 but decreased afterward.



This demonstrates that a larger learning rate can lead to faster learning but may also cause fluctuations during training.



The learning rate 0.0001 produced much lower accuracy after five epochs, indicating slower convergence under these training conditions.



\---



\## Question 8: What pattern is visible in the parallel coordinates plot?



The parallel coordinates plot was configured with:



\- Parameters: `lr`, `batch\_size`

\- Metric: `val\_accuracy`



The following patterns were observed:



1\. Learning rate 0.01 with batch size 32 achieved the highest final validation accuracy of 58.30%.

2\. Learning rate 0.001 with batch size 32 achieved 58.12%.

3\. Increasing batch size from 32 to 64 while keeping learning rate 0.001 slightly reduced validation accuracy from 58.12% to 57.48%.

4\. Learning rate 0.0001 with batch size 32 produced the lowest validation accuracy of 26.73%.



The results indicate that learning rate had a more noticeable effect on performance than the tested change in batch size.



However, these observations apply to the configurations and five training epochs tested in this lab.



\---



\## Question 9: Which run performed best? What is its run ID?



After sorting the MLflow runs table by `val\_accuracy` in descending order, the best completed run was identified.



\*\*Best Run Information:\*\*



\- Run name: adaptable-fly-126

\- Model: ResNet18

\- Dataset: Food-11 Mini

\- Epochs: 5

\- Learning rate: 0.01

\- Batch size: 32

\- Final validation accuracy: 58.30%

\- Test accuracy: 63.69%



\*\*Best Run ID:\*\*



`6abdccfd361f43709d1f8719cce0ae01`



This run ID will be used in the next MLOps lab.



\---



\## Conclusion



In Lab 2, we successfully implemented model training and experiment tracking using MLflow.



We configured a local MLflow tracking server with SQLite metadata storage and local artifact storage.



A pretrained ResNet18 model was modified to classify the 11 Food-11 categories.



We trained the final classification layer using the mini dataset, logged parameters and metrics, and saved the trained models in MLflow.



Four successful experiments were completed using different learning rates and batch sizes.



The experiments were compared using MLflow's comparison interface and parallel coordinates plot.



The best final validation accuracy was 58.30%, achieved using learning rate 0.01 and batch size 32.



The training script and dependency files were committed and pushed to the GitHub repository.


