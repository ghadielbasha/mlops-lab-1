# Lab 1 - Git, DVC and Data Preparation



## Question 1



### What files were created by `uv init` and what do they contain?



`uv init` created the initial Python project structure:



\- `src/` contains the Python source code.

\- `.python-version` specifies the Python version used by the project.

\- `pyproject.toml` contains project metadata, dependencies, and Python project configuration.

\- `README.md` contains project documentation.



Later, after adding Pillow with `uv add pillow`, `uv.lock` was also created to lock the exact package versions used by the project.



\---



## Question 2



### What files were created by DVC? What are they used for, and which ones should be pushed to Git?



Running `dvc init` created:



\- `.dvc/config` - contains repository-level DVC configuration such as the default DVC remote.

\- `.dvc/.gitignore` - prevents DVC internal cache files from being tracked by Git.

\- `.dvcignore` - allows files or folders to be excluded from DVC operations.



These configuration files should be committed to Git, as long as they do not contain secrets.



The actual DVC cache and dataset files should not be committed to Git.



\---



## Question 3



### Where are the credentials stored? What options exist other than `--global`? Should credentials be pushed to GitHub?



The DagsHub credentials were configured using `--global`, so they are stored in the user's global DVC configuration outside the Git repository.



Other configuration scopes include:



\- Repository configuration, by using the command without `--global`

\- `--local`, which stores configuration locally for the repository and is not meant to be committed

\- `--system`, which applies configuration system-wide



Credentials such as usernames, passwords, and tokens should never be pushed to GitHub.



Only non-secret configuration such as the remote URL and default remote should be committed.



\---



## Question 4



### What happened to `.gitignore` after running `dvc add data`?



DVC added:



```text

/data

```



to `.gitignore`.



This prevents Git from tracking and uploading the actual dataset.



The dataset is handled by DVC instead.



\---



## Question 5



### What does `data.dvc` contain?



`data.dvc` is a small DVC pointer/metadata file that represents the tracked `data` folder.



For the initial raw dataset it contained information such as:



\- the MD5 hash of the dataset

\- the total size

\- the number of files

\- the tracked path (`data`)



For the initial version:



\- Number of files: 16,643

\- Size: 1,188,442,712 bytes

\- Path: `data`



Git tracks `data.dvc`, while DVC tracks the actual dataset.



\---



## Question 6



### What can be seen on GitHub and DagsHub?



On GitHub:



\- The project code is present.

\- The DVC configuration files are present.

\- `data.dvc` is present.

\- The actual `data` folder is not stored in GitHub.



The `data.dvc` file acts as the pointer representing the dataset version.



On DagsHub:



\- The actual dataset was pushed using `dvc push`.

\- The raw and processed datasets are stored using the DVC remote.



Therefore:



\- Git/GitHub stores code and data pointer files.

\- DVC/DagsHub stores the actual data.



\---



## Question 7



### What happens when cloning the GitHub repository into a new folder?



After cloning the GitHub repository into a completely new temporary folder, there was no `data` folder.



The repository contained `data.dvc`, but not the actual dataset.



The following command was required to restore the data:



```bash

dvc pull

```



After running `dvc pull`, DVC downloaded the files from DagsHub and restored the `data` folder.



During the test:



\- 16,021 files were fetched.

\- 16,643 files were added to the workspace.



\---



## Question 8



### What happens when checking out an older Git and DVC version?



The commits that modified `data.dvc` were:



```text

82e5725 Add food11_processed and food11_processed_mini

646ccb4 Track data folder with dvc

```



The older version was checked out using:



```bash

git checkout 646ccb4

dvc checkout

```



After doing this, only:



```text

data/food11_raw

```



remained.



The newer folders:



```text

data/food11_processed

data/food11_processed_mini

```



were no longer present.



This happened because the older `data.dvc` file pointed to the older version of the dataset.



Then the latest version was restored using:



```bash

git checkout main

dvc checkout

```



After that, all three folders were restored:



```text

data/food11_raw

data/food11_processed

data/food11_processed_mini

```



This demonstrates that Git versions the DVC pointer file, while DVC restores the corresponding version of the actual data.



\---



## Data Preparation



The Food-11 raw dataset was organized as:



```text

data/food11_raw/training

data/food11_raw/evaluation

data/food11_raw/validation

```



The preprocessing script was created at:



```text

src/food11/data.py

```



It creates:



```text

data/food11_processed

data/food11_processed_mini

```



The processed images are resized to 128x128 and arranged into category folders.



The mini dataset contains at most 100 images per category and is used for faster development and testing.



The script is executed using:



```bash

uv run python ./src/food11/data.py

```



The updated datasets were then tracked and uploaded using:



```bash

dvc add data

git add data.dvc

git commit -m "Add food11_processed and food11_processed_mini"

git push

dvc push

```



## DVC Remote



DagsHub was used as the DVC remote.



GitHub stores the source code, configuration, and `data.dvc` pointer files.



DagsHub stores the actual dataset files managed by DVC.




