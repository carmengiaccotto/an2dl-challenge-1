# **How to start**
* in all notebooks you must edit in the last cell the variable `INFERENCE_RESULTS_CSV_NAME` with the new version number

### **If you are in a local environment**
If you want to use the GPU (NVIDIA only):
1. Make sure you have installed the NVIDIA drivers for your GPU
2. Go to pytorch website and follow the instructions to install the right version of pytorch with CUDA support: https://pytorch.org/get-started/locally/
3. delete from requirements.txt the line that says `torch` and `torchvision`

Then, follow these steps:
1. `pip install -r ./requirements.txt`
2. Go to kaggle website, click on profile settings and create an API token. This will download a `kaggle.json` file.
3. Place the `kaggle.json` file in the user home directory under the folder `.kaggle/`. For example, on Linux it would be `/home/username/.kaggle/kaggle.json`.
4. Go in the first cell of the notebook and edit the `current_dir` variable to point to the directory where you want to download the data and work on.
   - If you have Google Drive installed and mounted, you can set it to a folder in your Google Drive.
5. Run the notebook

### **If you are in Google Colab**
1. **You still have to manually download the data from Kaggle and upload it to your Google Drive.**
2. the update `current_dir` variable in the first cell of the notebook to point to a folder in your Google Drive where you uploaded the data and want to work on.

### **If you are in Kaggle Notebooks**
Currently, it does not work because there are some import issues, need to nest them inside an if

1. You don't need to download the data, it is already available in the Kaggle environment.
2. just run the notebook as is.
---
# **Relevant Notes**

- After importing the data, all data gets converted to a less memory intensive data type where possible.
For example, integers that fit into int8 will be converted to that type.
- there are no missing values
- some features may be useless:
  - joint_13 to joint_25 are all pretty much all zeros
  - joint_30 is constant to 0.5

---
# **Create a new notebook for each Network**
To keep things organized, create a new notebook for each experiment you want to run.
You can copy the content of the main notebook and modify it as needed.
Make sure to change the name of the notebook to reflect the experiment you are running.