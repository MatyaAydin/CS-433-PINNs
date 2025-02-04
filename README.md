[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/UDdkOEMs)


# CS-433 Project 2

### Introduction
Our goal is to solve the Navier-Stokes equations using PINNS, a neural network architecture that takes the physics of the problem into account by forcing the output to satisfy the PDE with an additional term in its loss.
This project is supervised by professor Marco Picasso.



### Authors:
* Adam Mesbahi 387382

* Ibrahim Beniffou 370940

* Matya Aydin 388895


### Installations:

Please run the following command to install all necessary packages:
```bash
pip install -r pip_requirements.txt
```



### Repo structure:

* `train.ipynb`: Unsteady case.
* `kovasznay.ipynb`: Steady case.
* The folder utils contains helpers such as formatting data and visualizations but also our model and architectures as well as our losses.
* The folder gridsearch contains the results of hyperparameters selection for reproducibility.
* The folder saved_models contains models that were trained for reproducibility. For the steady case the name is as follows:
    ```
    model_{NAMEOFWEIGHTSMETHOD}.pth
    ```
    with `_MLP` if the model is a MLP of the class `PINNS_MLP_kovasznay` and nothing if it is the novel architecture. Examples of loading are provided in `kovasznay.ipynb`.
    Model1 and model2 are the model described in the report.
* The folder plot contains the plots in the report and other visualizations that helped us.
* The folder trainset contains the data generated using Fluent.
* The folder ball contains the scripts to generate the results in the appendix.
* The folders figures_model1 and figures_model2 contains the plot of the unsteady problem presented in the appendix
