
Vote prediction of US Senators from graph properties
=====
Original fork: [ntds_2018](https://github.com/AgentAGadge/ntds_2018)


[Mathias Gonçalves](https://github.com/magoncal)  
[Julien Heitmann](https://github.com/jheitmann)  
[Mathieu Lamiot](https://github.com/AgentAGadge)  
[Louis Landelle](https://github.com/louislandelle)  

This is the EE-558 final project repository. It comprises the data used for our analysis, obtained via the loading script [load_data.py](load_data.py), the file [common.py](common.py) that defines multiple constants and lambda functions used throughout the project, and finally a Jupyter Notebook, [vote_predictions.ipynb](vote_predictions.ipynb) that contains the data preparation, followed by the classification task that aims to predict, given a tuple (_senator\_id_, _bill\_id_), the senators vote position on this specific bill, based on a regression that makes use of multiple features that were constructed during the preparation step.

This project was made using Python 3.6.5

## Loading script

Since we do not have a predefined dataset, the data must be retrieved using the [ProPublica Congress API](https://projects.propublica.org/api-docs/congress-api/). This script should be executed from a terminal. It fetches certain features of interest independently, which speeds up the retrieval time if some data is already present, and allows to deal with the API quota (5000 requests per day per API Key) in a more efficient way. An API Key was provided here. 

Note that it is not required to run the loading script since the files containing the datasets are already saved in the repository.

### Usage of the loading script

When running [load_data.py](load_data.py), the following flags may be set to control the script behavior:

| Flag                 | Description                                          |
| -------------------- | ---------------------------------------------------- |
| active_senators      | Get list of active senators                          |
| adjacency            | Get vote data and create adjacency matrix            |
| cosponsorship        | Get list of cosponsored bills by senators            |
| cosponsors           | Get list of cosponsors of certain bills              |
| committees           | Get committees data and creates npy files            |
| requests             | Requests per senator, defaults to 1                  |

For example, this is how to generate the adjacency matrix from the milestones (based on senators vote positions that are retrieved with the API), and also fetch, for every senator in the current version of the file `data/active_senators.npy`, the list of cosponsored bills (the number of API requests will affect the amount of data retrieved to construct these data structures): 

>python load_data.py --adjacency --cosponsorship --requests=40

## Jupyter notebook

The project core. Pretty straightforward, first imports the data that was retrieved by [load_data.py](load_data.py), then computes senator- and bill-specific features that will be used to perform a regression. We compare two models, Logistic Regression and Random Forest. After feature engineering the data is split into **train** and **test**, and the models are trained on the former. Finally, the models performance is evaluated on the **test** set.

## Python version and library used:
* Python: 3.6.5
* Numpy: 1.15.4
* Pandas: 0.23.4
* Networkx: 2.2
* Scikit-learn: 0.20.0
* Scipy: 1.1.0
* Jsonschema: 2.6.0
* Pickleshare: 0.7.4
* Pyunlocbox: 0.5.2
* Matplotlib: 3.0.0
