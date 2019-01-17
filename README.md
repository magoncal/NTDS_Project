
United States senators vote prediction
=====
Original fork: [ntds_2018](https://github.com/AgentAGadge/ntds_2018)


[Mathias Gonçalves](https://github.com/magoncal)  
[Julien Heitmann](https://github.com/jheitmann)  
[Mathieu Lamiot](https://github.com/AgentAGadge)  
[Louis Landelle](https://github.com/louislandelle)  

This is the EE-558 final project repository. It comprises the data used for our analysis, obtained via the loading script [load_data.py](load_data.py), the file [common.py](common.py) that defines multiple constants and lambda functions used throughout the project, and finally a Jupyter Notebook, [vote_predictions.ipynb](vote_predictions.ipynb) that contains the data preparation, followed by the classification task that aims to predict, given a tuple (_senator\_id_, _bill\_id_), the senators vote position on this specific bill, based on a regression that makes use of multiple features that were constructed during the preparation step.

This project was made using Python 3.6.5

## Loading script

Brief description

## Jupyter notebook

Brief description

## Usage of the loading script

We use the [ProPublica Congress API](https://projects.propublica.org/api-docs/congress-api/) to retrieve our data. When running [load_data.py](load_data.py), the following flags may be set to control the application behavior:

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
