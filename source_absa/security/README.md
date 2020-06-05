# Security
Encryption security features for DeepOpinion

| Project Maintainer | Proxy|
| ------------- | ------------- |
| Paul Opitz | Jamie Fox   |

![Build and Deploy Security](https://github.com/deepopinion/security/workflows/Build%20and%20Deploy%20Security/badge.svg)

## Installation

* Clone the repo

    ``git@github.com:deepopinion/security.git``
  
* Ensure that you have the environment variable `DO_UUID` available.  
    
* Nagivate to the security directory and install and activate virtual environment

    ``python3 -m venv env``
    
    ``source env/bin/activate``

* Install module locally 

    ``pip install -e . ``
    
* Build module locally

    ``sh devops/build``    

* Unzip to analyse content (optional)

    ``unzip dist/security-0.1.2-cp37-cp37m-linux_x86_64.whl -d dist/security``        
    
# Deployment

The package is deployed automatically at each commit at master on Gemfury (if all tests pass)
You can deploy it manually with:

    ``sh devops/deploy.sh``    
    


## Key Generation
Execute the following command and follow the instructions


    python create_product_key.py

To create a token, a valid `DO_KEY` has to be available as environment variable.
Please use a minimal functionality set while creating the key.
Never add the DeepOpinion functionality to a product key which is handed to a customer.

Currently, the following functionalities are available:

 * Training/Aspect-Sentiments or Training/*
 * Analysis/Aspect-Sentiments or Analyis/*
 * Models/*
 * DeepOpinion/* 
 
