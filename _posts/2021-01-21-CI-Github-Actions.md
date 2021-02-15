---
layout: post
title: "Creating a Continous Integration pipeline using Github Actions"
date: 2021-01-21 10:00:00
tag: Programming | DevOps
---


# 1) Introduction

Continous Integration (CI) is the practice of automation the integration of code changes into a single software project. This is a good practive because when the code changes, it builds the application and runs tests to see if the changes in the code doesn't break anything. This method of development is linked directly with Continous Deployment (CD) which is a an automation of deployment of your application 

Besides being a good thing to have all changes in the code being verified if it breaks something in the code, this is a key feature for agile development because the development process becomes more transparent, mainly if the team is using a Test-Driven Development approach.

There are various tools for CI, for instance Jenkins, CircleCI, github workflows, etc. In this article I will focus on building a CI tool for a python github repository. 


# 2) Creating the Github workflow

In order to create the githu workflow we need to create a YAML file into a folder on the path '.github/workflows', and below is an example of this file which builds and test a github repository on every push. If you want to read the YAML file without comments, check this [link](https://gist.github.com/nahumsa/feb22436d1beb82c7dbf22cbb63f14fe).

```yaml
# Define the name that you want your build to have, this is good 
# because you can read it on the workflow part of github
name: CI on [Python 3.6, 3.7, 3.8]

# Choose when you want to activate your workflow,
# it has a lot of options, for instance using the following strings or 
# array will trigger in the following command:
# - pull_request: will run the workflow anytime a pull request occurs;
# - push: will run the workflow anytime a push occurs on any branch;
# - realease: will run the workflow when you a release occurs;
#
# If you want to have various triggers you can use the list syntax:
# on: [push, pull_request]
on: push

# Jobs are what runs on the workflow, if you have more than one job
# they will run in parallel by default, but you can use a sequential 
# run using the needs keyphrase inside the job.
jobs: 
  
  # This is the name of the job that we are running
  build: 
    # We need to choose the type of the machine that we are running the job.
    # There are three types of runner types supported by github: Windows, 
    # Ubuntu, and macOS.
    runs-on: ubuntu-latest
    
    # strategy creates a matrix for your jobs that you can create different 
    # parameter for each run.
    strategy: 
      
      # using matrix you can create different job configurations using a key
      # valued pairs. 
      # Here we will change the python version on each run.
      matrix:
        python-version: [3.6, 3.7, 3.8]
    
    # This is what the job will in fact run so each step you can add a name
    # and environment variables, it is important to note that each step 
    # runs its own process in the runner environment and has access to the 
    # workspace and filesystem, because of that environment variables on each 
    # run will not be shared.
    steps:
      # Here we setup the action trigger on checkout
      - uses: actions/checkout@v2
      
      # Here we setup python for our build using the matrix values to
      # run in different python versions
      - name: Build using Python ${{ matrix.python-version }}
        uses: actions/setup-python@v2
        with:
          python-version: ${{ matrix.python-version }}
      
      # Here we install all dependencies using a requirements.txt file
      - name: Installing dependencies
        run: |
          python -m pip install pip --upgrade pip
          pip install -r requirements.txt
      
      # Here we change our folder using 'cd' and run all unittests on that folder
      - name: Running tests
        run: |
          cd tests/
          python -m unittest

```
You can learn more about the syntax on the [github docs](https://docs.github.com/en/actions/reference/workflow-syntax-for-github-actions).

# References
- [Trigger workflows documentation](https://docs.github.com/en/actions/reference/events-that-trigger-workflows)
- [Atlassian](https://www.atlassian.com/continuous-delivery/continuous-integration)