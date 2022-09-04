# Getting Started with the Sherpa.ai Federated Learning and Differential Privacy Framework

The Sherpa.ai Federated Learning and Differential Privacy Framework is a Python framework that provides an environment to 
develop research in the fields of private and distributed machine learning. The framework is designed with the goal of providing 
a set of tools that allow users to create and evaluate different aspects of these kinds of algorithms, with minimum code effort.

The main topics currently covered in the framework are federated learning and differential privacy. These techniques can 
be used together, in order to increase the privacy of a federated learning algorithm. 

We have also developed a set of notebooks that cover some of the most common use cases and explain the methodological 
aspects of the framework. The full list of notebooks is [here](https://github.com/sherpaai/Sherpa.ai-Federated-Learning-Framework/tree/master/notebooks)

Even if you are mainly interested in differential privacy, a good place to start is this [notebook](https://github.com/sherpaai/Sherpa.ai-Federated-Learning-Framework/blob/master/notebooks/federated_learning/federated_learning_basic_concepts.ipynb), 
where the main concepts of the Sherpa.ai Federated Learning and Differential Privacy Framework that are used in the tutorials 
and the documentation are explained.

The notebooks make the assumption that you are familiar with Python and some of the most popular libraries, like Numpy, 
Keras, and Tensorflow. The documentation is divided following the different packages in the framework. In every section 
there is a brief introduction of the module with the purpose and some illustrative examples. In many cases the documentation 
links with a notebook illustrating the use of the different modules and classes.


* The [private](../private/overview) package contains most of the core elements of the framework that are used in almost 
every line of code that you will write using the Sherpa.ai Federated Learning and Differential Privacy Framework.
* The [data_base](../databases) package introduces some datasets to work with.
* The [data_distribution](../data_distribution) package provides some modules to distribute data among nodes.
* The [federated_aggregator](../federated_aggregator) package has algorithms for aggregating models.
* The [federated_government](../federated_government) package defines the communication and the relationship between 
nodes.
* The [model](../model) package provides a set of common models that you might want to use.
* The [differential_privacy](../differential_privacy/overview) package introduces differential privacy algorithms to 
protect data privacy when data must be shared.