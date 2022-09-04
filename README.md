# Sherpa.ai Federated Learning and Differential Privacy Framework

The [Sherpa.ai](http://sherpa.ai) Federated Learning and Differential Privacy Framework has been developed to facilitate 
open research in the ﬁeld, with the objective of building models that learn from decentralized data, preserving data privacy. 
It is an open-source platform and aims to support 100 percent of the AI algorithms used in industry.

The Sherpa.ai Federated Learning and Differential Privacy Framework is an open-source framework for Machine Learning that 
allows collaborative learning to take place, without sharing private data. It has been developed to facilitate open research 
and experimentation in Federated Learning and Differential Privacy. Federated learning is a machine learning paradigm aimed 
at learning models from decentralized data, such as data located on users’ smartphones, in hospitals, or banks, and ensuring data privacy. 
This is achieved by training the model locally in each node (e.g., on each smartphone, at each hospital, or at each bank), 
sharing the model-updated parameters (not the data) and securely aggregating them to build a better global model. Federated 
Learning can be combined with Differential Privacy to ensure a higher degree of privacy. Differential Privacy is a statistical 
technique to provide data aggregations, while avoiding the leakage of individual data records. This technique ensures that 
malicious agents intervening in the communication of local parameters can not trace this information back to the data sources, 
adding an additional layer of data privacy. 

This technology could be disruptive in cases where it is compulsory to ensure data privacy, as in the following examples:

*    When data contains sensitive information, such as email accounts, personalized recommendations, and health information, 
applications should employ data privacy mechanisms to learn from a population of users whilst the sensitive data remains on each user’s device.

*    When data is located in data silos, an automotive parts manufacturer, for example, may be reluctant to disclose their data, 
but would benefit from models that learn from other manufacturers' data, in order to improve production and supply chain management.

*    Due to data-privacy legislation, banks and telecom companies, for example, cannot share individual records, but would 
benefit from models that learn from data across several entities.

Sherpa.ai is focused on democratizing Federated Learning by providing methodologies, pipelines, and evaluation techniques 
specifically designed for Federated Learning. The Sherpa.ai Federated Learning Platform enables developers to simulate Federated 
Learning scenarios with models, algorithms, and data provided by the framework, as well as their own data.

The Sherpa.ai Federated Learning and Differential Privacy Framework is a project by [Sherpa.ai](http://sherpa.ai), in collaboration 
with the [Andalusian Research Institute in Data Science and Computational Intelligence (DaSCI)](https://dasci.es/) research 
group from the [University of Granada](https://www.ugr.es/).

<p align="center"><img src="https://sherpa-cdn.s3-eu-west-1.amazonaws.com/forContest/Sherpa_ai_logo_nodegrade.png" alt="Sherpa AI" height = "120" /></p>
<p align="center"><img src="https://sherpa-cdn.s3-eu-west-1.amazonaws.com/third_parties_logos/DaSCI_logo_vertical.png"  alt="Andalusian Research Institute in Data Science and Computational Intelligence" height = "240"/> <img src="https://sherpa-cdn.s3-eu-west-1.amazonaws.com/third_parties_logos/ugr_logo.png"  alt="University of Granada" height = "240"/></p>

## Installation

See the [Installation](install.md) documentation for instructions on how to install the Sherpa.ai Federated Learning and 
Differential Privacy Framework.

## Getting Started

See the [Getting Started](https://sherpaai.github.io/Sherpa.ai-Federated-Learning-Framework/getting-started/) documentation 
for a brief introduction to using the Sherpa.ai Federated Learning and Differential Privacy Framework.

## Contributing

If you are interested in contributing to the Sherpa.ai Federated Learning and Differential Privacy Framework with tutorials, 
datasets, models, aggregation mechanisms, or any other code that others could benefit from, please be sure to review the 
[contributing guidelines](CONTRIBUTING.md).

## Issues

Use [GitHub issues](https://github.com/sherpaai/Sherpa.ai-Federated-Learning-Framework/issues) for tracking requests and bugs.

## Questions

Please direct questions to [Sherpa Developers Slack](https://sherpa-developers-invitation.herokuapp.com/).
