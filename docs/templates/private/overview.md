# Private

This package contains most of the core elements of the framework that are used in almost every line of code that you will 
write using the Sherpa.ai Federated Learning and Differential Privacy Framework.

The most important element in the framework is the [DataNode](../data_node). A DataNode represents a 
device or element containing private data. In real world scenarios, this data is typically property of a user or company. 
This data is private and access must be defined, in order for it to be used. In this framework, it is defined using [DataAccessDefinition](../data/#dataaccessdefinition-class), 
a function that is applied to data before sharing private information from the node. There is a special class of access 
where there are no restrictions required to access the private data, [UnprotectedAccess](../data/#unprotectedaccess-class).