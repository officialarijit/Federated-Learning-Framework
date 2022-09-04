# Bug Reporting, Feature Requests, and Pull Requests

We are happy to accept contributions in different ways.

## Bug Reporting
If you find a bug or come across unexpected behavior in the framework, please follow these steps to report it:

1. Check your code version. Maybe the problem is already fixed in a new version.

2. Search in the opened issues to avoid duplicity.

3. If the bug is not covered yet, please provide as much information as possible about your environment and, if possible, 
provide code to reproduce the behavior.

4. If you are able to solve the problem, please propose a solution in a pull request.

## Feature Requests

If you are interested in a new feature that is not developed at the moment, you can use the issue tracker to request it. 
Just be sure that you explain the new behaviour that you would like, as clearly as possible. It is also hepful to provide 
pseudocode or a schema to clarify the new feature being requested.

## Pull Requests

If you are going to add code that modifies the software architecture or changes the behavior of a functionality, we 
recommend that you describe the proposed changes, to save time. If you are just fixing an obvious bug, it is not necessary.


### Developing Process

The following are the basic conventions that we are using in this project.

#### Github Workflow

The main strategy that we are using is the "Feature Branch Workflow", which is described very well [here](https://www.atlassian.com/git/tutorials/comparing-workflows/feature-branch-workflow) 
(in this case, it is for Bitbucket, but the same can be applied to Github).

#### Code Style

We are using the [Google Python Style Guide](http://google.github.io/styleguide/pyguide.html)

#### Code Tests

All tests have to pass with 100% line coverage. You only need to execute the following command in the base project directory:
`pytest --cov=shfl test/`
