# Reservoir-Computing-Examination
A small project created with the intention of teaching myself the ins and outs of Reservoir Computing (RC). 
## What Is RC?
Reservoir computing is a machine learning approach that uses the properties of dynamical systems to process time-series data. A fixed, nonlinear 'reservoir', often a recurrent network of neurons, transforms input signals into higher-dimensional trajectories while retaining a fading memory of past inputs. A simple readout layer, often linear, then maps these reservoir states to outputs. Traditionally, this readout is the only part that is trained. As a result, training an RC model generally requires much less compute than training a recurrent network end-to-end with backpropagation through time.
## My Interest in RC
In general, I am interested in finding methods to perform machine learning that are more sustainable, less computationally (and economically) expensive, and less time-consuming; being able to experiment with new ideas cheaply is of obvious interest. For certain problems, RC could potentially provide an alternative that meets these criteria.
