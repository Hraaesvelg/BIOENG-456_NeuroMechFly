# BIOENG-456_NeuroMechFly

Even the most advanced robots are unable to match the highly dexterous and adaptive behaviors exhibited by animals. However, the mechanisms that enable animals to coordinate their movements so skillfully remain largely unclear. Studying these mechanisms could enhance our understanding of neurobiology and lead to advancements in artificial intelligence and robotics.

To investigate these principles, we will use the fruit fly Drosophila melanogaster as a model organism. This species is widely used in biological research and is highly genetically tractable, allowing researchers to study specific neurons repeatedly. Additionally, its relatively small nervous system (~100k neurons) may make it easier to model comprehensively.

Our project involves reverse-engineering the limb coordination circuits in the fruit fly's Ventral Nerve Cord (VNC) using a combination of Coupled oscillators and Reinforcement learning. The VNC is equivalent in function to the spinal cord in vertebrates and is responsible for coordinating limb movements and other behaviors. We aim to develop an artificial controller to control the walking of a virtual fly in a physics simulator (MuJoCo). This simulator uses an accurate body morphology of the fruit fly, reconstructed from a micro CT scan of a real animal, as described in Lobato-Rios et al, 2022.

## Method used 

Controlling the behavior of a simulated fly in the physics simulator serves as an intermediate step towards our ultimate goal of building a fly-inspired robot. We will explore three control approaches to mimic the neural computations in the VNC, namely, Coupled oscillators, Reinforcement learning, and a combination of both.
