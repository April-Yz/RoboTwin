# Architecture

The task layer builds named Cartesian actions; the IK layer returns six-joint positions and velocities; execution holds the endpoint for PD convergence; serialization stores schema, version, action names, targets, and paths; collection validates and replays in the same seeded scene before writing HDF5/MP4.
