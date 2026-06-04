#!/bin/bash

echo "Cloning DeepMETv2 (Teacher)..."
git clone https://github.com/DeepMETv2/graph-met.git DeepMETv2
cd DeepMETv2
git checkout 72dd1015720edba06f8e3dd36a0de6b48c23f497
cd ..

echo "Setup complete!"
