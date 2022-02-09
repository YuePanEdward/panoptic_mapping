# PanMap
This is a temporary repository used for my master thesis "Mapping for online path planning and 3D reconstruction". It is built on top of **[panoptic_mapping](https://github.com/ethz-asl/panoptic_mapping)** and **[voxblox](https://github.com/ethz-asl/voxblox)**. My improvement over voxblox on ESDF mapping can be found **[here](https://github.com/YuePanEdward/voxblox)** under the ``devel/voxfield`` branch.


# Panoptic Mapping

This package contains **panoptic_mapping**, a general framework for semantic volumetric mapping. We provide, among other, a submap-based approach that leverages panoptic scene understanding towards adaptive spatio-temporally consistent volumetric mapping, as well as regular, monolithic semantic mapping.

![combined](https://user-images.githubusercontent.com/36043993/135645102-e5798e36-e2b0-4611-9260-ec9d54d38e47.png)

Multi-resolution 3D Reconstruction, active and inactive panoptic submaps for temporal consistency, online change detection, and more.

# Table of Contents
**Credits**
* [Paper](#Paper)
* [Video](#Video)

**Setup**
* [Installation](#Installation)
* [Datasets](#Datasets)

**Examples**
* [Running the Panoptic Mapper](#running-the-panoptic-mapper)
* [Monolithic Semantic Mapping](#monolithic-semantic-mapping)
* [Running the RIO Dataset](#running-the-rio-dataset)

**Other**
* [Contributing](#Contributing)


# Paper
If you find this package useful for your research, please consider citing our paper:

* Lukas Schmid, Jeffrey Delmerico, Johannes Schönberger, Juan Nieto, Marc Pollefeys, Roland Siegwart, and Cesar Cadena. "**Panoptic Multi-TSDFs: a Flexible Representation for Online Multi-resolution Volumetric Mapping and Long-term Dynamic Scene Consistency**" arXiv preprint arXiv:2109.10165 (2021).
  \[[ArXiv](https://arxiv.org/abs/2109.10165)\]
  ```bibtex
  @ARTICLE{schmid2021panoptic,
    title={Panoptic Multi-TSDFs: a Flexible Representation for Online Multi-resolution Volumetric Mapping and Long-term Dynamic Scene Consistency},
    author={Schmid, Lukas and Delmerico, Jeffrey and Sch{\"o}nberger, Johannes and Nieto, Juan and Pollefeys, Marc and Siegwart, Roland and Cadena, Cesar},
    journal={arXiv preprint arXiv:2109.10165},
    year={2021}
  }
  ```
  
# Video
A short video overview explaining the approach will be released upon publication.

# Installation
Installation instructions for Linux. The repository was developed on Ubuntu 18.04 with ROS melodic and also tested on Ubuntu 20.04 with ROS noetic.

**Prerequisites**

1. If not already done so, install [ROS](http://wiki.ros.org/ROS/Installation) (Desktop-Full is recommended).

2. If not already done so, create a catkin workspace with [catkin tools](https://catkin-tools.readthedocs.io/en/latest/):
    ```shell script    
    # Create a new workspace
    sudo apt-get install python-catkin-tools
    mkdir -p ~/catkin_ws/src
    cd ~/catkin_ws
    catkin init
    catkin config --extend /opt/ros/$ROS_DISTRO
    catkin config --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo
    catkin config --merge-devel
    ```

**Installation**

1. Install system dependencies:
    ```shell script
    sudo apt-get install python-wstool python-catkin-tools
    ```

2. Move to your catkin workspace:
    ```shell script
    cd ~/catkin_ws/src
    ```

3. Download repo using [SSH](https://docs.github.com/en/github/authenticating-to-github/connecting-to-github-with-ssh):
    ```shell script
    git clone git@github.com:ethz-asl/panoptic_mapping.git
    ```

4. Download and install package dependencies using ros install:
    * If you created a new workspace.
    ```shell script
    wstool init . ./panoptic_mapping/panoptic_mapping.rosinstall
    wstool update
    ```

    * If you use an existing workspace. Notice that some dependencies require specific branches that will be checked out.
    ```shell script
    wstool merge -t . ./panoptic_mapping/panoptic_mapping.rosinstall
    wstool update
    ```

5. Compile and source:
    ```shell script
    catkin build panoptic_mapping_utils
    source ../devel/setup.bash
    ```
# Datasets
The datasets described in the paper and used for the demo can be downloaded from the [ASL Datasets](https://projects.asl.ethz.ch/datasets/doku.php?id=panoptic_mapping).

To a utility script is provided to directly download the data:
```
roscd panoptic_mapping_utils
export FLAT_DATA_DIR="/home/$USER/Documents"  # Or whichever path you prefer.
chmod +x panoptic_mapping_utils/scripts/download_flat_dataset.sh
./panoptic_mapping_utils/scripts/download_flat_dataset.sh
```
Additional data to run the mapper on the 3RScan dataset will follow.

# Examples
## Running the Panoptic Mapper
This example explains how to run the Panoptic Multi-TSDF mapper on the flat dataset. 

1. First, download the flat dataset:
    ```
    export FLAT_DATA_DIR="/home/$USER/Documents"  # Or whichever path you prefer.
    chmod +x panoptic_mapping_utils/scripts/download_flat_dataset.sh
    ./panoptic_mapping_utils/scripts/download_flat_dataset.sh
    ```
2. Replace the data `base_path` in `launch/run.launch (L10)` and `file_name` in `config/mapper/flat_groundtruth.yaml (L15)` to the downloaded path.
3. Run the mapper:
    ```
    roslaunch panoptic_mapping_ros run.launch
    ```
4. You should now see the map being incrementally built:

    <img src="https://user-images.githubusercontent.com/36043993/135860249-6334cc41-5758-457b-8f65-b017e2905804.png" width="400">
    
5. After the map finished building, you can save the map:
    ```
    rosservice call /panoptic_mapper/save_map "file_path: '/path/to/run1.panmap'" 
    ```
6. Terminate the mapper pressing Ctrl+C. You can continue the experiment on `run2` of the flat dataset by changing the `base_path`-ending in `launch/run.launch (L10)` to `run2`, and `load_map` and `load_path` in `launch/run.launch (L26-27)` to `true` and `/path/to/run1.panmap`, respectively. Optionally, you can also change the `color_mode` in `config/mapper/flat_groundtruth.yaml (L118)` to `change` to better highlight the change detection at work.
     ```
    roslaunch panoptic_mapping_ros run.launch
    ```
7. You should now see the map being updated based on the first run:

    <img src="https://user-images.githubusercontent.com/36043993/135861611-4d576750-3104-4d73-87dc-60b7a4ad1df6.png" width="400">

## Monolithic Semantic Mapping
This example will follow shortly.

## Running the RIO Dataset
This example will follow shortly.

# Contributing
**panoptic_mapping** is an open-source project, any contributions are welcome! 

For issues, bugs, or suggestions, please open a [GitHub Issue](https://github.com/ethz-asl/panoptic_mapping/issues).

To add to this repository:
* Please employ the [feature-branch workflow](https://www.atlassian.com/git/tutorials/comparing-workflows/feature-branch-workflow).
* Setup our auto-formatter for coherent style (we follow the [google style guide](https://google.github.io/styleguide/cppguide.html)):
    ```
    # Download the linter
    cd <linter_dest>
    git clone git@github.com:ethz-asl/linter.git
    cd linter
    echo ". $(realpath setup_linter.sh)" >> ~/.bashrc
    bash
    roscd panoptic_mapping/..
    init_linter_git_hooks
    # You're all set to go!
    ```
* Please open a [Pull Request](https://github.com/ethz-asl/panoptic_mapping/pulls) for your changes.
* Thank you for contributing!
