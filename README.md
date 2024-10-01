# Policy Graphs and Theory of Mind for Explainable Autonomous Driving

## Description
This thesis
explores the application of Policy Graphs (PGs), an innovative explainable AI (XAI) technique, in
autonomous driving. PGs represent an agent’s policy as a directed graph with natural language
descriptors, offering a human-readable explanation of the agent’s behaviour. This framework is
further enhanced by incorporating Theory of Mind concepts, enabling a deeper understanding of
these systems as if they possessed beliefs, desires, and intentions. This approach allows the graph
to capture what the agent does and what it desires and intends to do. This work makes 2 main contributions:
1. Development of an advanced framework that integrates the agent’s desires and intentions
into Policy Graphs, thus facilitating the extraction of motivations behind specific driving
decisions and identifying abnormal or undesirable behaviours.
2. An exploration of how external factors, such as weather and lighting conditions, influence
AV decision-making, uncovering potential harmful biases and patterns under various driving
scenarios.

## Folder Structure
- `example/`: This folder contains the source code of the project.
    - `dataset/`: Contains the scripts for generating the dataset
            from NuScenes raw data.
        - `data/`: Contains the generated Policy Graphs and the NuScenes dataset.
- `pgeon/`: This folder contains a tailored version of the **_pgeon_** Python package, which provides explanations for opaque agents using Policy Graphs.
- `generate_pg.py`: Script for generating a Policy Graph from an agent.
- `pg_entropy.py`, `pg_likelihood.py`: Script for computing static metrics on the Policy Graph.

## Installation

1. **Clone This Repository**

    ```bash
    git clone https://github.com/SaraMo14/nuscenes-pg.git

    cd nuscenes-pg
    ```
    
2. **Install Dependencies**

    ```bash
    pip install -r PATH_TO_REQUIREMENTS
    ```

3. **Install NuScenes Dataset**

    Download and uncompress the nuScenes dataset. For the mini version:

    ```bash
    cd nuscenes-pg/example/nuscenes/dataset

    mkdir -p /data/sets/nuscenes

    wget https://www.nuscenes.org/data/v1.0-mini.tgz

    tar -xf v1.0-mini.tgz -C ./data/sets/nuscenes
    ```
    You should have the following folder structure:
    ```bash
    /data/sets/nuscenes
          samples: Sensor data (not used in this work)
          sweeps: Sensor data (not used in this work)
          maps: Map files
          v1.0-mini: JSON tables that include all the meta data and annotations.

    ```
    Adjust the paths according to the dataset version chosen (i.e. mini, trainval). 
    Folders `samples` and `example` are not necessary. 
    
 

## Policy Graph Generation
Generate a probabilistic graphic represention of the behaviour of the vehicle by:

1. Process raw data from nuScenes to extract and compute the necessary data to build the policy graph (e.g. velocity, location, time of the day using the scripts in the `./dataset` folder.

2. Generate a Policy Graph considering different options for the scenes (city, weather, time of day) and proposed discretisation approaches:

   ```bash
   python3 generate_pg.py --input <input_data_folder> --file <input_data_file> --city <city_option> --weather <weather_option> --tod <time_of_day> --discretizer <discretization_option> --output <output_format> --verbose
   ```
   Most relevant arguments are;

   - `--city`: City filter for the PG (choices: all, b, s1, s2, s3, default: all).
   - `--weather`: Weather filter for scenes (choices: all, rain, no_rain, default: all).
   - `--tod`: Time of day filter (choices: all, day, night, default: all).
   - `--discretizer`: Discretizer option for the input data (choices: 0a, 0b, 1a, 1b, 2a, 2b, default: 0a).

   An example of the resulting Policy Graph is stored in `example/data/policy_graphs`.



## Register Vehicle's Desires
   After generating the Policy Graph, formalise the set of desires that hypothetically guide the vehicle’s behaviour using `pgeon/desire.py`.

For example, the desire "Lane Keeping" can be defined as:

```bash
lane_keeping = Desire(
    'Lane Keeping',        # Desire name
    [4, 5, 6],                # List of desirable actions (Gas, Brake, Go Straigh)
    {   # Dictionary with conditions over the desirable state
        LanePosition: [LanePosition.ALIGNED],               # The vehicle should be aligned in its lane
        Rotation: [Rotation.FORWARD],                       # The vehicle should be heading forward
        NextIntersection: [NextIntersection.NONE, 
                           NextIntersection.STRAIGHT],      # The vehicle will not turn at the next intersection
        Velocity: [Velocity.HIGH, Velocity.LOW, 
                   Velocity.MEDIUM, Velocity.MOVING]        # The vehicle is moving 
    }
)
```

## Compute Desires and Intention Metrics
Compute desire and intention metrics for the hypothesised desires using `pgeon/intention_introspector.py`. For example, for the "Lane Keeping" desire:
```bash
ii = IntentionIntrospector(desires = [lane_keeping], pg)

# Retrieve the desires data using the introspector
desires_data = ii.find_desires()

# Retrieve the intentions data based on a given commitment threshold
intentions_data = {
    desire: ii.find_intentions(commitment_threshold)[self.get_intention_metrics(commitment_threshold, desire)]
    for desire in self.desires
}
```
