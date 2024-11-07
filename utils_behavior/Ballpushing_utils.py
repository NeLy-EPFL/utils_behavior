import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import itertools
from operator import itemgetter

import holoviews as hv
from bokeh.models import HoverTool
from bokeh.plotting import show
from bokeh.models import TapTool
from bokeh.models import CustomJS
from bokeh.io import show, curdoc
from bokeh.plotting import figure
import webbrowser

from holoviews import streams

from scipy.ndimage import median_filter, gaussian_filter
from pathlib import Path
import sys
import traceback
import json
import datetime
import subprocess
from collections import Counter
import pickle
import os
import platform

import cv2
from moviepy.editor import VideoClip
from moviepy.editor import VideoFileClip
from moviepy.video.fx import all as vfx
import pygame

import warnings

from .Utils import *
from .Processing import *
from .Sleap_utils import *

from .HoloviewsTemplates import hv_main

sys.modules["Ballpushing_utils"] = sys.modules[
    __name__
]  # This line creates an alias for utils_behavior.Ballpushing_utils to utils_behavior.__init__ so that the previously made pkl files can be loaded.

brain_regions_path = "/mnt/upramdya_data/MD/Region_map_240122.csv"


def save_object(obj, filename):
    """Save a custom object as a pickle file.

    Args:
        obj (object): The object to be saved.
        filename (Pathlib path): the path where to save the object. No need to add the .pkl extension.
    """

    # If the filename does not end with .pkl, add it
    if not filename.endswith(".pkl"):
        filename = filename.with_suffix(".pkl")

    with open(filename, "wb") as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def load_object(filename):
    """Load a custom object from a pickle file.

    Args:
        filename (Pathlib path): the path to the object. No need to add the .pkl extension.
    """

    # If the filename does not end with .pkl, add it
    if not filename.endswith(".pkl"):
        filename = filename.with_suffix(".pkl")

    with open(filename, "rb") as input:
        obj = pickle.load(input)

    return obj


def find_interaction_events(
        protagonist1,
        protagonist2,
        nodes1=["Lfront", "Rfront"],
        nodes2=["x_centre", "y_centre"],
        threshold=[0,11],
        gap_between_events=4,
        event_min_length=2,
        fps=29
    ):
    """
    This function finds interaction events where the specified nodes of two protagonists are within a certain distance for a minimum amount of time.

    Parameters:
    protagonist1 (DataFrame): DataFrame containing the first protagonist's tracking data.
    protagonist2 (DataFrame): DataFrame containing the second protagonist's tracking data.
    nodes1 (list): List of nodes for the first protagonist to check the distance with the second protagonist (e.g., ["Lfront", "Rfront"]).
    nodes2 (list): List of nodes for the second protagonist to check the distance with the first protagonist (e.g., ["Centre",]).
    threshold (int): The distance threshold (in pixels) for the nodes to be considered in interaction.
    gap_between_events (int): The minimum gap required between two events, expressed in seconds.
    event_min_length (int): The minimum length of an event, expressed in seconds.
    fps (int): Frames per second of the video.

    Returns:
    list: A list of interaction events, where each event is represented as [start_frame, end_frame, duration].
    """

    # Convert the gap between events and the minimum event length from seconds to frames
    gap_between_events = gap_between_events * fps
    event_min_length = event_min_length * fps

    # Initialize a list to store distances for all specified nodes
    distances = []

    # Compute the Euclidean distance for each specified node
    for node1 in nodes1:
        for node2 in nodes2:
            
            # x1 = protagonist1[f"x_{node1}"]
            # y1 = protagonist1[f"y_{node1}"]
            # x2 = protagonist2[f"x_{node2}"]
            y2 = protagonist2[f"y_{node2}"]

            # # Check for NaN values
            # if x1.isna().any() or y1.isna().any() or x2.isna().any() or y2.isna().any():
            #     print(f"NaN values found in data for nodes {node1} and {node2}")
            #     print(f"x1: {x1}")
            #     print(f"y1: {y1}")
            #     print(f"x2: {x2}")
            #     print(f"y2: {y2}")

            # # Print intermediate values
            # print(f"x1: {x1}")
            # print(f"y1: {y1}")
            # print(f"x2: {x2}")
            print(f"y2: {y2}")
            
            distances_node = np.sqrt((protagonist1[f"x_{node1}"] - protagonist2[f"x_{node2}"])**2 + (protagonist1[f"y_{node1}"] - protagonist2[f"y_{node2}"])**2)
            #print(f"Distances for {node1} and {node2}: {distances_node}")
            distances.append(distances_node)

    # Combine distances to find frames where any node is within the threshold distance
    combined_distances = np.min(distances, axis=0)
    interaction_frames = np.where(
                (np.array(combined_distances) > threshold[0]) & (np.array(combined_distances) < threshold[1])
            )[0]
    
    # Debug: Print the combined distances and interaction frames
    # print(f"Combined distances: {combined_distances}")
    # print(f"Interaction frames: {interaction_frames}")

    # If no interaction frames are found, return an empty list
    if len(interaction_frames) == 0:
        return []

    # Find the distance between consecutive interaction frames
    distance_betw_frames = np.diff(interaction_frames)

    # Find the points where the distance between frames is greater than the gap between events
    split_points = np.where(distance_betw_frames > gap_between_events)[0]

    # Add the first and last points to the split points
    split_points = np.insert(split_points, 0, -1)
    split_points = np.append(split_points, len(interaction_frames) - 1)

    # Initialize the list of interaction events
    interaction_events = []

    # Iterate over the split points to find events
    for f in range(0, len(split_points) - 1):
        # Define the start and end of the region of interest (ROI)
        start_roi = interaction_frames[split_points[f] + 1]
        end_roi = interaction_frames[split_points[f + 1]]

        # Calculate the duration of the event
        duration = end_roi - start_roi

        # If the duration of the event is greater than the minimum length, add the event to the list
        if duration > event_min_length:
            interaction_events.append([start_roi, end_roi, duration])

    return interaction_events

def filter_experiments(source, **criteria):
    """Generates a list of Experiment objects based on criteria.

    Args:
        source (list): A list of flies, experiments or folders to create Experiment objects from.
        criteria (dict): A dictionary of criteria to filter the experiments.

    Returns:
        list: A list of Experiment objects that match the criteria.
    """

    flies = []

    # If the source is a list of flies, check directly for the criteria in flies
    if isinstance(source[0], Fly):
        for fly in source:
            if all(
                fly.arena_metadata.get(key) == value for key, value in criteria.items()
            ):
                flies.append(fly)

    else:
        if isinstance(source[0], Experiment):
            Exps = source

        else:
            # Create Experiment objects from the folders
            Exps = [Experiment(f) for f in source]

        # Get a list of flies based on the criteria
        for exp in Exps:
            for fly in exp.flies:
                if all(
                    fly.arena_metadata.get(key) == value
                    for key, value in criteria.items()
                ):
                    flies.append(fly)

    return flies


# Pixel size: 30 mm = 500 pixels, 4 mm = 70 pixels, 1.5 mm = 25 pixels


# TODO : Test the valid_data function in conditions where I know the fly is dead or the arena is empty or not to check success
class Fly:
    """
    A class for a single fly. This represents a folder containing a video, associated tracking files, and metadata files. It is usually contained in an Experiment object, and inherits the Experiment object's metadata.
    """

    def __init__(
        self,
        directory,
        experiment=None,
        as_individual=False,
    ):
        """
        Initialize a Fly object.

        Args:
            directory (Path): The path to the fly directory.
            experiment (Experiment, optional): An optional Experiment object. If not provided, an Experiment object will be created based on the parent directory of the given directory.

        Attributes:
            directory (Path): The path to the fly directory.
            experiment (Experiment): The Experiment object associated with the Fly.
            arena (str): The name of the parent directory of the fly directory.
            corridor (str): The name of the fly directory.
            name (str): A string combining the name of the experiment directory, the arena, and the corridor.
            arena_metadata (dict): Metadata for the arena, obtained by calling the get_arena_metadata method.
            video (Path): The path to the video file in the fly directory.
            flytrack (Path): The path to the fly tracking file in the fly directory. If not found, a message is printed and this attribute is not set.
            balltrack (Path): The path to the ball tracking file in the fly directory. If not found, a message is printed and this attribute is not set.
            flyball_positions (DataFrame): The coordinates of the fly and the ball, obtained by calling the get_coordinates function with the flytrack and balltrack paths.
        """

        self.directory = Path(directory)
        
        if experiment is not None:
            self.experiment = experiment
        elif as_individual:
            self.experiment = Experiment(self.directory.parent.parent, metadata_only=True)
        else:
            self.experiment = Experiment(self.directory.parent.parent)                
        
        self.arena = self.directory.parent.name
        self.corridor = self.directory.name
        
        self.name = f"{self.experiment.directory.name}_{self.arena}_{self.corridor}"
        self.arena_metadata = self.get_arena_metadata()
        # For each value in the arena metadata, add it as an attribute of the fly
        for var, data in self.arena_metadata.items():
            setattr(self, var, data)

        self.flyball_positions = None
        self.fly_skeleton = None

        # Get the brain regions table
        brain_regions = pd.read_csv(brain_regions_path, index_col=0)

        # If the fly's genotype is defined in the arena metadata, find the associated nickname and brain region from the brain_regions_path file
        if "Genotype" in self.arena_metadata:
            try:
                genotype = self.arena_metadata["Genotype"]

                # If the genotype is None, skip the fly
                if genotype.lower() == "none":
                    print(f"Genotype is None: {self.name} is empty.")
                    self.valid_data = True
                    return

                # Convert to lowercase for comparison
                lowercase_index = brain_regions.index.str.lower()
                matched_index = lowercase_index.get_loc(genotype.lower())

                self.nickname = brain_regions.iloc[matched_index]["Nickname"]
                self.brain_region = brain_regions.iloc[matched_index][
                    "Simplified region"
                ]
            except KeyError:
                print(
                    f"Genotype {genotype} not found in brain regions table for {self.name}. Defaulting to PR"
                )
                self.nickname = "PR"
                self.brain_region = "Control"

        # Load the video file
        self.video = self.load_video()

        # Load the tracking files
        try:
            self.balltrack = self.load_tracking_file("*ball*.h5", "ball")
            self.flytrack = self.load_tracking_file("*fly*.h5", "fly")
            self.skeletontrack = self.load_tracking_file("*full_body*.h5", "fly")

            # Check if the balltrack file exists and either flytrack or skeletontrack exists
            if self.balltrack is None or (self.flytrack is None and self.skeletontrack is None):
                print(f"Missing required tracking files for {self.name}.")
                self.flytrack = None
                self.balltrack = None
                self.skeletontrack = None
                return

        except FileNotFoundError as e:
            print(f"Error loading tracking files for {self.name}: {e}")
            self.flytrack = None
            self.balltrack = None
            self.skeletontrack = None
            return
        
        self.interaction_events = self.find_flyball_interactions()

        # Check if the fly is alive or dead and if it interacts with the ball
        self.valid_data = self.check_data_quality()
        
        if not self.valid_data:
            print(f"Invalid data for: {self.name}. Skipping.")
            return

        self.start_x, self.start_y = self.get_initial_position()
        
        self.metrics = {}
        
        self.compute_metrics()

        # Compute the skeleton tracking data
        self.fly_skeleton = self.get_skeleton()
        
        self.exit_time = self.get_exit_time()
        
    def __str__(self):
        # Get the genotype from the metadata
        genotype = self.arena_metadata["Genotype"]

        return f"Fly: {self.name}\nArena: {self.arena}\nCorridor: {self.corridor}\nVideo: {self.video}\nFlytrack: {self.flytrack}\nBalltrack: {self.balltrack}\nGenotype: {genotype}"

    def __repr__(self):
        return f"Fly({self.directory})"

    def load_video(self):
        """Load the video file for the fly."""
        try:
            return list(self.directory.glob(f"{self.corridor}.mp4"))[0]
        except IndexError:
            try:
                return list(
                    self.directory.glob(
                        f"{self.directory.parent.name}_corridor_{self.corridor[-1]}.mp4"
                    )
                )[0]
            except IndexError:
                try:
                    # Look for a video file in the corridor directory
                    return list(self.directory.glob("*.mp4"))[0]
                except IndexError:
                    raise FileNotFoundError(f"No video found for {self.name}.")

    def load_tracking_file(self, pattern, object_type):
        """Load a tracking file for the fly."""
        try:
            tracking_file = list(self.directory.glob(pattern))[0]
            return Sleap_Tracks(tracking_file, object_type=object_type, debug=False)
        except IndexError:
            return None
    
    def get_initial_position(self):
        """
        Get the initial x and y positions of the fly. First, try to use the fly tracking data.
        If not available, use the skeleton data.

        Returns:
            tuple: The initial x and y positions of the fly.
        """
        # Check if fly tracking data is available
        if hasattr(self, 'flytrack') and self.flytrack is not None:
            fly_data = self.flytrack.objects[0].dataset
            if 'y_thorax' in fly_data.columns and 'x_thorax' in fly_data.columns:
                return fly_data['x_thorax'].iloc[0], fly_data['y_thorax'].iloc[0]
            elif 'y_thorax' in fly_data.columns and 'x_thorax' in fly_data.columns:
                return fly_data['x_thorax'].iloc[0], fly_data['y_thorax'].iloc[0]

        # Fallback to skeleton data if fly tracking data is not available
        if self.fly_skeleton is not None:
            if 'y_thorax' in self.fly_skeleton.columns and 'x_thorax' in self.fly_skeleton.columns:
                return self.fly_skeleton['x_thorax'].iloc[0], self.fly_skeleton['y_thorax'].iloc[0]
            elif 'y_thorax' in self.fly_skeleton.columns and 'x_thorax' in self.fly_skeleton.columns:
                return self.fly_skeleton['x_thorax'].iloc[0], self.fly_skeleton['y_thorax'].iloc[0]

        raise ValueError(f"No valid position data found for {self.name}.")
    
    def get_exit_time(self):
        """
        Get the exit time, which is the first time at which the fly x position has been 100 px away from the initial fly x position.

        Returns:
            float: The exit time, or None if the fly did not move 100 px away from the initial position.
        """

        if self.flytrack is None:
            return None

        fly_data = self.flytrack.objects[0].dataset

        # Get the initial x position of the fly
        initial_x = self.start_x

        # Get the x position of the fly
        x = fly_data["x_thorax"]

        # Find the first time at which the fly x position has been 100 px away from the initial fly x position
        exit_condition = x > initial_x + 100
        if not exit_condition.any():
            return None

        exit_time = x[exit_condition].index[0] / self.experiment.fps

        return exit_time


    # TODO : Also find flies that die during the recording
    def check_data_quality(self):
        """Check if the fly is dead or in poor condition.

        This method loads the smoothed fly tracking data and checks if the fly moved more than 30 pixels in the y or x direction. If it did, it means the fly is alive and in good condition.

        Returns:
            bool: True if the fly is dead or in poor condition, False otherwise.
        """
        # Ensure that flytrack is not None
        if self.flytrack is None:
            print(f"{self.name} has no tracking data.")
            return False

        # Use the flytrack dataset
        fly_data = self.flytrack.objects[0].dataset

        # Check if any of the smoothed fly x and y coordinates are more than 30 pixels away from their initial position
        moved_y = np.any(
            abs(
                fly_data["y_thorax"]
                - fly_data["y_thorax"].iloc[0]
            )
            > 30
        )
        moved_x = np.any(
            abs(
                fly_data["x_thorax"]
                - fly_data["x_thorax"].iloc[0]
            )
            > 30
        )

        if not moved_y and not moved_x:
            print(f"{self.name} did not move significantly.")
            return False

        # Check if the interaction events dictionary is empty
        if not self.interaction_events or not any(self.interaction_events.values()):
            print(f"{self.name} did not interact with the ball.")
            return False

        print(f"{self.name} is alive and interacted with the ball.")
        return True

    def get_arena_metadata(self):
        """
        Retrieve the metadata for the Fly object's arena.

        This method looks up the arena's metadata in the experiment's metadata dictionary.
        The arena's name is converted to lowercase and used as the key to find the corresponding metadata.

        Returns:
            dict: A dictionary containing the metadata for the arena. The keys are the metadata variable names and the values are the corresponding metadata values. If no metadata is found for the arena, an empty dictionary is returned.
        """
        # Get the metadata for this fly's arena
        arena_key = self.arena.lower()
        return {
            var: data[arena_key]
            for var, data in self.experiment.metadata.items()
            if arena_key in data
        }

    def display_metadata(self):
        """
        Print the metadata for the Fly object's arena.

        This method iterates over the arena's metadata dictionary and prints each key-value pair.

        Prints:
            str: The metadata variable name and its corresponding value, formatted as 'variable: value'.
        """
        # Print the metadata for this fly's arena
        for var, data in self.arena_metadata.items():
            print(f"{var}: {data}")

    def detect_boundaries(self, threshold1=30, threshold2=100):
        """Detects the start and end of the corridor in the video. This is later used to compute the relative distance of the fly from the start of the corridor.

        Args:
            threshold1 (int, optional): the first threshold for the hysteresis procedure in Canny edge detection. Defaults to 30.
            threshold2 (int, optional): the second threshold for the hysteresis procedure in Canny edge detection. Defaults to 100.

        Returns:
            frame (np.array): the last frame of the video.
            start (int): the start of the corridor.
            end (int): the end of the corridor.
        """

        video_file = self.video

        if not video_file.exists():
            print(f"Error: Video file {video_file} does not exist")
            return None, None, None

        # open the video
        cap = cv2.VideoCapture(str(video_file))

        # get the total number of frames
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # set the current position to the last frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)

        # read the last frame
        ret, frame = cap.read()
        cap.release()

        if not ret:
            print(f"Error: Could not read frame from video {video_file}")
            return None, None, None
        elif frame is None:
            print(f"Error: Frame is None for video {video_file}")
            return None, None, None

        # Convert to grayscale
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply a Gaussian blur
        frame = cv2.GaussianBlur(frame, (5, 5), 0)

        # Use the Canny edge detection method
        edges = cv2.Canny(frame, threshold1=threshold1, threshold2=threshold2)

        # Get the top and bottom edges of the corridor in y-direction
        top_edge = np.min(np.where(edges > 0)[0])
        bottom_edge = np.max(np.where(edges > 0)[0])

        start = bottom_edge - 110
        end = top_edge + 110

        # Save a .npy file with the start and end coordinates in the video folder
        np.save(video_file.parent / "coordinates.npy", [start, end])

        return frame, start, end

    def get_coordinates(self, ball=True, fly=True, xvals=True):
        """Extracts the coordinates from the ball and fly h5 sleap data.

        Parameters:
            ball (bool): Whether to extract the ball coordinates.
            fly (bool): Whether to extract the fly coordinates.
            xvals (bool): Whether to extract the x coordinates.

        Returns:
            list: A list of DataFrames, each containing the coordinates of the ball and fly.
        """
        data_dict = {}

        if ball:
            # Iterate over all ball tracks
            for ball_idx in range(1, len(self.balltrack.objects) + 1):
                #print(f"Processing ball index: {ball_idx}")
                data = []
                columns = []

                ball_data = self.balltrack.dataset[self.balltrack.dataset["object"] == f"ball_{ball_idx}"]

                xball = ball_data["x_centre"]
                yball = ball_data["y_centre"]

                try:
                    # Replace NaNs in yball
                    replace_nans_with_previous_value(yball)

                    # Replace NaNs in xball
                    replace_nans_with_previous_value(xball)
                except ValueError as e:
                    warnings.warn(
                        f"Skipping ball coordinates for {self.name} due to error: {e}"
                    )
                    continue

                data.append(yball)
                columns.append(f"yball")

                if xvals:
                    data.append(xball)
                    columns.append(f"xball")

                if fly:
                    flydata = self.flytrack.dataset[self.flytrack.dataset["object"] == "fly_1"]

                    xfly = flydata["x_thorax"]
                    yfly = flydata["y_thorax"]

                    try:
                        # Replace NaNs in yfly
                        replace_nans_with_previous_value(yfly)

                        # Replace NaNs in xfly
                        replace_nans_with_previous_value(xfly)
                    except ValueError as e:
                        warnings.warn(
                            f"Skipping fly coordinates for {self.name} due to error: {e}"
                        )
                        continue

                    data.append(yfly)
                    columns.append("yfly")

                    if xvals:
                        data.append(xfly)
                        columns.append("xfly")

                # Combine the x and y arrays into a single 2D array
                data = np.stack(data, axis=1)

                # Convert the 2D array into a DataFrame
                data = pd.DataFrame(data, columns=columns)

                data = data.assign(Frame=data.index + 1)

                data["Frame"] = data["Frame"].astype(int)

                data["time"] = data["Frame"] / self.experiment.fps
                
                data["ball_idx"] = ball_idx

                if self.start_y:
                    data[f"yfly_relative"] = abs(data["yfly"] - self.start_y)

                    # Fill missing values using linear interpolation
                    data[f"yfly_relative"] = data[f"yfly_relative"].interpolate(method="linear")

                    data[f"yball_relative"] = abs(data[f"yball"] - self.start_y)

                    data[f"yball_relative"] = data[f"yball_relative"].interpolate(method="linear")
                    
                # Determine if the ball is a training ball or a generalisation ball
                if xvals:
                    # Drop NaN values and check if there are any valid entries
                    xfly_non_nan = data["xfly"].dropna()
                    xball_non_nan = data["xball"].dropna()
                    
                    #print(f"Non-NaN xfly count: {len(xfly_non_nan)}, Non-NaN xball count: {len(xball_non_nan)}")
                    
                    if not xfly_non_nan.empty and not xball_non_nan.empty:
                        # Pick the first x fly and ball values that aren't NaN
                        x0_fly = xfly_non_nan.iloc[0]
                        x0_ball = xball_non_nan.iloc[0]
                        
                        initial_separation = abs(x0_ball - x0_fly)
                        #print(f"Initial separation: {initial_separation}")
                        
                        if initial_separation <= 100:
                            ball_type = "training"
                        else:
                            ball_type = f"generalisation"
                        
                        data["ball_type"] = ball_type
                        data_dict[ball_type] = data
                    else:
                        print(f"No valid tracking data for xfly or xball for {ball_idx}")

        #print(f"Final data_dict keys: {list(data_dict.keys())}")
        return data_dict
    
    def get_skeleton(self):
        """
        Extracts the coordinates of the fly's skeleton from the full body tracking data.

        Returns:
            DataFrame: A DataFrame containing the coordinates of the fly's skeleton.
        """

        if self.skeletontrack is None:
            warnings.warn(f"No skeleton tracking file found for {self.name}.")
            return None


        # Get the first track
        full_body_data = self.skeletontrack.objects[0].dataset
        
        # For each node, replace NaNs with the previous value
        # Get the columns containing the x and y coordinates
        x_columns = [col for col in full_body_data.columns if "x_" in col]
        y_columns = [col for col in full_body_data.columns if "y_" in col]
        
        for x_col, y_col in zip(x_columns, y_columns):
            
            x = full_body_data[x_col]
            y = full_body_data[y_col]
            
            if x.empty or y.empty:
                warnings.warn(f"Skipping skeleton coordinates for {self.name} and node: {x_col} and {y_col} due to empty data.")
                return None
            
            try:
                replace_nans_with_previous_value(x)
                replace_nans_with_previous_value(y)
            except ValueError as e:
                warnings.warn(
                    f"Skipping skeleton coordinates for {self.name} and node: {x_col} and {y_col} due to error: {e}"
                )
                return None
            
            full_body_data[x_col] = x
            full_body_data[y_col] = y

        return full_body_data
    
    ################################ Interaction events and associated metrics #################################

    def find_flyball_interactions(self, gap_between_events=4, event_min_length=2, thresh=[0, 70]):
        """This function applies find_interaction_events for each fly in the flytrack dataset and each ball in the balltrack dataset. It returns a dictionary where keys are the fly and ball indices and values are the interaction events.

        Args:
            gap_between_events (int, optional): The minimum gap required between two events, expressed in seconds. Defaults to 4.
            event_min_length (int, optional): The minimum length of an event, expressed in seconds. Defaults to 2.
            thresh (list, optional): The lower and upper limit values (in pixels) for the signal to be considered an event. Defaults to [0, 70].

        Returns:
            dict: A nested dictionary where the outer keys are fly indices and the inner keys are ball indices, with interaction events as values.
        """
        
        if self.flytrack is None or self.balltrack is None:
            print(f"Skipping interaction events for {self.name} due to missing tracking data.")
            return None

        fly_interactions = {}
        
        for fly_idx in range(0, len(self.flytrack.objects)):
            fly_data = self.flytrack.objects[fly_idx].dataset
            
            for ball_idx in range(0, len(self.balltrack.objects)):
                # print(f"Fly data for fly_{fly_idx}:")
                # print(fly_data)
                # print(f"Fly data columns: {fly_data.columns}")
                
                
                ball_data = self.balltrack.objects[ball_idx].dataset
                #print(self.balltrack.dataset[self.balltrack.dataset["object"] == f"ball_{ball_idx}"])
                # print(f"Ball data for ball_{ball_idx}:")
                # print(f"Ball data columns: {ball_data.columns}")
                # print(ball_data)
                
                interaction_events = find_interaction_events(
                    fly_data,
                    ball_data,
                    nodes1=["thorax"],
                    nodes2=["centre"],
                    threshold=thresh,
                    gap_between_events=gap_between_events,
                    event_min_length=event_min_length,
                    fps=self.experiment.fps
                )
                
                print(f"Interaction events for fly {fly_idx} and ball {ball_idx}: {interaction_events}")
                
                if fly_idx not in fly_interactions:
                    fly_interactions[fly_idx] = {}
                
                fly_interactions[fly_idx][ball_idx] = interaction_events
                    
        return fly_interactions

    def get_events_number(self):
        """
        For each fly ball key in the interaction_events dictionary, this method returns the number of events.

        Returns:
            dict: A dictionary where keys are fly ball keys and values are the number of events.
        """
        
        events_count = {}
        
        for key, events in self.interaction_events.items():
            event_numbers = len(events)
            print(f"Number of events for {key}: {event_numbers}")
            events_count[key] = event_numbers

        # Return the dictionary of events count
        return events_count

    def compute_metrics(self):
        """
        Compute and store various metrics for each pair of fly and ball.
        """
        for fly_idx, ball_dict in self.interaction_events.items():
            for ball_idx, events in ball_dict.items():
                key = f"fly_{fly_idx}_ball_{ball_idx}"
                print(f"Computing metrics for {key}")
                
                self.metrics[key] = {
                    "max_event": self.get_final_event(fly_idx, ball_idx),
                    "final_event": self.get_final_event(fly_idx, ball_idx),
                    "significant_events": self.get_significant_events(fly_idx, ball_idx),
                    "breaks": self.find_breaks(fly_idx, ball_idx),
                    "cumulated_breaks_duration": self.get_cumulated_breaks_duration(fly_idx, ball_idx),
                    "events_direction": self.find_events_direction(fly_idx, ball_idx)
                }
                
                print(self.metrics[key])
                
    def find_event_by_distance(self, fly_idx, ball_idx, threshold, distance_type="max"):
        """
        Find the event where the fly pushed the ball to a certain distance using Euclidean distance.

        Args:
            fly_idx (int): The index of the fly.
            ball_idx (int): The index of the ball.
            threshold (float): The distance threshold.
            distance_type (str): The type of distance to check ("max" or "min").

        Returns:
            tuple: The event and its index if found, otherwise (None, None).
        """
        
        ball_data = self.balltrack.objects[ball_idx].dataset

        # Calculate the Euclidean distance for each frame
        ball_data["euclidean_distance"] = np.sqrt(
            (ball_data["x_centre"] - ball_data["x_centre"].iloc[0])**2 +
            (ball_data["y_centre"] - ball_data["y_centre"].iloc[0])**2
        )
        
        if distance_type == "max":
            max_distance = ball_data["euclidean_distance"].max() - threshold
            distance_check = lambda event: ball_data.loc[event[0]:event[1], "euclidean_distance"].max() >= max_distance
        elif distance_type == "threshold":
            distance_check = lambda event: ball_data.loc[event[0]:event[1], "euclidean_distance"].max() >= threshold
        else:
            raise ValueError("Invalid distance_type. Use 'max' or 'threshold'.")

        try:
            event, event_index = next(
                (event, i)
                for i, event in enumerate(self.interaction_events[fly_idx][ball_idx])
                if distance_check(event)
            )
        except StopIteration:
            event, event_index = None, None

        return event, event_index

    def get_max_event(self, fly_idx, ball_idx, threshold=10):
        """
        Find the event at which the fly pushed the ball to its maximum relative distance using Euclidean distance.
        """
        return self.find_event_by_distance(fly_idx, ball_idx, threshold, distance_type="max")

    def get_final_event(self, fly_idx, ball_idx, threshold=170):
        """
        Find the event (if any) where the fly pushed the ball at least 170 px away from its initial position using Euclidean distance.
        """
        return self.find_event_by_distance(fly_idx, ball_idx, threshold, distance_type="threshold")
    
    def get_significant_events(self, fly_idx, ball_idx, distance=2):
        """
        Get the events where the ball was displaced by more than a given distance.
        """

        ball_data = self.balltrack.objects[ball_idx].dataset

        significant_events = [
            event
            for event in self.interaction_events[fly_idx][ball_idx]
            if self.check_yball_variation(event, ball_data, threshold=distance)
        ]

        return significant_events

    def check_yball_variation(self, event, ball_data, threshold=5):
        """
        Check if the variation in the 'yball' value during an event exceeds a given threshold.
        """
        yball_event = ball_data.loc[event[0]:event[1], "y_centre"]
        variation = yball_event.max() - yball_event.min()
        return variation > threshold

    def find_breaks(self, fly_idx, ball_idx):
        """
        Finds the periods where the fly is not interacting with the ball.
        """
        
        ball_data = self.balltrack.objects[ball_idx].dataset

        breaks = []
        if not self.interaction_events[fly_idx][ball_idx]:
            breaks.append((0, len(ball_data), len(ball_data)))
            return breaks

        if self.interaction_events[fly_idx][ball_idx][0][0] > 0:
            breaks.append((0, self.interaction_events[fly_idx][ball_idx][0][0], self.interaction_events[fly_idx][ball_idx][0][0]))

        for i, event in enumerate(self.interaction_events[fly_idx][ball_idx][:-1]):
            start = event[1]
            end = self.interaction_events[fly_idx][ball_idx][i + 1][0]
            duration = end - start
            breaks.append((start, end, duration))

        if self.interaction_events[fly_idx][ball_idx][-1][1] < len(ball_data):
            breaks.append(
                (
                    self.interaction_events[fly_idx][ball_idx][-1][1],
                    len(ball_data),
                    len(ball_data) - self.interaction_events[fly_idx][ball_idx][-1][1],
                )
            )

        return breaks

    def get_cumulated_breaks_duration(self, fly_idx, ball_idx):
        """
        Compute the total duration of the breaks between events.
        """
        breaks = self.find_breaks(fly_idx, ball_idx)
        cumulated_breaks_duration = sum([break_[2] for break_ in breaks])
        return cumulated_breaks_duration

    def find_events_direction(self, fly_idx, ball_idx):
        """
        Find the events where the fly pushed or pulled the ball.
        """
        
        fly_data = self.flytrack.objects[fly_idx].dataset
        ball_data = self.balltrack.objects[ball_idx].dataset
        
        significant_events = self.get_significant_events(fly_idx, ball_idx)

        pushing_events = []
        pulling_events = []

        for event in significant_events:
            start_roi = event[0]
            end_roi = event[1]

            # Compute the Euclidean distance between the ball and the fly at the start and end of the event
            start_distance = np.sqrt(
                (ball_data.loc[start_roi, "x_centre"] - fly_data.loc[start_roi, "x_thorax"])**2 +
                (ball_data.loc[start_roi, "y_centre"] - fly_data.loc[start_roi, "y_thorax"])**2
            )
            end_distance = np.sqrt(
                (ball_data.loc[end_roi, "x_centre"] - fly_data.loc[start_roi, "x_thorax"])**2 +
                (ball_data.loc[end_roi, "y_centre"] - fly_data.loc[start_roi, "y_thorax"])**2
            )

            #print(f"Start distance: {start_distance}, End distance: {end_distance}")
            
            if end_distance > start_distance:
                #print(f"Pushing event: {event}")
                pushing_events.append(event)
            else:
                #print(f"Pulling event: {event}")
                pulling_events.append(event)

        return pushing_events, pulling_events
    
    ################################ F1 experiment related functions ################################
    
    def find_adjusted_time(self, distances):
        """
        Find the time when the ball has moved by each distance from the initial position.

        Parameters:
            distances (list): A list of distances to check.

        Returns:
            dict: A dictionary where keys are ball types and values are tuples containing ball movement times and adjusted ball movement times.
        """
        results = {}

        for ball_type, flyball_positions in self.flyball_positions.items():
            ball_type = flyball_positions["ball_type"].iloc[0]
            initial_y_centre = flyball_positions["yball"].iloc[0]

            ball_movement_times = {}
            adjusted_ball_movement_times = {}

            for distance in distances:
                filtered_ball_data = flyball_positions[abs(flyball_positions["yball"] - initial_y_centre) > distance]
                if not filtered_ball_data.empty:
                    ball_movement_time = filtered_ball_data.iloc[0]["time"]
                    ball_movement_times[distance] = ball_movement_time
                    if ball_type == "generalisation" and pd.notna(self.exit_time):
                        adjusted_ball_movement_times[distance] = ball_movement_time - self.exit_time
                    else:
                        adjusted_ball_movement_times[distance] = np.nan
                else:
                    ball_movement_times[distance] = np.nan
                    adjusted_ball_movement_times[distance] = np.nan

            results[ball_type] = (ball_movement_times, adjusted_ball_movement_times)

        return results

    def generate_clip(
        self, event, outpath=None, fps=None, width=None, height=None, tracks=False
    ):
        """
        Generate a video clip for a given event.

        This method creates a video clip from the original video for the duration of the event. It also adds text to each frame indicating the event number and start time. If the 'yball' value varies more than a certain threshold during the event, a red dot is added to the frame.

        Args:
            event (list or int): : A list containing the start and end indices of the event in the 'flyball_positions' DataFrame. Alternatively, an integer can be provided to indicate the index of the event in the 'interaction_events' list.
            outpath (Path): The directory where the output video clip should be saved.
            fps (int): The frames per second of the original video.
            width (int): The width of the output video frames.
            height (int): The height of the output video frames.

        Returns:
            str: The path to the output video clip.
        """

        # If no outpath is provided, use a default path based on the fly's name and the event number
        if not outpath:
            outpath = get_labserver() / "Videos"

        # Check if the event is an integer or a list
        if isinstance(event, int):
            event = self.interaction_events[event - 1]

        start_frame, end_frame = event[0], event[1]
        cap = cv2.VideoCapture(str(self.video))

        # If no fps, width or height is provided, use the original video's fps, width and height
        if not fps:
            fps = cap.get(cv2.CAP_PROP_FPS)
        if not width:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        if not height:
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        try:
            start_time = start_frame / fps
            start_time_str = str(datetime.timedelta(seconds=int(start_time)))
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")

            # Get the index of the event in the list to apply it to the output file name
            event_index = self.interaction_events.index(event)

            if outpath == get_labserver() / "Videos":
                clip_path = outpath.joinpath(
                    f"{self.name}_{event_index}.mp4"
                ).as_posix()
            else:
                clip_path = outpath.joinpath(f"output_{event_index}.mp4").as_posix()
            out = cv2.VideoWriter(clip_path, fourcc, fps, (height, width))
            try:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                for _ in range(start_frame, end_frame):
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # If tracks is True, add the tracking data to the frame
                    if tracks and self.flyball_positions is not None:
                        # Get the tracking data for the current frame
                        flyball_coordinates = self.flyball_positions.loc[_]

                        # Use the draw_circles method to add the tracking data to the frame
                        frame = self.draw_circles(frame, flyball_coordinates)

                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

                    # Write some Text
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    text = f"Event:{event_index+1} - start:{start_time_str}"
                    font_scale = width / 150
                    thickness = int(4 * font_scale)
                    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)

                    # Position the text at the top center of the frame
                    text_x = (frame.shape[1] - text_size[0]) // 2
                    text_y = 25
                    cv2.putText(
                        frame,
                        text,
                        (text_x, text_y),
                        font,
                        font_scale,
                        (255, 255, 255),
                        thickness,
                        cv2.LINE_AA,
                    )

                    # Check if yball value varies more than threshold
                    if self.check_yball_variation(event)[
                        0
                    ]:  # You need to implement this function
                        # Add red dot to segment
                        dot = np.zeros((10, 10, 3), dtype=np.uint8)
                        dot[:, :, 0] = 0
                        dot[:, :, 1] = 0
                        dot[:, :, 2] = 255
                        dot = cv2.resize(dot, (20, 20))

                        # Position the dot right next to the text at the top of the frame
                        dot_x = (
                            text_x + text_size[0] + 10
                        )  # Position the dot right next to the text with a margin of 10

                        # Adjusted position for dot_y to make it slightly higher
                        dot_y_adjustment_factor = 1.2
                        dot_y = (
                            text_y
                            - int(dot.shape[0] * dot_y_adjustment_factor)
                            + text_size[1] // 2
                        )

                        frame[
                            dot_y : dot_y + dot.shape[0], dot_x : dot_x + dot.shape[1]
                        ] = dot

                    # Write the frame into the output file
                    out.write(frame)

            # Release everything when done
            finally:
                out.release()
        finally:
            cap.release()
        return clip_path

    def concatenate_clips(self, clips, outpath, fps, width, height, vidname):
        """
        Concatenate multiple video clips into a single video.

        This method takes a list of video clip paths, reads each clip frame by frame, and writes the frames into a new video file. The new video file is saved in the specified output directory with the specified name.

        Args:
            clips (list): A list of paths to the video clips to be concatenated.
            outpath (Path): The directory where the output video should be saved.
            fps (int): The frames per second for the output video.
            width (int): The width of the output video frames.
            height (int): The height of the output video frames.
            vidname (str): The name of the output video file (without the extension).

        """
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            outpath.joinpath(f"{vidname}.mp4").as_posix(), fourcc, fps, (height, width)
        )
        try:
            for clip_path in clips:
                cap = cv2.VideoCapture(clip_path)
                try:
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        out.write(frame)
                finally:
                    cap.release()
        finally:
            out.release()

    def generate_interactions_video(self, outpath=None, tracks=False):
        """
        Generate a video of the fly's interactions with the ball.

        This method detects interaction events, generates a video clip for each event, concatenates the clips into a single video, and saves the video in the specified output directory. The video is named after the fly's name and genotype. If the genotype is not defined, 'undefined' is used instead. After the video is created, the individual clips are deleted.

        Args:
            outpath (Path, optional): The directory where the output video should be saved. If None, the video is saved in the fly's directory. Defaults to None.
        """

        if self.flyball_positions is None:
            print(f"No tracking data available for {self.name}. Skipping...")
            return

        if outpath is None:
            outpath = self.directory
        events = self.interaction_events
        clips = []

        cap = cv2.VideoCapture(str(self.video))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        vidname = f"{self.name}_{self.Genotype if self.Genotype else 'undefined'}"

        for i, event in enumerate(events):
            clip_path = self.generate_clip(event, outpath, fps, width, height, tracks)
            clips.append(clip_path)
        self.concatenate_clips(clips, outpath, fps, width, height, vidname)
        for clip_path in clips:
            os.remove(clip_path)
        print(f"Finished processing {vidname}!")

    def draw_circles(self, frame, flyball_coordinates):
        """
        Draw circles at the positions of the fly and ball in a video frame.

        Parameters:
        frame (numpy.ndarray): The video frame.
        fly_coordinates (pandas.Series): The tracking data for the fly.
        ball_coordinates (pandas.Series): The tracking data for the ball.
        frame_number (int): The number of the frame.

        Returns:
        numpy.ndarray: The video frame with the circles.
        """

        # Extract the x and y coordinates from the pandas Series
        fly_pos = tuple(
            map(int, [flyball_coordinates["xfly"], flyball_coordinates["yfly"]])
        )
        ball_pos = tuple(
            map(int, [flyball_coordinates["xball"], flyball_coordinates["yball"]])
        )

        # Draw a smaller circle at the fly's position
        cv2.circle(frame, fly_pos, 5, (0, 0, 255), -1)  # Adjust the radius to 5

        # Draw a smaller circle at the ball's position
        cv2.circle(frame, ball_pos, 5, (255, 0, 0), -1)  # Adjust the radius to 5

        # Draw an empty circle around the ball
        cv2.circle(
            frame, ball_pos, 11, (255, 0, 0), 2
        )  # Adjust the radius to 25 and the thickness to 2

        return frame

    def generate_preview(
        self, speed=60.0, save=False, preview=False, output_path=None, tracks=True
    ):
        """
        Generate an accelerated version of the video using moviepy.

        This method uses the moviepy library to speed up the video, add circles at the positions of the fly and ball in each frame if tracking data is provided, and either save the resulting video or preview it using Pygame.

        Parameters:
        speed (float): The speedup factor. For example, 2.0 will double the speed of the video.
        save (bool, optional): Whether to save the sped up video. If True, the video is saved. If False, the video is previewed using Pygame. Defaults to False.
        output_path (str, optional): The path to save the sped up video. If not provided and save is True, a default path will be used. Defaults to None.
        tracks (dict, optional): A dictionary containing the tracking data for the fly and ball. Each key should be a string ('fly' or 'ball') and each value should be a numpy array with the x and y coordinates for each frame. Defaults to None.
        """

        if save and output_path is None:
            # Use the default output path
            output_path = (
                get_labserver()
                / "Videos"
                / "Previews"
                / f"{self.name}_{self.Genotype if self.Genotype else 'undefined'}_x{speed}.mp4"
            )

        # Load the video file
        clip = VideoFileClip(self.video.as_posix())

        # If tracks is True, add circles to the video

        # Create a new video clip with the draw_circles function applied to each frame
        if tracks:
            # Check if tracking data is available
            if self.flyball_positions is None:
                raise ValueError("No tracking data available.")

            # Create a new video clip with the draw_circles function applied to each frame
            marked_clip = VideoClip(
                lambda t: self.draw_circles(
                    clip.get_frame(t),
                    self.flyball_positions.loc[
                        round(t * clip.fps)  # Ensure correct frame is accessed
                    ],
                ),
                duration=clip.duration,
            )
        else:
            marked_clip = clip

        # Apply speed effect
        sped_up_clip = marked_clip.fx(vfx.speedx, speed)

        # If saving, write the new video clip to a file
        if save:
            print(f"Saving {self.video.name} at {speed}x speed in {output_path.parent}")
            sped_up_clip.write_videofile(
                str(output_path), fps=clip.fps
            )  # Save the sped-up clip

        # If not saving, preview the new video clip
        if preview:
            # Check if running over SSH
            if "SSH_CLIENT" in os.environ or "SSH_TTY" in os.environ:
                raise EnvironmentError(
                    "Preview mode shouldn't be run over SSH. Set preview argument to False or run locally."
                )

            # Initialize Pygame display
            pygame.display.init()

            # Set the title of the Pygame window
            pygame.display.set_caption(f"Preview (speed = x{speed})")

            print(f"Previewing {self.video.name} at {speed}x speed")

            sped_up_clip.preview(fps=self.experiment.fps * speed)

            # Manually close the preview window
            pygame.quit()

        # Close the video file to release resources
        clip.close()

        if not save and not preview:
            print("No action specified. Set save or preview argument to True.")


class Experiment:
    """
    A class for an experiment. This represents a folder containing multiple flies, each of which is represented by a Fly object.
    """

    def __init__(self, directory, metadata_only=False):
        """
        Parameters
        ----------
        directory : Path
            The path to the experiment directory.

        Attributes
        ----------
        directory : Path
            The path to the experiment directory.
            metadata : dict
            A dictionary containing the metadata for the experiment.
            fps : str
            The frame rate of the videos.
        """
        self.directory = Path(directory)
        self.metadata = self.load_metadata()
        self.fps = self.load_fps()
        
        # If metadata_only is True, don't load the flies
        if not metadata_only:
            self.flies = self.load_flies()

        # Check if the experiment directory contains a grid image and if there is "corridor" in sub subfolder names. if no grid, generate it
        if not (self.directory / "grid.png").exists() and any("corridor" in str(subdir) for subdir in self.directory.rglob('*')):
            self.generate_grid()

    def __str__(self):
        # Generate a list of unique genotypes from the flies in the experiment
        tested_genotypes = list(set([fly.Genotype for fly in self.flies]))

        return f"Experiment: {self.directory.name}\n  Genotypes: {', '.join(tested_genotypes)}\n  Flies: {len(self.flies)}\n  FPS: {self.fps}"

    def __repr__(self):
        return f"Experiment({self.directory})"

    def load_metadata(self):
        """
        Loads the metadata for the experiment. The metadata is stored in a JSON file in the experiment directory. The file is loaded as a dictionary and each variable is stored as a key in the dictionary. Each variable key contains a dictionary with the arena number as the key and the value for that variable in that arena as the value.

        Returns:
            dict: A dictionary containing the metadata for the experiment.
        """
        with open(self.directory / "Metadata.json", "r") as f:
            metadata = json.load(f)
            variables = metadata["Variable"]
            metadata_dict = {}
            for var in variables:
                metadata_dict[var] = {}
                for arena in range(1, 10):
                    arena_key = f"Arena{arena}"
                    var_index = variables.index(var)
                    metadata_dict[var][arena_key] = metadata[arena_key][var_index]

            # In the metadata_dict, make all they Arena subkeys lower case

            for var in variables:
                metadata_dict[var] = {
                    k.lower(): v for k, v in metadata_dict[var].items()
                }
            # print(metadata_dict)
            return metadata_dict

    def load_fps(self):
        """
        Loads the frame rate of the videos in the experiment directory.

        Returns:
            int: The frame rate of the videos.
        """
        # Load the fps value from the fps.npy file in the experiment directory
        fps_file = self.directory / "fps.npy"
        if fps_file.exists():
            fps = np.load(fps_file)

        else:
            fps = 30
            print(
                f"Warning: fps.npy file not found in {self.directory}; Defaulting to 30 fps."
            )

        return fps

    def load_flies(self):
        """
        Loads all flies in the experiment directory. Find subdirectories containing at least one .mp4 file, then find all .mp4 files that are named the same as their parent directory. Create a Fly object for each found folder.

        Returns:
            list: A list of Fly objects.
        """
        # Find all directories containing at least one .mp4 file
        mp4_directories = [
            dir for dir in self.directory.glob("**/*") if any(dir.glob("*.mp4"))
        ]

        # Find all .mp4 files that are named the same as their parent directory
        mp4_files = []
        for dir in mp4_directories:
            if dir.is_dir():
                try:
                    # Look for a video file named the same as the directory
                    mp4_file = list(dir.glob(f"{dir.name}.mp4"))[0]
                except IndexError:
                    try:
                        # Look for a video file named with the parent directory and corridor
                        mp4_file = list(dir.glob(f"{dir.parent.name}_corridor_{dir.name[-1]}.mp4"))[0]
                    except IndexError:
                        try:
                            # Look for any .mp4 file in the directory
                            mp4_file = list(dir.glob("*.mp4"))[0]
                        except IndexError:
                            print(f"No video found for {dir.name}. Moving to the next directory.")
                            continue  # Move on to the next directory
                print(f"Found video {mp4_file.name} for {dir.name}")
                mp4_files.append(mp4_file)

        print(mp4_files)

        # Create a Fly object for each .mp4 file
        flies = []
        for mp4_file in mp4_files:
            print(f"Loading fly from {mp4_file.parent}")
            try:
                fly = Fly(mp4_file.parent, experiment=self)
                flies.append(fly)
            except TypeError as e:
                print(f"Error while loading fly from {mp4_file.parent}: {e}")

        return flies

    def find_flies(self, on, value):
        """
        Makes a list of Fly objects matching a certain criterion.

        Parameters
        ----------
        on : str
            The name of the attribute to filter on.

        value : str
            The value of the attribute to filter on.

        Returns
        ----------
        list
            A list of Fly objects matching the criterion.
        """

        return [fly for fly in self.flies if getattr(fly, on, None) == value]

    def generate_grid(self, preview=False, overwrite=False):
        # Check if the grid image already exists
        if (self.directory / "grid.png").exists() and not overwrite:
            print(f"Grid image already exists for {self.directory.name}")
            return
        else:
            print(f"Generating grid image for {self.directory.name}")

            frames = []
            starts = []
            paths = []

            for fly in self.flies:
                frame, start, end = fly.detect_boundaries()
                if frame is not None and start is not None:
                    frames.append(frame)
                    starts.append(start)
                    paths.append(fly.video)

            # Set the number of rows and columns for the grid

            nrows = 9
            ncols = 6

            # Create a figure with subplots
            fig, axs = plt.subplots(nrows, ncols, figsize=(20, 20))

            # Loop over the frames, minimum row indices, and video paths
            for i, (frame, start, flypath) in enumerate(zip(frames, starts, paths)):
                # Get the row and column index for this subplot
                row = i // ncols
                col = i % ncols

                # Plot the frame on this subplot
                try:
                    axs[row, col].imshow(frame, cmap="gray", vmin=0, vmax=255)
                except:
                    print(f"Error: Could not plot frame {i} for video {flypath}")
                    # go to the next folder
                    continue

                # Plot the horizontal lines on this subplot
                axs[row, col].axhline(start, color="red")
                axs[row, col].axhline(start - 290, color="blue")

            # Remove the axis of each subplot and draw them closer together
            for ax in axs.flat:
                ax.axis("off")
            plt.subplots_adjust(wspace=0, hspace=0)

            # Save the grid image in the main folder
            plt.savefig(self.directory / "grid.png")

            if preview == True:
                plt.show()


class Dataset:
    def __init__(
        self,
        source,
        brain_regions_path="/mnt/upramdya_data/MD/Region_map_240312.csv",
    ):
        """
        A class to generate a Dataset from Experiments and Fly objects.

        It is in essence a list of Fly objects that can be used to generate a pandas DataFrame containing chosen metrics for each fly.

        Parameters
        ----------
        source : can either be a list of Experiment objects, one Experiment object, a list of Fly objects or one Fly object.

        """
        self.source = source

        # Define the experiments and flies attributes
        if isinstance(source, list):
            # If the source is a list, check if it contains Experiment or Fly objects, otherwise raise an error
            if type(source[0]).__name__ == "Experiment":
                # If the source contains Experiment objects, generate a dataset from the experiments
                self.experiments = source

                self.flies = [
                    fly for experiment in self.experiments for fly in experiment.flies
                ]

            elif type(source[0]).__name__ == "Fly":
                # make a list of distinct experiments associated with the flies
                self.experiments = list(set([fly.experiment for fly in source]))

                self.flies = source

            else:
                raise TypeError(
                    "Invalid source format: source must be a (list of) Experiment objects or a list of Fly objects"
                )

        elif type(source).__name__ == "Experiment":
            # If the source is an Experiment object, generate a dataset from the experiment
            self.experiments = [source]

            self.flies = source.flies

        elif type(source).__name__ == "Fly":
            # If the source is a Fly object, generate a dataset from the fly
            self.experiments = [source.experiment]

            self.flies = [source]
        else:
            raise TypeError(
                "Invalid source format: source must be a (list of) Experiment objects or a list of Fly objects"
            )

        self.flies = [
            fly
            for fly in self.flies
            if hasattr(fly, "flyball_positions") and fly.flyball_positions is not None
        ]
        self.flies = [fly for fly in self.flies if not fly.valid_data]

        self.brain_regions_path = brain_regions_path
        self.regions_map = pd.read_csv(self.brain_regions_path)

        self.metadata = []

    def __str__(self):
        # Look for recurring words in the experiment names
        experiment_names = [
            experiment.directory.name for experiment in self.experiments
        ]
        experiment_names = "_".join(experiment_names)
        experiment_names = experiment_names.split("_")  # Split by "_"

        # Ignore certain labels
        labels_to_ignore = {"Tracked", "Videos"}
        experiment_names = [
            name for name in experiment_names if name not in labels_to_ignore
        ]

        # Ignore words that are found only once
        experiment_names = [
            name for name in experiment_names if experiment_names.count(name) > 1
        ]

        experiment_names = Counter(experiment_names)
        experiment_names = experiment_names.most_common(3)
        experiment_names = [name for name, _ in experiment_names]
        experiment_names = ", ".join(experiment_names)

        return f"Dataset with {len(self.flies)} flies and {len(self.experiments)} experiments\nkeyword: {experiment_names}"

    def __repr__(self):
        # Adapt the repr function to the source attribute
        # If the source is a list, check if it is Fly or Experiment objects
        if isinstance(self.source, list):
            if isinstance(self.source[0], Experiment):
                return f"Dataset({[experiment.directory for experiment in self.experiments]})"
            elif isinstance(self.source[0], Fly):
                return f"Dataset({[fly.directory for fly in self.flies]})"
        elif isinstance(self.source, Experiment):
            return f"Dataset({self.experiments[0].directory})"
        elif isinstance(self.source, Fly):
            return f"Dataset({self.flies[0].directory})"

    def generate_dataset(
        self, metrics="coordinates", success_cutoff=True, time_range=None
    ):
        """Generates a pandas DataFrame from a list of Experiment objects. The dataframe contains the smoothed fly and ball positions for each experiment.

        Args:
            experiments (list): A list of Experiment objects.
            metrics (str): The kind of dataset to generate. Currently, the following metrics are available:
            - 'coordinates': The fly and ball coordinates for each frame.
            - 'summary': Summary metrics for each fly. These are single values for each fly (e.g. number of events, duration of the breaks between events, etc.). A list of available summary metrics can be found in _prepare_dataset_summary_metrics documentation.

        Returns:
            dict: A dictionary where keys are ball types and values are DataFrames containing selected metrics for each fly and associated metadata.
        """

        try:
            dataset_dict = {}

            if metrics == "coordinates":
                for fly in self.flies:
                    for ball_type, flyball_positions in fly.flyball_positions.items():
                        dataset = self._prepare_dataset_coordinates(
                            fly, flyball_positions, success_cutoff=success_cutoff, time_range=time_range
                        )
                        if ball_type not in dataset_dict:
                            dataset_dict[ball_type] = []
                        dataset_dict[ball_type].append(dataset)

            elif metrics == "summary":
                for fly in self.flies:
                    for ball_type, flyball_positions in fly.flyball_positions.items():
                        df = self._prepare_dataset_summary_metrics(
                            fly, flyball_positions, success_cutoff=success_cutoff, time_range=time_range
                        )
                        if df is not None:
                            if ball_type not in dataset_dict:
                                dataset_dict[ball_type] = []
                            dataset_dict[ball_type].append(df)
                        else:
                            print(f"Empty DataFrame for fly {fly.directory}")

            # Concatenate datasets for each ball type
            for ball_type in dataset_dict:
                if dataset_dict[ball_type]:  # Only concatenate if the list is not empty
                    dataset_dict[ball_type] = pd.concat(dataset_dict[ball_type], ignore_index=True).reset_index(drop=True)
                else:
                    dataset_dict[ball_type] = pd.DataFrame()

            self.data = dataset_dict

        except Exception as e:
            print(f"An error occurred: {e}")
            traceback.print_exc()
            self.data = {}

        return self.data

    def _prepare_dataset_coordinates(
        self, fly, flyball_positions, success_cutoff=True, time_range=None
    ):
        """
        Helper function to prepare individual fly dataset with fly and ball coordinates. It also adds the fly name, experiment name and arena metadata as categorical data.

        Args:
            fly (Fly): A Fly object.
            flyball_positions (pd.DataFrame): DataFrame containing the fly and ball coordinates.
            success_cutoff (bool): Whether to apply the success cutoff. Defaults to True.
            time_range (list): A list containing the start and end times for the dataset. Defaults to None.

        Returns:
            pandas.DataFrame: A DataFrame containing the fly's coordinates and associated metadata.
        """

        dataset = flyball_positions

        if time_range is not None:
            # If one value is provided, set the end of the range to the end of the video and the start to the provided value
            if len(time_range) == 1:
                dataset = dataset[dataset["time"] >= time_range[0]]

                # Reindex events if any
                unique_events = dataset["event"].dropna().unique()
                event_mapping = {
                    event: i + 1 for i, event in enumerate(unique_events)
                }
                dataset["event"] = dataset["event"].map(event_mapping)
            # If two values are provided, set the start and end of the range to the provided values
            elif len(time_range) == 2:
                dataset = dataset[
                    (dataset["time"] >= time_range[0])
                    & (dataset["time"] <= time_range[1])
                ]

                # Reindex events
                unique_events = dataset["event"].dropna().unique()
                event_mapping = {event: i + 1 for i, event in enumerate(unique_events)}
                dataset["event"] = dataset["event"].map(event_mapping)
            else:
                print(
                    "Invalid time range. Please provide one or two values for the time range."
                )

        if success_cutoff:
            cutoff_index = (
                # Get the initial yball and subtract -180. Find the first index where yball is less than this value
                (dataset["yball"] <= dataset["yball"].iloc[0] - 180)
            ).idxmax()
            if cutoff_index != 0:  # idxmax returns 0 if no True value is found
                dataset = dataset[:cutoff_index]

        dataset = self._add_metadata(dataset, fly)

        return dataset

    # TODO : Events should be reannotated if the dataset is subsetted
    def _prepare_dataset_summary_metrics(
        self,
        fly,
        metrics=[
            "NumberEvents",
            "FinalEvent",
            "FinalTime",
            "SignificantEvents",
            "SignificantFirst",
            "SignificantFirstTime",
            "Pushes",
            "Pulls",
            "PullingRatio",
            "InteractionProportion",
            "AhaMoment",
            "AhaMomentIndex",
            "InsightEffect",
            "TimeToFinish",
            "SignificantRatio",
        ],
        success_cutoff=True,
        time_range=None,
    ):
        """
        Prepares a dataset with summary metrics for a given fly. The metrics are computed for all events, but only the ones specified in the 'metrics' argument are included in the returned DataFrame.

        Available metrics:
        - NumberEvents: The number of events found in the flyball_positions DataFrame.
        - FinalEvent: The event at which the fly pushed the ball to its maximum relative distance from the start of the corridor.
        - FinalTime: The time at which the final event occurred.
        - SignificantEvents: The number of events where the ball was displaced by more than a given distance (this distance is set in check_yball_variation function and its default value is 10px).
        - SignificantRatio: The ratio of significant events to the total number of events.
        - AhaMoment: The first event at which the fly pushed the ball to at least a distance of 125px from the start of the corridor.
        - SignificantFirst: The index of the first significant event.
        - SignificantFirstTime: The time at which the first significant event occurred.
        - CumulatedBreaks: The total duration of the breaks between events.
        - Pushes: The number of events where the fly pushed the ball.
        - Pulls: The number of events where the fly pulled the ball.
        - PullingRatio: The ratio of the number of push events to the total number of significant events.
        - InteractionProportion: The proportion of the video or subset during which the fly was interacting with the ball.

        Args:
            fly (Fly): A Fly object.
            metrics (list): A list of metrics to include in the dataset. The metrics require valid ball and fly tracking data.
            success_cutoff (bool): If True, only events before the ball reaches the end of the corridor are considered.

        Returns:
            pandas.DataFrame: A DataFrame containing the fly's summary metrics and associated metadata.

        """
        # TODO: Implement events duration
        # TODO: Implement some metric about whether the fly brought the ball close enought to the end of the corridor

        # Store the results of function calls in variables

        fly.annotate_events()
        positions = fly.flyball_positions

        # print(positions.head())

        if success_cutoff:
            cutoff_index = (
                fly.flyball_positions["yball"] <= fly.end + 40
            ).idxmax()
            if cutoff_index != 0:  # idxmax returns 0 if no True value is found
                positions = fly.flyball_positions[:cutoff_index]

        if time_range is not None:
            # If one value is provided, set the end of the range to the end of the video and the start to the provided value
            if len(time_range) == 1:
                positions = positions[positions["time"] >= time_range[0]]

                # Reindex events
                unique_events = positions["event"].dropna().unique()
                event_mapping = {event: i + 1 for i, event in enumerate(unique_events)}
                positions["event"] = positions["event"].map(event_mapping)

            # If two values are provided, set the start and end of the range to the provided values
            elif len(time_range) == 2:
                positions = positions[
                    (positions["time"] >= time_range[0])
                    & (positions["time"] <= time_range[1])
                ]

                # Reindex events
                unique_events = positions["event"].dropna().unique()
                event_mapping = {event: i + 1 for i, event in enumerate(unique_events)}
                positions["event"] = positions["event"].map(event_mapping)

            else:
                print(
                    "Invalid time range. Please provide one or two values for the time range."
                )

        final_event = fly.get_final_event(subset=positions)
        significant_events = fly.get_significant_events(subset=positions)
        events_direction = fly.find_events_direction(subset=positions)
        aha_moment = fly.get_significant_events(distance=50, subset=positions)
        aha_moment_index = (
            fly.find_interaction_events(subset=positions).index(aha_moment[0])
            if aha_moment
            else None
        )

        # Create a dictionary of metric calculation functions
        metric_funcs = {
            "NumberEvents": lambda: [fly.get_events_number(subset=positions)],
            "FinalEvent": lambda: (
                [final_event[1]] if final_event != (None, None) else [None]
            ),
            "FinalTime": lambda: (
                [final_event[0][2] / fly.experiment.fps]
                if final_event != (None, None)
                else [None]
            ),
            "SignificantEvents": lambda: (
                [len(significant_events)] if significant_events else [0]
            ),
            "SignificantRatio": lambda: (
                [len(significant_events) / fly.get_events_number(subset=positions)]
                if significant_events
                else [0]
            ),
            "AhaMoment": lambda: (
                [aha_moment[0][0] / fly.experiment.fps] if aha_moment else [None]
            ),
            "AhaMomentIndex": lambda: (
                [aha_moment_index] if aha_moment_index is not None else [None]
            ),
            "InsightEffect": lambda: (
                [
                    np.mean(
                        [
                            fly.check_yball_variation(event, subset=positions)[1]
                            for event in fly.find_interaction_events(subset=positions)[
                                aha_moment_index:
                            ]
                        ]
                    )
                    / np.mean(
                        [
                            fly.check_yball_variation(event, subset=positions)[1]
                            for event in fly.find_interaction_events(subset=positions)[
                                :aha_moment_index
                            ]
                        ]
                    )
                ]
                if aha_moment_index is not None
                and aha_moment_index
                < len(fly.find_interaction_events(subset=positions))
                else [None]
            ),
            "TimeToFinish": lambda: [len(positions) / fly.experiment.fps],
            "SignificantFirst": lambda: (
                [
                    fly.find_interaction_events(subset=positions).index(
                        significant_events[0]
                    )
                ]
                if significant_events
                else [None]
            ),
            "SignificantFirstTime": lambda: (
                [significant_events[0][0] / fly.experiment.fps]
                if significant_events
                else [None]
            ),
            "CumulatedBreaks": lambda: [
                fly.get_cumulated_breaks_duration(subset=positions) / fly.experiment.fps
            ],
            "InteractionProportion": lambda: [
                (positions["event"].notnull().sum()) / len(positions)
            ],
            "Pushes": lambda: [len(events_direction[0])] if events_direction else [0],
            "Pulls": lambda: [len(events_direction[1])] if events_direction else [0],
            "PullingRatio": lambda: [
                (
                    (len(events_direction[1]))
                    / (len(events_direction[0]) + len(events_direction[1]))
                    if events_direction
                    and (
                        len(events_direction[0]) != 0
                    )  # or len(events_direction[1]) != 0)
                    else float("nan")
                )
            ],
        }

        # Initialize an empty dictionary
        data = {}

        # Compute all metrics
        computed_metrics = {metric: func() for metric, func in metric_funcs.items()}

        # Add only the selected metrics to the data dictionary
        data = {
            metric: computed_metrics[metric]
            for metric in metrics
            if metric in computed_metrics
        }

        # Convert the dictionary to a DataFrame
        dataset = pd.DataFrame(data)

        dataset = self._add_metadata(dataset, fly)

        return dataset

    def _add_metadata(self, data, fly):
        """
        Adds the metadata to a dataset generated by a _prepare_dataset_... method.

        Args:
            data (pandas.DataFrame): A pandas DataFrame generated by a _prepare_dataset_... method.
            fly (Fly): A Fly object.

        Returns:
            pandas.DataFrame: A DataFrame containing the fly's coordinates and associated metadata.
        """

        try:
            dataset = data

            # Add a column with the fly name as categorical data
            dataset["fly"] = fly.name
            dataset["fly"] = dataset["fly"].astype("category")

            # Add a column with the path to the fly's folder
            dataset["flypath"] = fly.directory.as_posix()
            dataset["flypath"] = dataset["flypath"].astype("category")

            # Add a column with the experiment name as categorical data
            dataset["experiment"] = fly.experiment.directory.name
            dataset["experiment"] = dataset["experiment"].astype("category")

            # Handle missing values for 'Nickname' and 'Brain region'
            dataset["Nickname"] = (
                fly.nickname if fly.nickname is not None else "Unknown"
            )
            dataset["Brain region"] = (
                fly.brain_region if fly.brain_region is not None else "Unknown"
            )

            # Add the metadata for the fly's arena as columns
            for var, data in fly.arena_metadata.items():
                # Handle missing values in arena metadata
                data = data if data is not None else "Unknown"
                dataset[var] = data
                dataset[var] = dataset[var].astype("category")

                # If the variable name is not in the metadata list, add it
                if var not in self.metadata:
                    self.metadata.append(var)

        except Exception as e:
            print(f"Error occurred while adding metadata for fly {fly.name}: {str(e)}")
            print(f"Current dataset:\n{dataset}")

        return dataset

    # def compute_sample_size(self, data, group_by_columns=["Nickname", "Brain region"]):
    #     """
    #     Function used to compute the sample size of a dataset generated by the generate_dataset method. it compute the size of the dataset grouped by columns of interest.

    #     Args:
    #         data (pandas.DataFrame): A pandas DataFrame generated by the generate_dataset method.

    #         group_by_columns (list): A list of columns to group the data by.

    #     Returns:
    #         pandas.DataFrame: A DataFrame containing the sample size for each group.
    #     """
    #     # Group the data by the columns of interest and compute the sample size
    #     sample_size = (
    #         data.groupby(group_by_columns)
    #         .nunique()["fly"]
    #         .reset_index()
    #         .rename(columns={"fly": "SampleSize"})
    #     )

    #     # Merge the sample size with the original data
    #     data = pd.merge(data, sample_size, on=group_by_columns)

    #     return data

    def get_event_numbers(self):
        # Check that events have been annotated
        if "event" not in self.data.columns:
            raise ValueError(
                "No events have been annotated. Run the annotate_events method first."
            )

        # Group the data by Fly and Event
        GroupedData = (
            self.data.groupby(["fly", "Nickname", "Simplified region"])
            .nunique(["event"])
            .reset_index()
        )

        # Calculate sample size
        SampleSize = (
            self.data.groupby(["Nickname", "Simplified region"])
            .nunique()["fly"]
            .reset_index()
            .rename(columns={"fly": "SampleSize"})
        )

        # Merge GroupedData and SampleSize
        GroupedData = pd.merge(
            GroupedData, SampleSize, on=["Nickname", "Simplified region"]
        )

        # Make a new column with the nickname and the sample size
        GroupedData["label"] = (
            GroupedData["Nickname"]
            + " (n = "
            + GroupedData["SampleSize"].astype(str)
            + ")"
        )

        # Add the metadata to the GroupedData
        GroupedData.set_index("fly", inplace=True)

        GroupedData.update(self.dropdata[self.metadata])
        GroupedData.reset_index(inplace=True)

        # TODO: Add handling when there's no simplified region, to get the simpler version of the data
