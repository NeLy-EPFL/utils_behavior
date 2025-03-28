# import h5py
# import h5pickle as h5py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# from sklearn.manifold import TSNE
# from sklearn.decomposition import PCA
# from openTSNE import TSNE
# from openTSNE.callbacks import Callback
from tqdm.auto import tqdm

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

import multiprocessing
from multiprocessing import Pool

from concurrent.futures import ThreadPoolExecutor, as_completed
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
import moviepy
import moviepy.config as mpconfig

moviepy.config.change_settings({"IMAGEMAGICK_BINARY": "magick"})
mpconfig.change_settings(
    {"IMAGEMAGICK_BINARY": "/home/durrieu/miniforge3/envs/tracking_analysis/bin/magick"}
)

os.environ["MAGICK_FONT_PATH"] = "/etc/ImageMagick-6"
# os.environ['MAGICK_CONFIGURE_PATH'] = '/etc/ImageMagick-6'

from moviepy.editor import (
    VideoFileClip,
    clips_array,
    ColorClip,
    concatenate_videoclips,
    TextClip,
    CompositeVideoClip,
)
from moviepy.editor import VideoClip
from moviepy.editor import VideoFileClip
from moviepy.video.fx import all as vfx


# import moviepy as mpy

import pygame

import warnings

from .Utils import *
from .Processing import *
from .Sleap_utils import *

from .HoloviewsTemplates import hv_main

from dataclasses import dataclass

from scipy.signal import find_peaks, savgol_filter

sys.modules["Ballpushing_utils"] = sys.modules[
    __name__
]  # This line creates an alias for utils_behavior.Ballpushing_utils to utils_behavior.__init__ so that the previously made pkl files can be loaded.

print("Loading BallPushing utils version 10 Mar 2025")

brain_regions_path = "/mnt/upramdya_data/MD/Region_map_250116.csv"


# class ProgressCallback(Callback):
#     def __init__(self, n_iter):
#         self.pbar = tqdm(total=n_iter, desc="t-SNE progress")

#     def __call__(self, iteration, error, embedding):
#         self.pbar.update(1)
#         return False

#     def close(self):
#         self.pbar.close()


def save_object(obj, filename):
    """Save a custom object as a pickle file.

    Args:
        obj (object): The object to be saved.
        filename (Path): The path where to save the object. No need to add the .pkl extension.
    """
    # Ensure filename is a Path object
    filename = Path(filename)

    # If the filename does not end with .pkl, add it
    if filename.suffix != ".pkl":
        filename = filename.with_suffix(".pkl")

    with open(filename, "wb") as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def load_object(filename):
    """Load a custom object from a pickle file.

    Args:
        filename (Pathlib path): the path to the object. No need to add the .pkl extension.
    """

    # Ensure filename is a Path object
    filename = Path(filename)

    # If the filename does not end with .pkl, add it
    if filename.suffix != ".pkl":
        filename = filename.with_suffix(".pkl")

    with open(filename, "rb") as input:
        obj = pickle.load(input)

    return obj


def find_interaction_events(
    protagonist1,
    protagonist2,
    nodes1=["Lfront", "Rfront"],
    nodes2=["x_centre", "y_centre"],
    threshold=[0, 11],
    gap_between_events=4,
    event_min_length=2,
    fps=29,
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
            # y2 = protagonist2[f"y_{node2}"]

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
            # print(f"y2: {y2}")

            distances_node = np.sqrt(
                (protagonist1[f"x_{node1}"] - protagonist2[f"x_{node2}"]) ** 2
                + (protagonist1[f"y_{node1}"] - protagonist2[f"y_{node2}"]) ** 2
            )
            # print(f"Distances for {node1} and {node2}: {distances_node}")
            distances.append(distances_node)

    # Combine distances to find frames where any node is within the threshold distance
    combined_distances = np.min(distances, axis=0)
    interaction_frames = np.where(
        (np.array(combined_distances) > threshold[0])
        & (np.array(combined_distances) < threshold[1])
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


def find_interaction_start(
    data,
    distance_col,
    frame_col,
    threshold_multiplier=1.5,
    window_size=30,
    min_plateau_length=50,
    peak_prominence=1,
    peak_window_size=10,
):
    # Work on a copy to avoid modifying original data
    data = data.copy()

    # Smoothing and derivative calculation
    data["smoothed_distance"] = savgol_filter(
        data[distance_col], window_length=11, polyorder=3
    )
    data["smoothed_diff"] = data["smoothed_distance"].diff()
    data["smoothed_accel"] = data["smoothed_diff"].diff()  # Second derivative

    # Dynamic threshold for plateaus based on rolling standard deviation
    rolling_std = data["smoothed_diff"].rolling(window=window_size).std()
    dynamic_threshold = (
        rolling_std.mean() * threshold_multiplier
    )  # Adjust multiplier as needed

    # Plateau detection with dynamic threshold
    plateau_mask = data["smoothed_diff"].abs() < dynamic_threshold
    plateau_groups = (plateau_mask != plateau_mask.shift()).cumsum()

    # Initialize plateau markers
    data["plateau_start"] = 0
    valid_plateaus = (
        data[plateau_mask]
        .groupby(plateau_groups)
        .filter(lambda x: len(x) >= min_plateau_length)
    )
    if not valid_plateaus.empty:
        start_indices = valid_plateaus.groupby(plateau_groups).head(1).index
        data.loc[start_indices, "plateau_start"] = 1

    # Peak detection with stricter prominence
    peaks, _ = find_peaks(
        -data["smoothed_distance"], prominence=peak_prominence, width=3
    )

    # Refine peak detection for better alignment
    refined_peaks = []
    for peak in peaks:
        if peak > 0 and peak < len(data) - 1:
            # Perform local search around the detected peak to refine position
            local_region = data.iloc[
                max(0, peak - peak_window_size) : min(
                    len(data), peak + peak_window_size
                )
            ]
            true_peak_idx = local_region[
                distance_col
            ].idxmin()  # Find true minimum in this region
            refined_peaks.append(true_peak_idx)

    # Combine plateau and refined peak detections
    plateau_indices = data[data["plateau_start"] == 1].index
    all_candidates = sorted(list(plateau_indices) + refined_peaks)

    # Fallback to minimum distance if no markers found
    if not all_candidates:
        return data[distance_col].idxmin()

    return all_candidates[0]


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


def load_fly(
    mp4_file,
    experiment,
    #experiment_type,
):
    print(f"Loading fly from {mp4_file.parent}")
    try:
        fly = Fly(
            mp4_file.parent,
            experiment=experiment,
            #experiment_type=experiment_type,
        )
        if fly.tracking_data and fly.tracking_data.valid_data:
            return fly
    except TypeError as e:
        print(f"Error while loading fly from {mp4_file.parent}: {e}")
    return None


# Pixel size: 30 mm = 500 pixels, 4 mm = 70 pixels, 1.5 mm = 25 pixels


@dataclass
class Config:
    """
    Configuration class for the Ball pushing experiments

    Attributes:
    interaction_threshold: tuple: The lower and upper limit values (in pixels) for the signal to be considered an event. Defaults to (0, 70).
    time_range: tuple: The time range (in seconds) to filter the tracking data. Defaults to None.
    success_cutoff: bool: Whether to filter the tracking data based on the success cutoff time range. Defaults to False. If True, the success_cutoff_time_range computed in the FlyTrackingData class will be used to filter the tracking data.
    tracks_smoothing: bool: Whether to apply smoothing to the tracking data. Defaults to True.
    dead_threshold: int: The threshold value (in pixels traveled) for the fly to be considered dead. Defaults to 30.
    adjusted_events_normalisation: int: The normalisation value for the adjusted number of events. It's an arbitrary value that multiplies all the adjusted events values to make it more easily readable. Defaults to 1000.
    significant_threshold: int: The threshold value (in pixels) for an event to be considered significant. Defaults to 5.
    aha_moment_threshold: int: The threshold value (in pixels) for an event to be considered an Aha moment. Defaults to 20.
    success_direction_threshold: int: The threshold value (in pixels) for an event to be considered a success direction, which is used to check whether a fly is a "pusher" or a "puller", or "both. Defaults to 25.
    final_event_threshold: int: The threshold value (in pixels) for an event to be considered a final event. Defaults to 170.
    final_event_F1_threshold: int: The threshold value (in pixels) for an event to be considered a final event in the F1 condition. Defaults to 100.
    max_event_threshold: int: The threshold value (in pixels) for an event to be considered a maximum event. Defaults to 10.

    """

    # General configuration attributes
    
    experiment_type: str = "Learning"

    time_range: tuple = None
    success_cutoff: bool = False
    success_cutoff_method: str = "final_event"
    tracks_smoothing: bool = True

    log_missing = True
    log_path = get_data_server() / "MD/MultiMazeRecorder"

    keep_idle = True

    # Coordinates dataset attributes

    downsampling_factor: int = None  # Classic values used are 5 or 10.

    # Random events attributes

    generate_random: bool = False
    random_exclude_interactions: bool = True
    random_interaction_map: str = "full"  # Options: "full" or "onset"

    # Events related thresholds

    interaction_threshold: tuple = (0, 45)  # Default was 70
    gap_between_events: int = 1  # Default was 2
    events_min_length: int = 1  # Default was 2

    frames_before_onset = 10  # Default was 20
    frames_after_onset = 290  # Default was 20

    dead_threshold: int = 30
    adjusted_events_normalisation: int = 1000
    significant_threshold: int = 5
    aha_moment_threshold: int = 20
    success_direction_threshold: int = 25
    final_event_threshold: int = 170
    final_event_F1_threshold: int = 100
    max_event_threshold: int = 10

    # Skeleton tracking configuration attributes

    # skeleton_tracks_smoothing: bool = False

    # Template size
    template_width: int = 96
    template_height: int = 516

    padding: int = 20
    y_crop: tuple = (74, 0)

    # # Skeleton metrics

    contact_nodes = ["Rfront", "Lfront"]

    contact_threshold: tuple = (0, 13)
    gap_between_contacts: int = 1 / 2
    contact_min_length: int = 1 / 2

    # Skeleton metrics: longer

    skeleton_tracks_smoothing: bool = False

    # contact_nodes = ["Thorax", "Head"]

    # contact_threshold: tuple = (0, 40)
    # gap_between_contacts: int = 3 / 2
    # contact_min_length: int = 2

    # hidden_value: int = -1
    hidden_value: int = 9999

    def __post_init__(self):
        print("Config loaded with the following parameters:")
        for field_name, field_value in self.__dict__.items():
            print(f"{field_name}: {field_value}")

    def set_experiment_config(self, experiment_type):
        """
        Set the configuration for the experiment based on the experiment type.

        Args:
            experiment_type (str): The type of experiment, e.g., 'MagnetBlock', 'Training', etc.
        """
        if experiment_type == "MagnetBlock":
            self.time_range = (3600, None)
            self.final_event_threshold = 100
            # Add other specific settings for MagnetBlock
        elif experiment_type == "Learning":
            # Learning experiment specific settings
            self.trial_peak_height = 0.23  # Height threshold for peak detection
            self.trial_peak_distance = 500  # Minimum distance between peaks
            self.trial_skip_frames = 500  # Initial frames to skip in a trial
            self.trial_min_count = 2  # Minimum number of trials for a valid fly
        else:
            print(f"{experiment_type} has no particular configuration.")

    def set_property(self, property_name, value):
        """
        Set the value of a property in the configuration.

        Args:
            property_name (str): The name of the property to set.
            value: The value to set for the property.
        """
        if hasattr(self, property_name):
            setattr(self, property_name, value)
        else:
            raise AttributeError(f"Config has no attribute named '{property_name}'")


# @dataclass
class FlyMetadata:
    def __init__(self, fly):

        self.fly = fly
        self.directory = self.fly.directory
        self.experiment = self.fly.experiment
        self.arena = self.directory.parent.name
        self.corridor = self.directory.name
        self.name = f"{self.experiment.directory.name}_{self.arena}_{self.corridor}"
        self.arena_metadata = self.get_arena_metadata()

        # For each value in the arena metadata, add it as an attribute of the fly
        for var, data in self.arena_metadata.items():
            setattr(self, var, data)

        if fly.config.experiment_type == "F1":
            self.compute_F1_condition()

        self.nickname, self.brain_region = self.load_brain_regions(brain_regions_path)

        self.video = self.load_video()
        self.fps = self.experiment.fps

        self.original_size = self.get_video_size()

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

    def load_brain_regions(self, brain_regions_path):
        # Get the brain regions table
        brain_regions = pd.read_csv(brain_regions_path, index_col=0)

        # If the fly's genotype is defined in the arena metadata, find the associated nickname and brain region from the brain_regions_path file
        if "Genotype" in self.arena_metadata:
            try:
                genotype = self.arena_metadata["Genotype"]

                # If the genotype is None, skip the fly
                if genotype.lower() == "none":
                    print(f"Genotype is None: {self.name} is empty.")
                    return

                # Convert to lowercase for comparison
                lowercase_index = brain_regions.index.str.lower()
                matched_index = lowercase_index.get_loc(genotype.lower())

                self.nickname = brain_regions.iloc[matched_index]["Nickname"]
                self.brain_region = brain_regions.iloc[matched_index][
                    "Simplified region"
                ]
                self.simplified_nickname = brain_regions.iloc[matched_index][
                    "Simplified Nickname"
                ]
                self.split = brain_regions.iloc[matched_index]["Split"]
            except KeyError:
                print(
                    f"Genotype {genotype} not found in brain regions table for {self.name}. Defaulting to PR"
                )
                self.nickname = "PR"
                self.brain_region = "Control"
                self.simplified_nickname = "PR"
                self.split = "m"

        return self.nickname, self.brain_region

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

    def get_video_size(self):
        """Get the size of the video."""

        # Load the video
        video = cv2.VideoCapture(str(self.video))

        return (
            video.get(cv2.CAP_PROP_FRAME_WIDTH),
            video.get(cv2.CAP_PROP_FRAME_HEIGHT),
        )

    def compute_F1_condition(self):
        if "Pretraining" in self.arena_metadata and "Unlocked" in self.arena_metadata:
            pretraining = self.arena_metadata["Pretraining"]
            unlocked = self.arena_metadata["Unlocked"]

            if "n" in pretraining:
                self.arena_metadata["F1_condition"] = "control"
            elif "y" in pretraining:
                if "Left" in self.corridor:
                    if unlocked[0] == "y":
                        self.arena_metadata["F1_condition"] = "pretrained_unlocked"
                    else:
                        self.arena_metadata["F1_condition"] = "pretrained"
                elif "Right" in self.corridor:
                    if unlocked[1] == "y":
                        self.arena_metadata["F1_condition"] = "pretrained_unlocked"
                    else:
                        self.arena_metadata["F1_condition"] = "pretrained"
            else:
                print(f"Error: Pretraining value not valid for {self.name}")

        # Add F1_condition as an attribute of the fly
        if "F1_condition" in self.arena_metadata:
            setattr(self, "F1_condition", self.arena_metadata["F1_condition"])

    def display_metadata(self):
        """
        Print the metadata for the Fly object's arena.

        This method iterates over the arena's metadata dictionary and prints each key-value pair.

        Prints:
            str: The metadata variable name and its corresponding value, formatted as 'variable: value'.
        """
        # Print the metadata for this fly's arena
        for var, data in self.metadata.arena_metadata.items():
            print(f"{var}: {data}")


class FlyTrackingData:
    def __init__(self, fly, time_range=None, log_missing=False, keep_idle=False):
        self.fly = fly
        self.flytrack = None
        self.balltrack = None
        self.skeletontrack = None
        self.valid_data = True
        self.log_missing = log_missing
        self.keep_idle = keep_idle
        self.cutoff_reference = None  # New cutoff reference storage

        try:
            # Load tracking files
            self.balltrack = self.load_tracking_file("*ball*.h5", "ball")
            self.flytrack = self.load_tracking_file("*fly*.h5", "fly")
            self.skeletontrack = self.load_tracking_file(
                "*full_body*.h5",
                "fly",
                smoothing=self.fly.config.skeleton_tracks_smoothing,
            )

            if self.balltrack is None or (
                self.flytrack is None and self.skeletontrack is None
            ):
                print(f"Missing tracking files for {self.fly.metadata.name}")
                self.valid_data = False
                if self.log_missing:
                    self.log_missing_fly()
                return

            # Calculate euclidean distances for all balls (new line)
            self.calculate_euclidean_distances()

        except Exception as e:
            print(f"Error loading files for {self.fly.metadata.name}: {e}")
            self.valid_data = False
            if self.log_missing:
                self.log_missing_fly()
            return

        # Data quality checks
        self.valid_data = self.check_data_quality()
        self.check_dying()

        if self.valid_data or self.keep_idle:
            self.duration = self.balltrack.objects[0].dataset["time"].iloc[-1]
            self.start_x, self.start_y = self.get_initial_position()
            self.fly_skeleton = self.get_skeleton()
            self.exit_time = self.get_exit_time()
            self.adjusted_time = self.compute_adjusted_time()

        # Apply initial time range filter if specified in config
        if self.fly.config.time_range:
            self.filter_tracking_data(self.fly.config.time_range)

        # Determine success cutoff reference
        if self.fly.config.success_cutoff:
            self._determine_success_cutoff()

    def load_tracking_file(
        self,
        pattern,
        object_type,
        smoothing=None,
    ):
        """Load a tracking file for the fly."""

        if smoothing is None:
            smoothing = self.fly.config.tracks_smoothing

        try:
            tracking_file = list(self.fly.directory.glob(pattern))[0]
            return Sleap_Tracks(
                tracking_file,
                object_type=object_type,
                smoothed_tracks=smoothing,
                debug=False,
            )
        except IndexError:
            return None

    def calculate_euclidean_distances(self):
        """
        Calculate euclidean distance for all balls in the tracking data.
        This is the distance from the ball's initial position.
        """
        if self.balltrack is None:
            return

        for ball_idx in range(len(self.balltrack.objects)):
            ball_data = self.balltrack.objects[ball_idx].dataset

            # Calculate euclidean distance from initial position
            ball_data["euclidean_distance"] = np.sqrt(
                (ball_data["x_centre"] - ball_data["x_centre"].iloc[0]) ** 2
                + (ball_data["y_centre"] - ball_data["y_centre"].iloc[0]) ** 2
            )

            # Also add an alias column for learning experiments
            ball_data[f"distance_ball_{ball_idx}"] = ball_data["euclidean_distance"]

    def _determine_success_cutoff(self):
        """Calculate success cutoff based on the final event threshold."""
        if self.fly.config.success_cutoff_method == "final_event":
            ball_data = self.balltrack.objects[0].dataset
            threshold = self.fly.config.final_event_threshold

            # # Calculate ball displacement
            # ball_data["euclidean_distance"] = np.sqrt(
            #     (ball_data["x_centre"] - ball_data["x_centre"].iloc[0]) ** 2
            #     + (ball_data["y_centre"] - ball_data["y_centre"].iloc[0]) ** 2
            # )

            # Find frames exceeding threshold
            over_threshold = ball_data[ball_data["euclidean_distance"] >= threshold]

            if not over_threshold.empty:
                final_frame = over_threshold.index[0]
                self.cutoff_reference = final_frame / self.fly.experiment.fps

        # Reset cached calculations
        for attr in [
            "_interaction_events",
            "_interactions_onsets",
            "_std_interactions",
        ]:
            if hasattr(self, attr):
                delattr(self, attr)

    @property
    def interaction_events(self):
        """Chunks of time where the fly is interacting with the ball."""
        if not hasattr(self, "_interaction_events"):
            time_range = (0, self.cutoff_reference) if self.cutoff_reference else None

            self._interaction_events = self._calculate_interactions(time_range)
        return self._interaction_events

    def _calculate_interactions(self, time_range=None):
        """Actual event detection with optional time range"""
        events = {}
        for fly_idx in range(len(self.flytrack.objects)):
            events[fly_idx] = {}
            for ball_idx in range(len(self.balltrack.objects)):
                events[fly_idx][ball_idx] = self.find_flyball_interactions(
                    time_range=time_range, fly_idx=fly_idx, ball_idx=ball_idx
                )
        return events

    def find_flyball_interactions(
        self,
        gap_between_events=None,
        event_min_length=None,
        thresh=None,
        time_range=None,
        fly_idx=0,
        ball_idx=0,
    ):
        """
        Find interaction events between the fly and the ball.
        It uses the find_interaction_events function from the Ballpushing_utils module to find chunks of time where the fly ball distance is below a threshold thresh for a minimum of event_min_length seconds. It also merge together events that are separated by less than gap_between_events seconds.
        """
        # Parameter handling unchanged
        if thresh is None:
            thresh = self.fly.config.interaction_threshold
        if gap_between_events is None:
            gap_between_events = self.fly.config.gap_between_events
        if event_min_length is None:
            event_min_length = self.fly.config.events_min_length

        if self.flytrack is None or self.balltrack is None:
            return []

        # Apply time range filtering at detection time
        fly_data = self.flytrack.objects[fly_idx].dataset
        ball_data = self.balltrack.objects[ball_idx].dataset

        if time_range:
            # Convert frame indices to integers for iloc
            start = int(time_range[0] * self.fly.experiment.fps)
            end = (
                int(time_range[1] * self.fly.experiment.fps) if time_range[1] else None
            )
            fly_data = fly_data.iloc[start:end]
            ball_data = ball_data.iloc[start:end]

        # Original event detection logic
        interaction_events = find_interaction_events(
            fly_data,
            ball_data,
            nodes1=["thorax"],
            nodes2=["centre"],
            threshold=thresh,
            gap_between_events=gap_between_events,
            event_min_length=event_min_length,
            fps=self.fly.experiment.fps,
        )
        return interaction_events

    @property
    def events_before_cutoff(self):
        """Number of events before cutoff."""
        if self.cutoff_reference:
            all_events = self._calculate_interactions(None)
            filtered_events = self._calculate_interactions((0, self.cutoff_reference))
            all_count = sum(
                len(events)
                for fly_dict in all_events.values()
                for events in fly_dict.values()
            )
            filtered_count = sum(
                len(events)
                for fly_dict in filtered_events.values()
                for events in fly_dict.values()
            )
            return filtered_count, all_count
        return None, None

    @property
    def interactions_onsets(self):
        """
        For each interaction event, get the onset of the fly interaction with the ball.
        """
        if not hasattr(self, "_interactions_onsets"):
            self._interactions_onsets = self._calculate_interactions_onsets()
        return self._interactions_onsets

    def _calculate_interactions_onsets(self):
        """Calculate interaction onsets."""
        if self.flytrack is None or self.balltrack is None:
            print(
                f"Skipping interaction events for {self.fly.metadata.name} due to missing tracking data."
            )
            return {}

        interactions_onsets = {}
        event_count = 0

        for fly_idx in range(0, len(self.flytrack.objects)):
            fly_data = self.flytrack.objects[fly_idx].dataset

            for ball_idx in range(0, len(self.balltrack.objects)):
                ball_data = self.balltrack.objects[ball_idx].dataset

                # Access events through the property to ensure they're calculated
                interaction_events = self.interaction_events[fly_idx][ball_idx]
                event_count += len(interaction_events)

                onsets = []
                for event in interaction_events:
                    event_data = fly_data.loc[event[0] : event[1]]
                    event_data["adjusted_frame"] = range(len(event_data))

                    event_data["distance"] = np.sqrt(
                        (event_data["x_thorax"] - ball_data["x_centre"]) ** 2
                        + (event_data["y_thorax"] - ball_data["y_centre"]) ** 2
                    )

                    onset = find_interaction_start(
                        event_data, "distance", "adjusted_frame"
                    )
                    onsets.append(onset)

                interactions_onsets[(fly_idx, ball_idx)] = onsets

        # print(f"Found {event_count} interaction events with onsets for {self.fly.metadata.name}")
        return interactions_onsets

    @property
    def standardized_interactions(self):
        """Standardized interaction events based on frames before and after onset."""
        if not hasattr(self, "_std_interactions") or (
            hasattr(self, "_prev_cutoff") and self._prev_cutoff != self.cutoff_reference
        ):
            # Make sure interaction_events exist and contain data
            if not self.interaction_events or not any(
                events
                for fly_dict in self.interaction_events.values()
                for events in fly_dict.values()
            ):
                print(f"No interaction events found for {self.fly.metadata.name}")
                self._std_interactions = {}
            else:
                self._std_interactions = self._calculate_standardized_interactions()
            self._prev_cutoff = self.cutoff_reference
        return self._std_interactions

    def _calculate_standardized_interactions(self):
        standardized = {}

        # Check if onsets exist
        if not self.interactions_onsets:
            print(f"No interaction onsets found for {self.fly.metadata.name}")
            return {}

        for (fly_idx, ball_idx), onsets in self.interactions_onsets.items():
            events = []
            for onset in onsets:
                if onset is None:
                    continue
                start = max(0, onset - self.fly.config.frames_before_onset)
                end = min(
                    len(self.flytrack.objects[fly_idx].dataset),
                    onset + self.fly.config.frames_after_onset,
                )
                events.append((start, end))

            # Add deduplication here
            unique_events = []
            for event in events:
                if event not in unique_events:
                    unique_events.append(event)

            standardized[(fly_idx, ball_idx)] = unique_events

        return standardized

    def detect_trials(self):
        """
        Detect trials for learning experiments.
        """
        if self.fly.config.experiment_type != "Learning":
            return None

        # This will call the LearningMetrics class methods
        return self.fly.learning_metrics.trials_data

    def get_trial_events(self, trial_number):
        """
        Get interaction events for a specific trial.

        Args:
            trial_number (int): The trial number.

        Returns:
            dict: A dictionary of events for the specified trial.
        """
        if self.fly.config.experiment_type != "Learning" or not hasattr(
            self.fly, "learning_metrics"
        ):
            return {}

        trial_interactions = self.fly.learning_metrics.metrics.get(
            "trial_interactions", {}
        )
        return trial_interactions.get(trial_number, [])

    def get_trial_standardized_interactions(self, trial_number):
        """
        Get standardized interaction events for a specific trial.

        Args:
            trial_number (int): The trial number.

        Returns:
            dict: A dictionary of standardized events for the specified trial.
        """
        if self.fly.config.experiment_type != "Learning" or not hasattr(
            self.fly, "learning_metrics"
        ):
            return {}

        # Filter standardized interactions by trial
        trial_events = self.get_trial_events(trial_number)

        # Create a set of event bounds for quick lookup
        event_bounds = {(event[0], event[1]) for event in trial_events}

        # Filter standardized interactions
        trial_std_interactions = {}

        for (fly_idx, ball_idx), events in self.standardized_interactions.items():
            matching_events = [
                event for event in events if (event[0], event[1]) in event_bounds
            ]

            if matching_events:
                trial_std_interactions[(fly_idx, ball_idx)] = matching_events

        return trial_std_interactions

    def check_data_quality(self):
        """Check if the fly is dead or in poor condition.

        This method loads the smoothed fly tracking data and checks if the fly moved more than 30 pixels in the y or x direction. If it did, it means the fly is alive and in good condition.

        Returns:
            bool: True if the fly is dead or in poor condition, False otherwise.
        """
        # Ensure that flytrack is not None
        if self.flytrack is None:
            print(f"{self.fly.metadata.name} has no tracking data.")
            return False

        # Use the flytrack dataset
        fly_data = self.flytrack.objects[0].dataset

        # Check if any of the smoothed fly x and y coordinates are more than 30 pixels away from their initial position
        moved_y = np.any(
            abs(fly_data["y_thorax"] - fly_data["y_thorax"].iloc[0])
            > self.fly.config.dead_threshold
        )
        moved_x = np.any(
            abs(fly_data["x_thorax"] - fly_data["x_thorax"].iloc[0])
            > self.fly.config.dead_threshold
        )

        if not moved_y and not moved_x:
            print(f"{self.fly.metadata.name} did not move significantly.")
            return False

        # Check if the interaction events dictionary is empty
        if not self.interaction_events or not any(self.interaction_events.values()):
            print(f"{self.fly.metadata.name} did not interact with the ball.")

            if not self.keep_idle:
                return False

        return True

    def check_dying(self):
        # Check if in the fly tracking data, there is any time where the fly doesn't move more than 30 pixels for 15 min

        if self.flytrack is None:
            return False

        fly_data = self.flytrack.objects[0].dataset

        # Get the velocity of the fly
        velocity = np.sqrt(
            np.diff(fly_data["x_thorax"], prepend=np.nan) ** 2
            + np.diff(fly_data["y_thorax"], prepend=np.nan) ** 2
        )

        # Ensure the length of the velocity array matches the length of the DataFrame index
        if len(velocity) != len(fly_data):
            velocity = np.append(velocity, np.nan)

        fly_data["velocity"] = velocity

        # Check if the fly has a continuous period of 15 min where it doesn't move more than 30 pixels

        # Get the time points where the fly's velocity is less than 2 px/s

        low_velocity = fly_data[fly_data["velocity"] < 2]

        # Get consecutive time points where the fly's velocity is less than 2 px/s

        consecutive_points = np.split(
            low_velocity, np.where(np.diff(low_velocity.index) != 1)[0] + 1
        )

        # Get the duration of each consecutive period

        durations = [len(group) for group in consecutive_points]

        # Check if there is any consecutive period of 15 min where the fly's velocity is less than 2 px/s

        for events in durations:
            if events > 15 * 60 * self.fly.experiment.fps:
                # Get the corresponding time

                time = fly_data.loc[
                    consecutive_points[durations.index(events)].index[0]
                ]["time"]

                print(f"Warning: {self.fly.metadata.name} is inactive at {time}")

                return True

    def get_initial_position(self):
        """
        Get the initial x and y positions of the fly. First, try to use the fly tracking data.
        If not available, use the skeleton data.

        Returns:
            tuple: The initial x and y positions of the fly.
        """
        # Check if fly tracking data is available
        if hasattr(self, "flytrack") and self.flytrack is not None:
            fly_data = self.flytrack.objects[0].dataset
            if "y_thorax" in fly_data.columns and "x_thorax" in fly_data.columns:
                return fly_data["x_thorax"].iloc[0], fly_data["y_thorax"].iloc[0]
            elif "y_thorax" in fly_data.columns and "x_thorax" in fly_data.columns:
                return fly_data["x_thorax"].iloc[0], fly_data["y_thorax"].iloc[0]

        # Fallback to skeleton data if fly tracking data is not available
        if self.fly_skeleton is not None:
            if (
                "y_thorax" in self.fly_skeleton.columns
                and "x_thorax" in self.fly_skeleton.columns
            ):
                return (
                    self.fly_skeleton["x_thorax"].iloc[0],
                    self.fly_skeleton["y_thorax"].iloc[0],
                )
            elif (
                "y_thorax" in self.fly_skeleton.columns
                and "x_thorax" in self.fly_skeleton.columns
            ):
                return (
                    self.fly_skeleton["x_thorax"].iloc[0],
                    self.fly_skeleton["y_thorax"].iloc[0],
                )

        raise ValueError(f"No valid position data found for {self.fly.metadata.name}.")

    def get_skeleton(self):
        """
        Extracts the coordinates of the fly's skeleton from the full body tracking data.

        Returns:
            DataFrame: A DataFrame containing the coordinates of the fly's skeleton.
        """

        if self.skeletontrack is None:
            warnings.warn(
                f"No skeleton tracking file found for {self.fly.metadata.name}."
            )
            return None

        # Get the first track
        full_body_data = self.skeletontrack.objects[0].dataset

        return full_body_data

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

        exit_time = x[exit_condition].index[0] / self.fly.experiment.fps

        return exit_time

    def compute_adjusted_time(self):
        """
        Compute adjusted time based on the fly's exit time if any, otherwise return NaN.
        """
        if self.exit_time is not None:

            flydata = self.flytrack.objects[0].dataset

            flydata["adjusted_time"] = flydata["time"] - self.exit_time

            return flydata["adjusted_time"]
        else:
            return None

    def filter_tracking_data(self, time_range):
        """Filter the tracking data based on the time range."""
        if self.flytrack is not None:
            self.flytrack.filter_data(time_range)
        if self.balltrack is not None:
            self.balltrack.filter_data(time_range)
        if self.skeletontrack is not None:
            self.skeletontrack.filter_data(time_range)

    def log_missing_fly(self):
        """Log the metadata of flies that do not pass the validity test."""
        log_path = self.fly.config.log_path

        # Get the metadata from the fly.FlyMetadata and write it to a log file

        # Get the fly's metadata
        name = self.fly.metadata.name
        metadata = self.fly.metadata.get_arena_metadata()

        # Write the metadata to a log file
        with open(f"{log_path}/missing_flies.log", "a") as f:
            f.write(f"{name}: {metadata}\n")


# TODO : Test the valid_data function in conditions where I know the fly is dead or the arena is empty or not to check success


class BallpushingMetrics:
    def __init__(self, tracking_data):

        self.tracking_data = tracking_data
        self.fly = tracking_data.fly
        self.metrics = {}
        self.compute_metrics()
        # TODO: Compute maximum distance pushed (corresponding to max_event)

    def compute_metrics(self):
        """
        Compute and store various metrics for each pair of fly and ball.
        """
        for fly_idx, ball_dict in self.tracking_data.interaction_events.items():
            for ball_idx, events in ball_dict.items():
                key = f"fly_{fly_idx}_ball_{ball_idx}"

                try:
                    nb_events = self.get_adjusted_nb_events(
                        fly_idx, ball_idx, signif=False
                    )
                except Exception as e:
                    nb_events = np.nan

                try:
                    max_event = self.get_max_event(fly_idx, ball_idx)
                except Exception as e:
                    max_event = (np.nan, np.nan)

                try:
                    max_distance = self.get_max_distance(fly_idx, ball_idx)
                except Exception as e:
                    max_distance = np.nan

                try:
                    significant_events = self.get_significant_events(fly_idx, ball_idx)
                except Exception as e:
                    significant_events = []

                try:
                    if self.fly.config.experiment_type == "F1":
                        nb_significant_events = self.get_adjusted_nb_events(
                            fly_idx, ball_idx, signif=True
                        )
                    else:
                        nb_significant_events = len(significant_events)
                except Exception as e:
                    nb_significant_events = np.nan

                try:
                    first_significant_event = self.get_first_significant_event(
                        fly_idx, ball_idx
                    )
                except Exception as e:
                    first_significant_event = (np.nan, np.nan)

                try:
                    aha_moment = self.get_aha_moment(fly_idx, ball_idx)
                except Exception as e:
                    aha_moment = (np.nan, np.nan)

                try:
                    breaks = self.find_breaks(fly_idx, ball_idx)
                except Exception as e:
                    breaks = []

                try:
                    events_direction = self.find_events_direction(fly_idx, ball_idx)
                except Exception as e:
                    events_direction = ([], [])

                try:
                    final_event = self.get_final_event(fly_idx, ball_idx)
                except Exception as e:
                    final_event = (np.nan, np.nan)

                try:
                    success_direction = self.get_success_direction(fly_idx, ball_idx)
                except Exception as e:
                    success_direction = np.nan

                try:
                    cumulated_breaks_duration = self.get_cumulated_breaks_duration(
                        fly_idx, ball_idx
                    )
                except Exception as e:
                    cumulated_breaks_duration = np.nan

                try:
                    distance_moved = self.get_distance_moved(fly_idx, ball_idx)
                except Exception as e:
                    distance_moved = np.nan

                try:
                    insight_effect = (
                        self.get_insight_effect(fly_idx, ball_idx)
                        if aha_moment
                        else np.nan
                    )
                except Exception as e:
                    insight_effect = np.nan

                self.metrics[key] = {
                    "nb_events": nb_events,
                    "max_event": max_event[0],
                    "max_event_time": max_event[1],
                    "max_distance": max_distance,
                    "final_event": final_event[0],
                    "final_event_time": final_event[1],
                    "nb_significant_events": nb_significant_events,
                    "significant_ratio": (
                        nb_significant_events / nb_events if nb_events > 0 else np.nan
                    ),
                    "first_significant_event": first_significant_event[0],
                    "first_significant_event_time": first_significant_event[1],
                    "aha_moment": aha_moment[0],
                    "aha_moment_time": aha_moment[1],
                    "aha_moment_first": insight_effect["first_event"],
                    "insight_effect": insight_effect["raw_effect"],
                    "insight_effect_log": insight_effect["log_effect"],
                    "cumulated_breaks_duration": cumulated_breaks_duration,
                    "pushed": len(events_direction[0]),
                    "pulled": len(events_direction[1]),
                    "pulling_ratio": (
                        len(events_direction[1])
                        / (len(events_direction[0]) + len(events_direction[1]))
                        if (len(events_direction[0]) + len(events_direction[1])) > 0
                        else np.nan
                    ),
                    "success_direction": success_direction,
                    "interaction_proportion": (
                        sum([event[2] for event in events])
                        / (
                            sum([event[2] for event in events])
                            + cumulated_breaks_duration
                        )
                        if cumulated_breaks_duration > 0
                        else np.nan
                    ),
                    "distance_moved": distance_moved,
                    "exit_time": self.tracking_data.exit_time,
                }

    def get_adjusted_nb_events(self, fly_idx, ball_idx, signif=False):
        """
        Calculate the adjusted number of events for a given fly and ball. adjustment is based on the duration of the experiment.

        Parameters
        ----------
        fly_idx : int
            Index of the fly.
        ball_idx : int
            Index of the ball.
        signif : bool, optional
            Whether to use significant events only (default is False).

        Returns
        -------
        float
            Adjusted number of events.
        """
        if signif:
            events = self.get_significant_events(fly_idx, ball_idx)
        else:
            events = self.tracking_data.interaction_events[fly_idx][ball_idx]

        adjusted_nb_events = 0  # Initialize to 0 in case there are no events

        key = f"fly_{fly_idx}_ball_{ball_idx}"

        if hasattr(self.fly.metadata, "F1_condition"):

            if self.fly.metadata.F1_condition == "control":
                if ball_idx == 0 and self.tracking_data.exit_time is not None:
                    adjusted_nb_events = (
                        len(events)
                        * self.fly.config.adjusted_events_normalisation
                        / (self.tracking_data.duration - self.tracking_data.exit_time)
                        if self.tracking_data.duration - self.tracking_data.exit_time
                        > 0
                        else 0
                    )
            else:
                if ball_idx == 1 and self.tracking_data.exit_time is not None:
                    adjusted_nb_events = (
                        len(events)
                        * self.fly.config.adjusted_events_normalisation
                        / (self.tracking_data.duration - self.tracking_data.exit_time)
                        if self.tracking_data.duration - self.tracking_data.exit_time
                        > 0
                        else 0
                    )
                elif ball_idx == 0:
                    adjusted_nb_events = (
                        len(events)
                        * self.fly.config.adjusted_events_normalisation
                        / self.tracking_data.exit_time
                        if (
                            self.tracking_data.exit_time
                            and self.tracking_data.exit_time > 0
                        )
                        else len(events)
                        * self.fly.config.adjusted_events_normalisation
                        / self.tracking_data.duration
                    )

        else:
            adjusted_nb_events = (
                len(events)
                * self.fly.config.adjusted_events_normalisation
                / self.tracking_data.duration
                if self.tracking_data.duration > 0
                else 0
            )

        return adjusted_nb_events

    def find_event_by_distance(self, fly_idx, ball_idx, threshold, distance_type="max"):
        """
        Find the event at which the ball has been moved a given amount of pixels for a given fly and ball.
        Threshold is the distance threshold to check, whereas max is the maximum distance reached by the ball for this particular fly and ball.

        Parameters
        ----------
        fly_idx : int
            Index of the fly.
        ball_idx : int
            Index of the ball.
        threshold : float
            Distance threshold.
        distance_type : str, optional
            Type of distance to check ("max" or "threshold", default is "max").

        Returns
        -------
        tuple
            Event and event index.
        """
        ball_data = self.tracking_data.balltrack.objects[ball_idx].dataset

        # ball_data["euclidean_distance"] = np.sqrt(
        #     (ball_data["x_centre"] - ball_data["x_centre"].iloc[0]) ** 2
        #     + (ball_data["y_centre"] - ball_data["y_centre"].iloc[0]) ** 2
        # )

        if distance_type == "max":
            max_distance = ball_data["euclidean_distance"].max() - threshold
            distance_check = (
                lambda event: ball_data.loc[
                    event[0] : event[1], "euclidean_distance"
                ].max()
                >= max_distance
            )
        elif distance_type == "threshold":
            distance_check = (
                lambda event: ball_data.loc[
                    event[0] : event[1], "euclidean_distance"
                ].max()
                >= threshold
            )
        else:
            raise ValueError("Invalid distance_type. Use 'max' or 'threshold'.")

        try:
            event, event_index = next(
                (event, i)
                for i, event in enumerate(
                    self.tracking_data.interaction_events[fly_idx][ball_idx]
                )
                if distance_check(event)
            )
        except StopIteration:
            event, event_index = None, None

        return event, event_index

    def get_max_event(self, fly_idx, ball_idx, threshold=None):
        """
        Get the event at which the ball was moved at its maximum distance for a given fly and ball. Maximum here doesn't mean the ball has reached the end of the corridor.

        Parameters
        ----------
        fly_idx : int
            Index of the fly.
        ball_idx : int
            Index of the ball.
        threshold : float, optional
            Distance threshold (default is None).

        Returns
        -------
        tuple
            Maximum event index and maximum event time.
        """
        if threshold is None:
            threshold = self.fly.config.max_event_threshold

        ball_data = self.tracking_data.balltrack.objects[ball_idx].dataset

        max_event, max_event_idx = self.find_event_by_distance(
            fly_idx, ball_idx, threshold, distance_type="max"
        )

        if abs(ball_data["x_centre"].iloc[0] - self.tracking_data.start_x) < 100:
            max_event_time = (
                max_event[0] / self.fly.experiment.fps if max_event else None
            )
        else:
            max_event_time = (
                (max_event[0] / self.fly.experiment.fps) - self.tracking_data.exit_time
                if max_event
                else None
            )

        return max_event_idx, max_event_time

    def get_max_distance(self, fly_idx, ball_idx):
        """
        Get the maximum distance moved by the ball for a given fly and ball.

        Parameters
        ----------
        fly_idx : int
            Index of the fly.
        ball_idx : int
            Index of the ball.

        Returns
        -------
        float
            Maximum distance moved by the ball.
        """
        ball_data = self.tracking_data.balltrack.objects[ball_idx].dataset

        max_distance = np.sqrt(
            (ball_data["x_centre"] - ball_data["x_centre"].iloc[0]) ** 2
            + (ball_data["y_centre"] - ball_data["y_centre"].iloc[0]) ** 2
        ).max()

        return max_distance

    def get_final_event(self, fly_idx, ball_idx, threshold=None, init=False):

        ball_data = self.tracking_data.balltrack.objects[ball_idx].dataset

        if abs(ball_data["x_centre"].iloc[0] - self.tracking_data.start_x) < 100:
            threshold = self.fly.config.final_event_threshold

            final_event, final_event_idx = self.find_event_by_distance(
                fly_idx, ball_idx, threshold, distance_type="threshold"
            )

            final_event_time = (
                final_event[0] / self.fly.experiment.fps if final_event else None
            )

        else:
            threshold = self.fly.config.final_event_F1_threshold

            final_event, final_event_idx = self.find_event_by_distance(
                fly_idx, ball_idx, threshold, distance_type="threshold"
            )

            final_event_time = (
                (final_event[0] / self.fly.experiment.fps)
                - self.tracking_data.exit_time
                if final_event
                else None
            )

        return final_event_idx, final_event_time

    def get_significant_events(self, fly_idx, ball_idx, distance=5):
        ball_data = self.tracking_data.balltrack.objects[ball_idx].dataset

        significant_events = [
            (event, i)
            for i, event in enumerate(
                self.tracking_data.interaction_events[fly_idx][ball_idx]
            )
            if self.check_yball_variation(event, ball_data, threshold=distance)
        ]

        return significant_events

    def get_first_significant_event(self, fly_idx, ball_idx, distance=5):
        ball_data = self.tracking_data.balltrack.objects[ball_idx].dataset

        significant_events = self.get_significant_events(
            fly_idx, ball_idx, distance=distance
        )

        if significant_events:
            first_significant_event = significant_events[0]
            first_significant_event_idx = first_significant_event[1]

            if abs(ball_data["x_centre"].iloc[0] - self.tracking_data.start_x) < 100:
                first_significant_event_time = (
                    first_significant_event[0][0] / self.fly.experiment.fps
                )
            else:
                first_significant_event_time = (
                    first_significant_event[0][0] / self.fly.experiment.fps
                ) - self.tracking_data.exit_time

            return first_significant_event_idx, first_significant_event_time
        else:
            return None, None

    def check_yball_variation(self, event, ball_data, threshold=None):

        if threshold is None:
            threshold = self.fly.config.significant_threshold

        yball_event = ball_data.loc[event[0] : event[1], "y_centre"]
        variation = yball_event.max() - yball_event.min()
        return variation > threshold

    def find_breaks(self, fly_idx, ball_idx):
        ball_data = self.tracking_data.balltrack.objects[ball_idx].dataset

        breaks = []
        if not self.tracking_data.interaction_events[fly_idx][ball_idx]:
            breaks.append((0, len(ball_data), len(ball_data)))
            return breaks

        if self.tracking_data.interaction_events[fly_idx][ball_idx][0][0] > 0:
            breaks.append(
                (
                    0,
                    self.tracking_data.interaction_events[fly_idx][ball_idx][0][0],
                    self.tracking_data.interaction_events[fly_idx][ball_idx][0][0],
                )
            )

        for i, event in enumerate(
            self.tracking_data.interaction_events[fly_idx][ball_idx][:-1]
        ):
            start = event[1]
            end = self.tracking_data.interaction_events[fly_idx][ball_idx][i + 1][0]
            duration = end - start
            breaks.append((start, end, duration))

        if self.tracking_data.interaction_events[fly_idx][ball_idx][-1][1] < len(
            ball_data
        ):
            breaks.append(
                (
                    self.tracking_data.interaction_events[fly_idx][ball_idx][-1][1],
                    len(ball_data),
                    len(ball_data)
                    - self.tracking_data.interaction_events[fly_idx][ball_idx][-1][1],
                )
            )

        return breaks

    def get_cumulated_breaks_duration(self, fly_idx, ball_idx):
        breaks = self.find_breaks(fly_idx, ball_idx)
        cumulated_breaks_duration = sum([break_[2] for break_ in breaks])
        return cumulated_breaks_duration

    def find_events_direction(self, fly_idx, ball_idx):
        fly_data = self.tracking_data.flytrack.objects[fly_idx].dataset
        ball_data = self.tracking_data.balltrack.objects[ball_idx].dataset

        significant_events = self.get_significant_events(fly_idx, ball_idx)

        pushing_events = []
        pulling_events = []

        for event in significant_events:
            event = event[0]

            start_roi = event[0]
            end_roi = event[1]

            start_distance = np.sqrt(
                (
                    ball_data.loc[start_roi, "x_centre"]
                    - fly_data.loc[start_roi, "x_thorax"]
                )
                ** 2
                + (
                    ball_data.loc[start_roi, "y_centre"]
                    - fly_data.loc[start_roi, "y_thorax"]
                )
                ** 2
            )
            end_distance = np.sqrt(
                (
                    ball_data.loc[end_roi, "x_centre"]
                    - fly_data.loc[start_roi, "x_thorax"]
                )
                ** 2
                + (
                    ball_data.loc[end_roi, "y_centre"]
                    - fly_data.loc[start_roi, "y_thorax"]
                )
                ** 2
            )

            if end_distance > start_distance:
                pushing_events.append(event)
            else:
                pulling_events.append(event)

        return pushing_events, pulling_events

    def get_distance_moved(self, fly_idx, ball_idx, subset=None):
        ball_data = self.tracking_data.balltrack.objects[ball_idx].dataset

        if subset is None:
            subset = self.tracking_data.interaction_events[fly_idx][ball_idx]

        for event in subset:
            ball_data.loc[event[0] : event[1], "euclidean_distance"] = np.sqrt(
                (
                    ball_data["x_centre"].iloc[event[1]]
                    - ball_data["x_centre"].iloc[event[0]]
                )
                ** 2
                + (
                    ball_data["y_centre"].iloc[event[1]]
                    - ball_data["y_centre"].iloc[event[0]]
                )
                ** 2
            )

        return ball_data["euclidean_distance"].sum()

    def get_aha_moment(self, fly_idx, ball_idx, distance=None):
        """
        Identify the aha moment for a given fly and ball.

        Parameters
        ----------
        fly_idx : int
            Index of the fly.
        ball_idx : int
            Index of the ball.
        distance : float, optional
            Distance threshold for the aha moment (default is None).

        Returns
        -------
        tuple
            Aha moment index and aha moment time.
        """
        if distance is None:
            distance = self.fly.config.aha_moment_threshold

        ball_data = self.tracking_data.balltrack.objects[ball_idx].dataset

        aha_moment = [
            (event, i)
            for i, event in enumerate(
                self.tracking_data.interaction_events[fly_idx][ball_idx]
            )
            if self.check_yball_variation(event, ball_data, threshold=distance)
        ]

        if aha_moment:
            # Select the event right before the event at which the ball was moved more than the threshold
            aha_moment_event, aha_moment_idx = aha_moment[0]
            if aha_moment_idx > 0:
                previous_event = self.tracking_data.interaction_events[fly_idx][
                    ball_idx
                ][aha_moment_idx - 1]
                aha_moment_event = previous_event
                aha_moment_idx -= 1

            if abs(ball_data["x_centre"].iloc[0] - self.tracking_data.start_x) < 100:
                aha_moment_time = aha_moment_event[0] / self.fly.experiment.fps
            else:
                aha_moment_time = (
                    aha_moment_event[0] / self.fly.experiment.fps
                ) - self.tracking_data.exit_time

            return aha_moment_idx, aha_moment_time
        else:
            return None, None

    def get_insight_effect(self, fly_idx, ball_idx, epsilon=1e-6, strength_threshold=2):
        """
        Calculate enhanced insight effect with performance analytics.

        Parameters
        ----------
        fly_idx : int
            Index of the fly.
        ball_idx : int
            Index of the ball.
        epsilon : float, optional
            Smoothing factor to prevent division by zero.
        strength_threshold : float, optional
            Threshold for strong/weak classification.

        Returns
        -------
        dict
            Dictionary containing multiple insight metrics:
            - raw_effect: Base ratio of post/pre aha performance
            - log_effect: Log-transformed effect for normal distribution
            - classification: Strong/weak based on threshold
            - first_event: Flag for aha moment as first interaction
            - post_aha_count: Number of post-aha events
        """
        significant_events = [
            event[0] for event in self.get_significant_events(fly_idx, ball_idx)
        ]
        aha_moment_index, _ = self.get_aha_moment(fly_idx, ball_idx)

        # Handle no significant events case early
        if not significant_events:
            return {
                "raw_effect": np.nan,
                "log_effect": np.nan,
                "classification": "none",
                "first_event": False,
                "post_aha_count": 0,
            }

        # Handle no aha moment case early
        if aha_moment_index is None:
            return {
                "raw_effect": np.nan,
                "log_effect": np.nan,
                "classification": "none",
                "first_event": False,
                "post_aha_count": 0,
            }

        # Segment events with aha moment in before period
        before_aha = significant_events[: aha_moment_index + 1]
        after_aha = significant_events[aha_moment_index + 1 :]

        # Calculate average distances with safety checks
        avg_before = self._calculate_avg_distance(fly_idx, ball_idx, before_aha)
        avg_after = self._calculate_avg_distance(fly_idx, ball_idx, after_aha)

        # Core insight calculation
        if aha_moment_index == 0:
            insight_effect = 1.0
        elif avg_before == 0:
            insight_effect = np.nan
        else:
            insight_effect = avg_after / avg_before

        # Transformations and classifications
        log_effect = np.log(insight_effect + 1) if insight_effect > 0 else np.nan
        classification = "strong" if insight_effect > strength_threshold else "weak"

        return {
            "raw_effect": insight_effect,
            "log_effect": log_effect,
            "classification": classification,
            "first_event": (aha_moment_index == 0),
            "post_aha_count": len(after_aha),
        }

    def _calculate_avg_distance(self, fly_idx, ball_idx, events):
        """Helper method to safely calculate average distances"""
        if not events:
            return np.nan

        try:
            distances = self.get_distance_moved(fly_idx, ball_idx, subset=events)
            return np.mean(distances) / len(events)
        except (ValueError, ZeroDivisionError):
            return np.nan

    def get_success_direction(self, fly_idx, ball_idx, threshold=None):

        if threshold is None:
            threshold = self.fly.config.success_direction_threshold

        ball_data = self.tracking_data.balltrack.objects[ball_idx].dataset

        initial_y = ball_data["y_centre"].iloc[0]

        ball_data["euclidean_distance"] = np.sqrt(
            (ball_data["x_centre"] - ball_data["x_centre"].iloc[0]) ** 2
            + (ball_data["y_centre"] - ball_data["y_centre"].iloc[0]) ** 2
        )

        moved_data = ball_data[ball_data["euclidean_distance"] >= threshold]

        if moved_data.empty:
            return None

        if abs(ball_data["x_centre"].iloc[0] - self.tracking_data.start_x) < 100:
            pushed = any(moved_data["y_centre"] < initial_y)
            pulled = any(moved_data["y_centre"] > initial_y)
        else:
            pushed = any(moved_data["y_centre"] > initial_y)
            pulled = any(moved_data["y_centre"] < initial_y)

        if pushed and pulled:
            return "both"
        elif pushed:
            return "push"
        elif pulled:
            return "pull"
        else:
            return None

    def get_time_chamber(self):
        """
        Get the time spent by the fly in the chamber, meaning within a 50 px radius of the fly start position.

        Returns
        -------
        float
            Time spent in the chamber in seconds.
        """
        # Get the tracking data for the fly
        tracking_data = self.tracking_data

        # Determine the start position by averaging the first 10-20 frames
        num_frames_to_average = 20
        start_position_x = np.mean(tracking_data.x[:num_frames_to_average])
        start_position_y = np.mean(tracking_data.y[:num_frames_to_average])

        # Calculate the distance from the start position for each frame
        distances = np.sqrt(
            (tracking_data.x - start_position_x) ** 2
            + (tracking_data.y - start_position_y) ** 2
        )

        # Determine the frames where the fly is within a 50 px radius of the start position
        in_chamber = distances <= 50

        # Calculate the time spent in the chamber
        time_in_chamber = np.sum(in_chamber) / self.fly.experiment.fps

        return time_in_chamber


class F1Metrics:
    def __init__(self, tracking_data):
        self.tracking_data = tracking_data
        self.fly = tracking_data.fly
        self.metrics = {}
        self.compute_metrics()

    def compute_metrics(self):

        self.compute_adjusted_time = self.compute_adjusted_time()

        self.training_ball_distances, self.test_ball_distances = (
            self.get_F1_ball_distances()
        )

        self.F1_checkpoints = self.find_checkpoint_times()

        self.direction_match = self.get_direction_match()

        self.metrics = {
            "adjusted_time": self.compute_adjusted_time,
            "training_ball_distances": self.training_ball_distances,
            "test_ball_distances": self.test_ball_distances,
            "F1_checkpoints": self.F1_checkpoints,
            "direction_match": self.direction_match,
        }

    def compute_adjusted_time(self):
        """
        Compute adjusted time based on the fly's exit time if any, otherwise return NaN.
        """
        if self.tracking_data.exit_time is not None:
            flydata = self.tracking_data.flytrack.objects[0].dataset
            flydata["adjusted_time"] = flydata["time"] - self.tracking_data.exit_time
            return flydata["adjusted_time"]
        else:
            return None

    def get_F1_ball_distances(self):
        """
        Compute the Euclidean distances for the training and test ball data.

        Returns:
            tuple: The training and test ball data with Euclidean distances.
        """
        training_ball_data = None
        test_ball_data = None

        for ball_idx in range(0, len(self.tracking_data.balltrack.objects)):
            ball_data = self.tracking_data.balltrack.objects[ball_idx].dataset

            # Check if the ball is > or < 100 px away from the fly's initial x position
            if abs(ball_data["x_centre"].iloc[0] - self.tracking_data.start_x) < 100:
                training_ball_data = ball_data
                training_ball_data["euclidean_distance"] = np.sqrt(
                    (
                        training_ball_data["x_centre"]
                        - training_ball_data["x_centre"].iloc[0]
                    )
                    ** 2
                    + (
                        training_ball_data["y_centre"]
                        - training_ball_data["y_centre"].iloc[0]
                    )
                    ** 2
                )
            else:
                test_ball_data = ball_data
                test_ball_data["euclidean_distance"] = np.sqrt(
                    (test_ball_data["x_centre"] - test_ball_data["x_centre"].iloc[0])
                    ** 2
                    + (test_ball_data["y_centre"] - test_ball_data["y_centre"].iloc[0])
                    ** 2
                )

        return training_ball_data, test_ball_data

    def find_checkpoint_times(self, distances=[10, 25, 35, 50, 60, 75, 90, 100]):
        """
        Find the times at which the ball reaches certain distances from its initial position.
        """
        _, test_ball_data = self.get_F1_ball_distances()

        checkpoint_times = {}

        for distance in distances:
            try:
                # Find the time at which the test ball reaches the distance
                checkpoint_time = test_ball_data.loc[
                    test_ball_data["euclidean_distance"] >= distance, "time"
                ].iloc[0]
                # Adjust the time by subtracting self.tracking_data.exit_time
                adjusted_time = checkpoint_time - self.tracking_data.exit_time
            except IndexError:
                # If the distance is not reached, set the checkpoint time to None
                adjusted_time = None

            checkpoint_times[f"{distance}"] = adjusted_time

        return checkpoint_times

    def get_direction_match(self):
        """
        For each of the flyball pair, check the success_direction metric and compare it with the success_direction of the other flyball pair.
        """

        success_directions = {}

        for fly_idx, ball_dict in self.tracking_data.interaction_events.items():
            for ball_idx, events in ball_dict.items():
                key = f"fly_{fly_idx}_ball_{ball_idx}"
                success_direction = self.fly.events_metrics[key]["success_direction"]
                success_directions[key] = success_direction

        if len(self.tracking_data.balltrack.objects) > 1:
            direction_1 = success_directions["fly_0_ball_0"]
            direction_2 = success_directions["fly_0_ball_1"]

            if direction_1 == direction_2:
                return "match"
            elif direction_1 == "both" or direction_2 == "both":
                return "partial_match"
            else:
                return "different"
        else:
            return None


class LearningMetrics:
    def __init__(self, tracking_data):
        self.tracking_data = tracking_data
        self.fly = tracking_data.fly
        self.metrics = {}
        self.trials_data = None
        self.trial_durations = None

        # Detect trials and compute metrics
        self.detect_trials()
        self.compute_metrics()

    def detect_trials(self):
        """
        Detect individual trials based on negative peaks in the ball position derivative.
        """
        # Get ball data
        ball_data = self.tracking_data.balltrack.objects[0].dataset.copy()

        # Compute the derivative of the ball position using the standard column
        ball_data["position_derivative"] = ball_data["euclidean_distance"].diff()

        # Find negative peaks (indicating ball reset)
        peaks, properties = find_peaks(
            -ball_data["position_derivative"],
            height=self.fly.config.trial_peak_height,
            distance=self.fly.config.trial_peak_distance,
        )

        # Assign trial numbers
        ball_data["trial"] = 0
        trial_number = 1
        previous_peak = 0

        for peak in peaks:
            ball_data.iloc[
                previous_peak : peak + 1, ball_data.columns.get_loc("trial")
            ] = trial_number
            trial_number += 1
            previous_peak = peak + 1

        # Assign last trial
        ball_data.iloc[previous_peak:, ball_data.columns.get_loc("trial")] = (
            trial_number
        )

        # Add trial_frame and trial_time columns
        ball_data["trial_frame"] = ball_data.groupby("trial").cumcount()
        ball_data["trial_time"] = ball_data["trial_frame"] / self.fly.experiment.fps

        # Store the data with trial information
        self.trials_data = ball_data

        # Clean trials (skip initial frames, trim at max position)
        self.clean_trials()

        # Compute trial durations
        self.compute_trial_durations()

        return self.trials_data

    def clean_trials(self):
        """
        Clean trial data by removing initial frames and trimming at maximum position.
        """
        cleaned_data = []

        for trial in self.trials_data["trial"].unique():
            # Get data for this trial
            trial_data = self.trials_data[self.trials_data["trial"] == trial].copy()

            # Skip initial frames
            if len(trial_data) > self.fly.config.trial_skip_frames:
                trial_data = trial_data.iloc[self.fly.config.trial_skip_frames :]

            # Find max position and trim
            max_index = trial_data["euclidean_distance"].idxmax()
            trial_data = trial_data.loc[:max_index]

            cleaned_data.append(trial_data)

        if cleaned_data:
            self.trials_data = pd.concat(cleaned_data).reset_index(drop=True)

    def compute_trial_durations(self):
        """
        Compute the duration of each trial.
        """
        durations = []

        for trial in self.trials_data["trial"].unique():
            trial_data = self.trials_data[self.trials_data["trial"] == trial]
            duration = trial_data["time"].max() - trial_data["time"].min()
            durations.append({"trial": trial, "duration": duration})

        self.trial_durations = pd.DataFrame(durations)

    def compute_metrics(self):
        """
        Compute metrics for learning experiments.
        """
        # Basic metrics about trials
        self.metrics["num_trials"] = len(self.trials_data["trial"].unique())

        if not self.trial_durations.empty:
            self.metrics["mean_trial_duration"] = self.trial_durations[
                "duration"
            ].mean()
            self.metrics["trial_durations"] = self.trial_durations["duration"].tolist()
            self.metrics["trial_numbers"] = self.trial_durations["trial"].tolist()

        # Map interaction events to trials
        self.map_interactions_to_trials()

    def map_interactions_to_trials(self):
        """
        Associate interaction events with trials.
        """
        if not hasattr(self.tracking_data, "interaction_events"):
            return

        # Create a mapping of frame to trial
        frame_to_trial = dict(zip(self.trials_data.index, self.trials_data["trial"]))

        # Map each interaction event to a trial
        trial_interactions = {}

        for fly_idx, ball_dict in self.tracking_data.interaction_events.items():
            for ball_idx, events in ball_dict.items():
                for event in events:
                    start_frame, end_frame = event[0], event[1]

                    # Find trial for the event (use start frame)
                    if start_frame in frame_to_trial:
                        trial = frame_to_trial[start_frame]

                        if trial not in trial_interactions:
                            trial_interactions[trial] = []

                        trial_interactions[trial].append(event)

        self.metrics["trial_interactions"] = trial_interactions

        # Count interactions per trial
        self.metrics["interactions_per_trial"] = {
            trial: len(events) for trial, events in trial_interactions.items()
        }


class SkeletonMetrics:
    """
    A class for computing metrics from the skeleton data. It requires to have a fly object with a valid skeleton data, and check whether there is a "preprocessed" ball data available.
    """

    def __init__(self, fly):
        self.fly = fly
        self.ball = self.fly.tracking_data.balltrack
        self.preprocess_ball()

        # Find all contact events
        self.all_contacts = self.find_contact_events()

        # print(f"Number of contact events: {len(self.all_contacts)}")

        # Determine the final contact if success_cutoff is enabled
        if self.fly.config.success_cutoff:
            final_contact_idx, _ = self.get_final_contact()
            if final_contact_idx is not None:
                self.contacts = self.all_contacts[: final_contact_idx + 1]
            else:
                self.contacts = self.all_contacts
        else:
            self.contacts = self.all_contacts

        # print(f"Number of final contact events: {len(self.contacts)}")

        self.ball_displacements = self.compute_ball_displacements()

        self.fly_centered_tracks = self.compute_fly_centered_tracks()

        self.events_based_contacts = self.compute_events_based_contacts()

    def resize_coordinates(
        self, x, y, original_width, original_height, new_width, new_height
    ):
        """Resize the coordinates according to the new frame size."""
        x_scale = new_width / original_width
        y_scale = new_height / original_height
        return int(x * x_scale), int(y * y_scale)

    def apply_arena_mask_to_labels(
        self, x, y, mask_padding, crop_top, crop_bottom, new_height
    ):
        """Adjust the coordinates according to the cropping and padding applied to the frame."""
        # Crop from top and bottom
        if crop_top <= y < (new_height - crop_bottom):
            y -= crop_top
        else:
            return None, None

        # Add padding to the left and right
        x += mask_padding

        return x, y

    def resize_and_transform_coordinate(
        self,
        x,
        y,
        original_width,
        original_height,
        new_width,
        new_height,
        mask_padding,
        crop_top,
        crop_bottom,
    ):
        """Resize and transform the coordinate to match the preprocessed frame."""
        # Resize the coordinate
        x, y = self.resize_coordinates(
            x, y, original_width, original_height, new_width, new_height
        )

        # Apply cropping offset and padding
        x, y = self.apply_arena_mask_to_labels(
            x, y, mask_padding, crop_top, crop_bottom, new_height
        )

        return x, y

    def preprocess_ball(self):
        """Transform the ball coordinates to match the skeleton data."""

        ball_data = self.ball.objects[0].dataset

        ball_coords = [
            (x, y) for x, y in zip(ball_data["x_centre"], ball_data["y_centre"])
        ]

        # Apply resizing, cropping, and padding to the ball tracking data
        ball_coords = [
            self.resize_and_transform_coordinate(
                x,
                y,
                self.fly.metadata.original_size[0],
                self.fly.metadata.original_size[1],
                self.fly.config.template_width,
                self.fly.config.template_height,
                self.fly.config.padding,
                self.fly.config.y_crop[0],
                self.fly.config.y_crop[1],
            )
            for x, y in ball_coords
        ]

        ball_data["x_centre_preprocessed"] = [
            x for x, y in ball_coords if x is not None and y is not None
        ]
        ball_data["y_centre_preprocessed"] = [
            y for x, y in ball_coords if x is not None and y is not None
        ]

        # Add x_centre_preprocessed and y_centre_preprocessed in the node_names

        self.ball.node_names.extend(["centre_preprocessed"])

        # print(self.ball.objects[0].dataset)
        # print(self.ball.node_names)

        self.ball

    def find_contact_events(
        self, threshold=None, gap_between_events=None, event_min_length=None
    ):
        if threshold is None:
            threshold = self.fly.experiment.config.contact_threshold
        if gap_between_events is None:
            gap_between_events = self.fly.experiment.config.gap_between_contacts
        if event_min_length is None:
            event_min_length = self.fly.experiment.config.contact_min_length

        fly_data = self.fly.tracking_data.skeletontrack.objects[0].dataset
        ball_data = self.ball.objects[0].dataset

        # Find all contact events
        contact_events = find_interaction_events(
            fly_data,
            ball_data,
            nodes1=self.fly.experiment.config.contact_nodes,
            nodes2=["centre_preprocessed"],
            threshold=threshold,
            gap_between_events=gap_between_events,
            event_min_length=event_min_length,
        )

        # print(f"Number of contact events: {len(contact_events)}")
        # print(f"Contact events: {contact_events}")

        return contact_events

    def find_contact_by_distance(self, threshold, distance_type="max"):
        ball_data = self.ball.objects[0].dataset

        # ball_data["euclidean_distance"] = np.sqrt(
        #     (
        #         ball_data["x_centre_preprocessed"]
        #         - ball_data["x_centre_preprocessed"].iloc[0]
        #     )
        #     ** 2
        #     + (
        #         ball_data["y_centre_preprocessed"]
        #         - ball_data["y_centre_preprocessed"].iloc[0]
        #     )
        #     ** 2
        # )

        if distance_type == "max":
            max_distance = ball_data["euclidean_distance"].max() - threshold
            distance_check = (
                lambda event: ball_data.loc[
                    event[0] : event[1], "euclidean_distance"
                ].max()
                >= max_distance
            )
        elif distance_type == "threshold":
            distance_check = (
                lambda event: ball_data.loc[
                    event[0] : event[1], "euclidean_distance"
                ].max()
                >= threshold
            )
        else:
            raise ValueError("Invalid distance_type. Use 'max' or 'threshold'.")

        try:
            event, event_index = next(
                (event, i)
                for i, event in enumerate(self.all_contacts)
                if distance_check(event)
            )
        except StopIteration:
            event, event_index = None, None

        return event, event_index

    def compute_events_based_contacts(self, generate_random=True):
        """
        List fly relative tracking data associated with interaction events with optional random negative examples.
        """

        generate_random = self.fly.config.generate_random

        events = []
        all_event_intervals = []

        if hasattr(self.fly.tracking_data, "standardized_interactions"):
            # Collect all interaction intervals for exclusion
            for (
                fly_idx,
                ball_idx,
            ), event_list in self.fly.tracking_data.standardized_interactions.items():
                for event in event_list:

                    start = event[0]
                    end = event[1]
                    all_event_intervals.append((start, end))

        else:
            print("No standardized interactions found")

        event_counter = 0
        if (
            hasattr(self.fly.tracking_data, "standardized_interactions")
            and self.fly.tracking_data.standardized_interactions
        ):

            for (
                fly_idx,
                ball_idx,
            ), event_list in self.fly.tracking_data.standardized_interactions.items():
                for event in event_list:
                    if len(event) != 2:
                        print(f"Skipping malformed event: {event}")
                        continue

                    start_frame = int(event[0])
                    end_frame = int(event[1])

                    # Convert to actual indices
                    start_idx = start_frame
                    end_idx = end_frame

                    # Validate indices
                    if start_idx >= len(self.fly_centered_tracks) or end_idx > len(
                        self.fly_centered_tracks
                    ):
                        print(
                            f"Invalid event bounds {start_idx}-{end_idx} for track length {len(self.fly_centered_tracks)}"
                        )
                        continue

                    # Process event data
                    event_data = self.fly_centered_tracks.iloc[start_idx:end_idx].copy()

                    event_data["event_id"] = event_counter
                    event_data["time_rel_onset"] = (
                        event_data.index - start
                    ) / self.fly.experiment.fps
                    event_data["fly_idx"] = fly_idx
                    event_data["ball_idx"] = ball_idx
                    event_data["adjusted_frame"] = range(end_idx - start_idx)
                    event_data["event_type"] = "interaction"

                    # Calculate ball displacement
                    ball_disp = np.sqrt(
                        (
                            event_data["x_centre_preprocessed"]
                            - event_data["x_centre_preprocessed"].iloc[0]
                        )
                        ** 2
                        + (
                            event_data["y_centre_preprocessed"]
                            - event_data["y_centre_preprocessed"].iloc[0]
                        )
                        ** 2
                    )
                    event_data["ball_displacement"] = ball_disp

                    events.append(event_data)

                    # Generate random negative example if requested
                    if generate_random:
                        random_data = self._generate_random_chunk(
                            desired_length=end - start,
                            exclude_intervals=all_event_intervals,
                            exclude_interactions=self.fly.config.random_exclude_interactions,
                            interaction_map=self.fly.config.random_interaction_map,
                        )
                        if random_data is not None:
                            random_data["event_id"] = event_counter
                            random_data["event_type"] = "random"
                            random_data["fly_idx"] = fly_idx
                            random_data["ball_idx"] = ball_idx
                            events.append(random_data)

                    event_counter += 1

        return pd.concat(events).reset_index(drop=True) if events else pd.DataFrame()

    def _generate_random_chunk(
        self,
        desired_length,
        exclude_intervals,
        exclude_interactions=True,
        interaction_map="full",
    ):
        """Generate random chunk of tracking data that doesn't overlap with any events"""
        max_start = len(self.fly_centered_tracks) - desired_length

        if max_start <= 0:
            return None

        for _ in range(100):  # Try up to 100 times to find non-overlapping segment
            random_start = np.random.randint(0, max_start)
            random_end = random_start + desired_length

            # Check overlap with existing events
            overlap = False
            for ex_start, ex_end in exclude_intervals:
                if (random_start < ex_end) and (random_end > ex_start):
                    overlap = True
                    break

            if not overlap:
                if exclude_interactions:
                    if interaction_map == "full":
                        # Collect ALL interaction event intervals across flies/balls
                        interaction_frames = [
                            (event[0], event[1])
                            for fly_dict in self.fly.tracking_data.interaction_events.values()
                            for ball_events in fly_dict.values()
                            for event in ball_events
                        ]
                    elif interaction_map == "onset":
                        # Existing onset handling (correct)
                        interaction_frames = [
                            (onset, onset + desired_length)
                            for _, onsets in self.fly.tracking_data.interactions_onsets.items()
                            for onset in onsets
                        ]
                    else:
                        raise ValueError(
                            "Invalid interaction_map. Use 'full' or 'onset'."
                        )

                    # Ensure interaction_frames is a list of tuples
                    if isinstance(interaction_frames, dict):
                        interaction_frames = [
                            (start, end) for start, end in interaction_frames.items()
                        ]
                    elif isinstance(interaction_frames, list) and all(
                        isinstance(i, int) for i in interaction_frames
                    ):
                        interaction_frames = [
                            (i, i + desired_length) for i in interaction_frames
                        ]

                    for ex_start, ex_end in interaction_frames:
                        if (random_start < ex_end) and (random_end > ex_start):
                            overlap = True
                            break

                if not overlap:
                    random_data = self.fly_centered_tracks.iloc[
                        random_start:random_end
                    ].copy()
                    random_data["time_rel_onset"] = np.nan
                    random_data["adjusted_frame"] = range(desired_length)

                    # Calculate ball displacement (should be near zero for non-events)
                    ball_disp = np.sqrt(
                        (
                            random_data["x_centre_preprocessed"]
                            - random_data["x_centre_preprocessed"].iloc[0]
                        )
                        ** 2
                        + (
                            random_data["y_centre_preprocessed"]
                            - random_data["y_centre_preprocessed"].iloc[0]
                        )
                        ** 2
                    )
                    random_data["ball_displacement"] = ball_disp

                    return random_data

        return None

    def get_final_contact(self, threshold=None, init=False):
        if threshold is None:
            threshold = self.fly.config.final_event_threshold

        final_contact, final_contact_idx = self.find_contact_by_distance(
            threshold, distance_type="threshold"
        )

        final_contact_time = (
            final_contact[0] / self.fly.experiment.fps if final_contact else None
        )

        # print(f"final_contact_idx from skeleton metric: {final_contact_idx}")

        return final_contact_idx, final_contact_time

    def compute_ball_displacements(self):
        """
        Compute the derivative of the ball position for each contact event
        """

        self.ball_displacements = []

        for event in self.contacts:
            # Get the ball positions for the event
            ball_positions = self.ball.objects[0].dataset.loc[event[0] : event[1]]
            # Get the derivative of the ball positions

            ball_velocity = np.mean(
                abs(np.diff(ball_positions["y_centre_preprocessed"], axis=0))
            )

            self.ball_displacements.append(ball_velocity)

        return self.ball_displacements

    def compute_fly_centered_tracks(self):
        """
        Compute fly-centric coordinates by:
        1. Translating all points to have thorax as origin
        2. Rotating to align thorax-head direction with positive y-axis

        Returns DataFrame with '_fly' suffix columns for transformed coordinates
        """
        tracking_data = self.fly.tracking_data.skeletontrack.objects[0].dataset.copy()

        # Add ball coordinates to tracking data
        tracking_data["x_centre_preprocessed"] = self.ball.objects[0].dataset[
            "x_centre_preprocessed"
        ]
        tracking_data["y_centre_preprocessed"] = self.ball.objects[0].dataset[
            "y_centre_preprocessed"
        ]

        # Get reference points
        thorax_x = tracking_data["x_Thorax"].values
        thorax_y = tracking_data["y_Thorax"].values
        head_x = tracking_data["x_Head"].values
        head_y = tracking_data["y_Head"].values

        # Calculate direction vector components
        dx = head_x - thorax_x
        dy = head_y - thorax_y
        mag = np.hypot(dx, dy)
        valid = mag > 1e-6  # Valid frames with measurable head direction

        # Calculate rotation components (vectorized operations)
        cos_theta = dy / mag
        sin_theta = dx / mag
        cos_theta[~valid] = 0  # Handle invalid frames
        sin_theta[~valid] = 0

        # Get all trackable nodes (excluding existing '_fly' columns)
        nodes = [
            col[2:]
            for col in tracking_data.columns
            if col.startswith("x_") and not col.endswith("_fly")
        ]

        # Create transformed dataframe
        transformed = tracking_data.copy()

        for node in nodes:
            x_col = f"x_{node}"
            y_col = f"y_{node}"

            # 1. Translate to thorax-centric coordinates
            x_trans = tracking_data[x_col] - thorax_x
            y_trans = tracking_data[y_col] - thorax_y

            # 2. Apply rotation matrix (vectorized)
            # New x = x_trans * cosθ - y_trans * sinθ
            # New y = x_trans * sinθ + y_trans * cosθ
            transformed[f"{x_col}_fly"] = x_trans * cos_theta - y_trans * sin_theta
            transformed[f"{y_col}_fly"] = x_trans * sin_theta + y_trans * cos_theta

            # Handle invalid frames using config value
            transformed.loc[~valid, [f"{x_col}_fly", f"{y_col}_fly"]] = (
                self.fly.config.hidden_value
            )

        return transformed

    def compute_fly_centered_tracks_old(self):
        """
        Compute the fly-relative tracks for each skeleton tracking datapoint
        """

        tracking_data = self.fly.tracking_data.skeletontrack.objects[0].dataset
        # Add the ball tracking data

        tracking_data["x_centre_preprocessed"] = self.ball.objects[0].dataset[
            "x_centre_preprocessed"
        ]
        tracking_data["y_centre_preprocessed"] = self.ball.objects[0].dataset[
            "y_centre_preprocessed"
        ]

        thorax = tracking_data[["x_Thorax", "y_Thorax"]].values
        head = tracking_data[["x_Head", "y_Head"]].values

        # Vectorized calculations
        dxdy = head - thorax
        mag = np.linalg.norm(dxdy, axis=1)
        valid = mag > 1e-6  # Filter frames with valid head direction

        # Only calculate rotation where valid
        cos_theta = np.zeros_like(mag)
        sin_theta = np.zeros_like(mag)
        cos_theta[valid] = dxdy[valid, 1] / mag[valid]
        sin_theta[valid] = dxdy[valid, 0] / mag[valid]

        # Transform all points using matrix operations
        translated = tracking_data.filter(like="x_").values - thorax[:, 0][:, None]
        rotated = np.empty_like(translated)
        rotated[valid] = (
            translated[valid] * cos_theta[valid, None]
            - translated[valid] * sin_theta[valid, None]
        )

        # Create transformed dataframe
        transformed = tracking_data.copy()
        for i, col in enumerate(
            [c for c in tracking_data.columns if c.startswith("x_")]
        ):
            transformed[f"{col}_fly"] = rotated[:, i]

        return transformed

    def plot_skeleton_and_ball(self, frame=2039):
        """
        Plot the skeleton and ball tracking data on a given frame.
        """

        nodes_list = self.fly.tracking_data.skeletontrack.node_names + [
            node for node in self.ball.node_names if "preprocessed" in node
        ]

        annotated_frame = generate_annotated_frame(
            video=self.fly.tracking_data.skeletontrack.video,
            sleap_tracks_list=[self.fly.tracking_data.skeletontrack, self.ball],
            frame=frame,
            nodes=nodes_list,
        )

        # Plot the frame with the skeleton and ball tracking data
        plt.imshow(annotated_frame)
        plt.axis("off")
        plt.show()

        return annotated_frame

    def generate_contacts_video(self, output_path):
        """
        Generate a video of all contact events concatenated with annotations.
        """
        video_clips = []
        video = VideoFileClip(str(self.fly.tracking_data.skeletontrack.video))

        for idx, event in enumerate(self.contacts):
            start_frame, end_frame = event[0], event[1]
            start_time = start_frame / self.fly.experiment.fps
            end_time = end_frame / self.fly.experiment.fps
            clip = video.subclip(start_time, end_time)

            # Calculate the start time in seconds
            start_time_seconds = start_frame / 29  # Assuming 29 fps

            # Convert start time to mm:ss format
            minutes, seconds = divmod(start_time_seconds, 60)
            start_time_formatted = f"{int(minutes):02d}:{int(seconds):02d}"

            # Create a text annotation with the contact index and start time
            annotation_text = f"Contact {idx + 1}\nStart Time: {start_time_formatted}"
            annotation = TextClip(
                annotation_text,
                fontsize=8,
                color="white",
                bg_color="black",
                font="Arial",  # Specify a font that's definitely installed
                method="label",
            )
            annotation = annotation.set_position(("center", "bottom")).set_duration(
                clip.duration
            )

            # Overlay the annotation on the video clip
            annotated_clip = CompositeVideoClip([clip, annotation])
            video_clips.append(annotated_clip)

        concatenated_clip = concatenate_videoclips(video_clips)
        concatenated_clip.write_videofile(str(output_path), codec="libx264")

        print(f"Contacts video saved to {output_path}")


class Fly:
    """
    A class for a single fly. This represents a folder containing a video, associated tracking files, and metadata files. It is usually contained in an Experiment object, and inherits the Experiment object's metadata.
    """

    def __init__(
        self,
        directory,
        experiment=None,
        #experiment_type=None,
        as_individual=False,
        #time_range=None,
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
            self.experiment = Experiment(
                self.directory.parent.parent, metadata_only=True
            )
        else:
            self.experiment = Experiment(self.directory.parent.parent)

        self.config = self.experiment.config

        #self.experiment_type = experiment_type

        if self.config.experiment_type:
            self.config.set_experiment_config(self.config.experiment_type)

        self.metadata = FlyMetadata(self)

        self._tracking_data = None

        # Check if the fly has valid tracking data

        if as_individual:
            if not self.tracking_data.valid_data:
                print(f"Invalid data for: {self.metadata.name}. Skipping.")
                return

        self.flyball_positions = None
        self.fly_skeleton = None

        self._events_metrics = None

        self._f1_metrics = None

        self._learning_metrics = None

        self._skeleton_metrics = None

    @property
    def tracking_data(self):
        if self._tracking_data is None:
            self._tracking_data = FlyTrackingData(self)
            if not self._tracking_data.valid_data:
                print(f"Invalid data for: {self.metadata.name}. Skipping.")
                self._tracking_data = None
        return self._tracking_data

    @property
    def events_metrics(self):
        if self._events_metrics is None:
            # print("Computing events metrics...")
            self._events_metrics = BallpushingMetrics(self.tracking_data).metrics
        return self._events_metrics

    @property
    def f1_metrics(self):
        if self._f1_metrics is None and self.config.experiment_type == "F1":
            self._f1_metrics = F1Metrics(self.tracking_data).metrics
        return self._f1_metrics

    @property
    def learning_metrics(self):
        if self._learning_metrics is None and self.config.experiment_type == "Learning":
            self._learning_metrics = LearningMetrics(self.tracking_data)
        return self._learning_metrics

    @property
    def skeleton_metrics(self):
        if self.tracking_data.skeletontrack is None:
            print("No skeleton data available.")
        elif (
            self._skeleton_metrics is None
            and self.tracking_data.skeletontrack is not None
        ):
            self._skeleton_metrics = SkeletonMetrics(self)
        return self._skeleton_metrics

    def __str__(self):
        # Get the genotype from the metadata
        genotype = self.metadata.arena_metadata["Genotype"]

        return f"Fly: {self.metadata.name}\nArena: {self.metadata.arena}\nCorridor: {self.metadata.corridor}\nVideo: {self.metadata.video}\nFlytrack: {self.tracking_data.flytrack}\nBalltrack: {self.tracking_data.balltrack}\nGenotype: {genotype}"

    def __repr__(self):
        return f"Fly({self.directory})"

    ################################ Video clip generation ################################

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
            event = self.tracking_data.interaction_events[event - 1]

        start_frame, end_frame = event[0], event[1]
        cap = cv2.VideoCapture(str(self.metadata.video))

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
            event_index = self.tracking_data.interaction_events.index(event)

            if outpath == get_labserver() / "Videos":
                clip_path = outpath.joinpath(
                    f"{self.metadata.name}_{event_index}.mp4"
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
            print(f"No tracking data available for {self.metadata.name}. Skipping...")
            return

        if outpath is None:
            outpath = self.directory
        events = self.tracking_data.interaction_events
        clips = []

        cap = cv2.VideoCapture(str(self.metadata.video))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        vidname = (
            f"{self.metadata.name}_{self.Genotype if self.Genotype else 'undefined'}"
        )

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
                / f"{self.metadata.name}_{self.Genotype if self.Genotype else 'undefined'}_x{speed}.mp4"
            )

        # Load the video file
        clip = VideoFileClip(self.metadata.video.as_posix())

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
            print(
                f"Saving {self.metadata.video.name} at {speed}x speed in {output_path.parent}"
            )
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

            print(f"Previewing {self.metadata.video.name} at {speed}x speed")

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

    def __init__(
        self,
        directory,
        metadata_only=False,
        experiment_type=None,
    ):
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

        self.config = Config()
        self.directory = Path(directory)
        self.metadata = self.load_metadata()
        self.fps = self.load_fps()

        #self.experiment_type = experiment_type

        # If metadata_only is True, don't load the flies
        if not metadata_only:
            self.flies = self.load_flies()

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

    def load_flies(self, multithreading=False):
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
                        mp4_file = list(
                            dir.glob(f"{dir.parent.name}_corridor_{dir.name[-1]}.mp4")
                        )[0]
                    except IndexError:
                        try:
                            # Look for any .mp4 file in the directory
                            mp4_file = list(dir.glob("*.mp4"))[0]
                        except IndexError:
                            print(
                                f"No video found for {dir.name}. Moving to the next directory."
                            )
                            continue  # Move on to the next directory
                # print(f"Found video {mp4_file.name} for {dir.name}")
                mp4_files.append(mp4_file)

        # Create a Fly object for each .mp4 file using multiprocessing
        flies = []
        if multithreading:
            with Pool(processes=os.cpu_count()) as pool:
                results = [
                    pool.apply_async(
                        load_fly,
                        args=(
                            mp4_file,
                            self,
                            #self.experiment_type,
                        ),
                    )
                    for mp4_file in mp4_files
                ]
                for result in results:
                    fly = result.get()
                    if fly is not None:
                        flies.append(fly)
        else:
            for mp4_file in mp4_files:
                fly = load_fly(
                    mp4_file,
                    self,
                    #experiment_type=self.experiment_type,
                )
                if fly is not None:
                    flies.append(fly)

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


class Dataset:
    def __init__(
        self,
        source,
        brain_regions_path="/mnt/upramdya_data/MD/Region_map_240312.csv",
        dataset_type="coordinates",
    ):
        """
        A class to generate a Dataset from Experiments and Fly objects.

        It is in essence a list of Fly objects that can be used to generate a pandas DataFrame containing chosen metrics for each fly.

        Parameters
        ----------
        source : can either be a list of Experiment objects, one Experiment object, a list of Fly objects or one Fly object.

        """
        self.config = Config()

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

        self.flies = [fly for fly in self.flies if fly._tracking_data.valid_data]

        self.brain_regions_path = brain_regions_path
        self.regions_map = pd.read_csv(self.brain_regions_path)

        self.metadata = []

        self.data = None

        self.generate_dataset(metrics=dataset_type)

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

    def find_flies(self, on, value):
        """
        Makes a list of Fly objects matching a certain criterion.

        Parameters
        ----------
        on : str
            The name of the attribute to filter on. Can be a nested attribute (e.g., 'metadata.name').

        value : str
            The value of the attribute to filter on.

        Returns
        ----------
        list
            A list of Fly objects matching the criterion.
        """

        def get_nested_attr(obj, attr):
            """Helper function to get nested attributes."""
            for part in attr.split("."):
                obj = getattr(obj, part, None)
                if obj is None:
                    return None
            return obj

        return [fly for fly in self.flies if get_nested_attr(fly, on) == value]

    def generate_dataset(self, metrics="coordinates"):
        """Generates a pandas DataFrame from a list of Experiment objects. The dataframe contains the smoothed fly and ball positions for each experiment.

        Args:
            experiments (list): A list of Experiment objects.
            metrics (str): The kind of dataset to generate. Currently, the following metrics are available:
            - 'coordinates': The fly and ball coordinates for each frame.
            - 'summary': Summary metrics for each fly. These are single values for each fly (e.g. number of events, duration of the breaks between events, etc.). A list of available summary metrics can be found in _prepare_dataset_summary_metrics documentation.

        Returns:
            dict: A dictionary where keys are ball types and values are DataFrames containing selected metrics for each fly and associated metadata.
        """

        Dataset = []

        try:
            if metrics == "coordinates":
                for fly in self.flies:
                    data = self._prepare_dataset_coordinates(fly)
                    Dataset.append(data)

            elif metrics == "contact_data":
                for fly in self.flies:
                    data = self._prepare_dataset_contact_data(fly)
                    if not data.empty:
                        Dataset.append(data)

            elif metrics == "summary":
                for fly in self.flies:
                    # print("Preparing dataset for", fly.name)
                    data = self._prepare_dataset_summary_metrics(fly)
                    # print(f"Data : {data}")
                    Dataset.append(data)

            elif metrics == "F1_coordinates":
                for fly in self.flies:
                    data = self._prepare_dataset_F1_coordinates(fly)
                    Dataset.append(data)

            elif metrics == "F1_summary":
                for fly in self.flies:
                    data = self._prepare_dataset_F1_summary(fly)
                    Dataset.append(data)

            elif metrics == "F1_checkpoints":
                for fly in self.flies:
                    data = self._prepare_dataset_F1_checkpoints(fly)
                    Dataset.append(data)
            elif metrics == "Skeleton_contacts":
                for fly in self.flies:
                    data = self._prepare_dataset_skeleton_contacts(fly)
                    Dataset.append(data)
            elif metrics == "standardized_contacts":
                for fly in self.flies:
                    data = self._prepare_dataset_standardized_contacts(fly)
                    if not data.empty:
                        Dataset.append(data)

            if Dataset:
                self.data = pd.concat(Dataset).reset_index()
            else:
                self.data = pd.DataFrame()  # Return an empty DataFrame if no data

        except Exception as e:
            print(f"An error occurred: {e}")
            traceback.print_exc()
            self.data = pd.DataFrame()  # Return an empty DataFrame in case of error

        return self.data

    def _prepare_dataset_coordinates(
        self, fly, downsampling_factor=None, annotate_events=True
    ):
        """
        Helper function to prepare individual fly dataset with fly and ball coordinates. It also adds the fly name, experiment name, and arena metadata as categorical data.

        Args:
            fly (Fly): A Fly object.
            downsampling_factor (int): The factor (in seconds) by which to downsample the dataset. Defaults to None.
            annotate_events (bool): Whether to annotate the dataset with interaction events. Defaults to False.

        Returns:
            pandas.DataFrame: A DataFrame containing the fly's coordinates and associated metadata.
        """
        downsampling_factor = fly.config.downsampling_factor or downsampling_factor

        # Extract fly and ball tracking data
        flydata = [obj.dataset for obj in fly.tracking_data.flytrack.objects]
        balldata = [obj.dataset for obj in fly.tracking_data.balltrack.objects]

        # Initialize dataset with time and frame columns
        dataset = pd.DataFrame(
            {
                "time": flydata[0]["time"],
                "frame": flydata[0]["frame"],
                "adjusted_time": (
                    fly.f1_metrics["adjusted_time"]
                    if fly.tracking_data.exit_time
                    else np.nan
                ),
            }
        )

        # Add fly coordinates and distances
        for i, data in enumerate(flydata):
            dataset[f"x_fly_{i}"] = data["x_thorax"] - fly.tracking_data.start_x
            dataset[f"y_fly_{i}"] = data["y_thorax"] - fly.tracking_data.start_y
            dataset[f"distance_fly_{i}"] = np.sqrt(
                dataset[f"x_fly_{i}"].pow(2) + dataset[f"y_fly_{i}"].pow(2)
            )

        # Add ball coordinates and distances
        for i, data in enumerate(balldata):
            dataset[f"x_ball_{i}"] = data["x_centre"] - fly.tracking_data.start_x
            dataset[f"y_ball_{i}"] = data["y_centre"] - fly.tracking_data.start_y
            dataset[f"distance_ball_{i}"] = np.sqrt(
                dataset[f"x_ball_{i}"].pow(2) + dataset[f"y_ball_{i}"].pow(2)
            )

        # Downsample the dataset if required
        if downsampling_factor:
            dataset = dataset.iloc[:: downsampling_factor * fly.experiment.fps]

        # Annotate interaction events if required
        if annotate_events:
            self._annotate_interaction_events(dataset, fly)

        # Add trial information for learning experiments
        if fly.config.experiment_type == "Learning" and hasattr(fly, "learning_metrics"):
            self._add_trial_information(dataset, fly)

        # Add metadata
        dataset = self._add_metadata(dataset, fly)

        return dataset

    # TODO : Events should be reannotated if the dataset is subsetted

    # TODO: implement events durations

    def _annotate_interaction_events(self, dataset, fly):
        """
        Annotate the dataset with interaction events and their onsets.

        Args:
            dataset (pd.DataFrame): The dataset to annotate.
            fly (Fly): A Fly object.
        """
        interaction_events = fly.tracking_data.interaction_events
        event_frames = np.full(len(dataset), np.nan)
        event_onset_frames = np.full(len(dataset), np.nan)

        event_index = 0
        for fly_idx, ball_events in interaction_events.items():
            for ball_idx, events in ball_events.items():
                for event in events:
                    event_frames[event[0] : event[1]] = event_index
                    event_index += 1

        dataset["interaction_event"] = event_frames

        # Annotate interaction event onsets
        event_onset = 0
        for (
            fly_idx,
            ball_idx,
        ), onsets in fly.tracking_data.interactions_onsets.items():
            for onset in onsets:
                if onset is not None and 0 <= onset < len(event_onset_frames):
                    event_onset_frames[onset] = event_onset
                    event_onset += 1

        dataset["interaction_event_onset"] = event_onset_frames

    def _add_trial_information(self, dataset, fly):
        """
        Add trial information (trial, trial_frame, trial_time) to the dataset.

        Args:
            dataset (pd.DataFrame): The dataset to annotate.
            fly (Fly): A Fly object.
        """
        trials_data = fly.learning_metrics.trials_data
        frame_to_trial = trials_data["trial"].to_dict()

        # Map trial numbers to the dataset
        dataset["trial"] = dataset.index.map(frame_to_trial).fillna(np.nan)

        # Calculate trial_frame as the frame index relative to the start of each trial
        dataset["trial_frame"] = dataset.groupby("trial").cumcount()

        # Compute trial_time from the trials_data
        
        dataset["trial_time"] = dataset["trial_frame"] / fly.experiment.fps

    def _prepare_dataset_contact_data(self, fly, hidden_value=None):
        if hidden_value is None:
            hidden_value = self.config.hidden_value

        all_contact_data = []
        contact_indices = []

        if fly.skeleton_metrics is None:
            print(f"No skeleton metrics found for fly {fly.metadata.name}. Skipping...")
            return pd.DataFrame()  # Return an empty DataFrame
        else:
            skeleton_data = fly.tracking_data.skeletontrack.objects[0].dataset
            ball_data = fly.tracking_data.balltrack.objects[0].dataset

            contact_events = fly.skeleton_metrics.all_contacts

            # Apply success_cutoff if enabled
            if fly.config.success_cutoff:

                # print("applying success cutoff to dataset")

                final_contact_idx, _ = fly.skeleton_metrics.get_final_contact()
                if final_contact_idx is not None:
                    contact_events = contact_events[: final_contact_idx + 1]

                    # print(f"Final contact index in dataset: {final_contact_idx}")
                    # print(
                    #     f"Applying success cutoff. Number of contact events after cutoff: {len(contact_events)}"
                    # )

            for event_idx, event in enumerate(contact_events):
                start_idx, end_idx = event[0], event[1]
                contact_data = skeleton_data.iloc[start_idx:end_idx]
                event_ball_data = ball_data.iloc[start_idx:end_idx]
                contact_data["contact_index"] = event_idx  # Add contact_index column

                # Add ball data to the contact data
                for col in event_ball_data.columns:
                    contact_data[col] = event_ball_data[col].values

                all_contact_data.append(contact_data)
                contact_indices.append(event_idx)

            if all_contact_data:
                combined_data = pd.concat(all_contact_data, ignore_index=True)
                combined_data.fillna(hidden_value, inplace=True)
                combined_data = self._add_metadata(combined_data, fly)
            else:
                combined_data = pd.DataFrame()

            # nb_events = len(contact_indices)
            # print(f"Number of contact events: {nb_events}")

        return combined_data

    def _prepare_dataset_summary_metrics(
        self,
        fly,
        metrics=[],
    ):
        """
        Prepares a dataset with summary metrics for a given fly. The metrics are computed for all events, but only the ones specified in the 'metrics' argument are included in the returned DataFrame.

        Args:
            fly (Fly): A Fly object.
            metrics (list): A list of metrics to include in the dataset. The metrics require valid ball and fly tracking data. If the list is empty, all available metrics are included. Defaults to [].

        Returns:
            pandas.DataFrame: A DataFrame containing the fly's summary metrics and associated metadata.
        """
        # Initialize an empty DataFrame
        dataset = pd.DataFrame()

        # If no specific metrics are provided, include all available metrics
        if not metrics:
            metrics = list(fly.events_metrics[next(iter(fly.events_metrics))].keys())

        # For each pair of fly and ball, get the metrics from the Fly metrics
        for key, metric_dict in fly.events_metrics.items():

            for metric in metrics:
                if metric in metric_dict:
                    value = metric_dict[metric]
                    # Ensure the value is in the expected format
                    if isinstance(value, list) or isinstance(value, dict):
                        value = str(value)  # Convert lists and dicts to strings
                    dataset.at[key, metric] = value

        if fly.f1_metrics:
            dataset["direction_match"] = fly.f1_metrics["direction_match"]

            # Assign summaries condition based on F1 condition

            dataset["F1_condition"] = fly.metadata.F1_condition

            if dataset["F1_condition"].iloc[0] == "control":
                for key in fly.events_metrics.keys():
                    if key == "fly_0_ball_0":
                        dataset.at[key, "ball_condition"] = "test"
            else:
                for key in fly.events_metrics.keys():
                    if key == "fly_0_ball_0":
                        dataset.at[key, "ball_condition"] = "training"
                    elif key == "fly_0_ball_1":
                        dataset.at[key, "ball_condition"] = "test"

        # Add metadata to the dataset
        dataset = self._add_metadata(dataset, fly)

        # print(dataset.columns)

        return dataset

    def _prepare_dataset_F1_coordinates(self, fly, downsampling_factor=None):
        # Check if the fly ever exits the corridor
        if fly.tracking_data.exit_time is None:
            print(f"Fly {fly.metadata.name} never exits the corridor")
            return

        downsampling_factor = fly.config.downsampling_factor

        dataset = pd.DataFrame()

        dataset["time"] = fly.tracking_data.flytrack.objects[0].dataset["time"]
        dataset["frame"] = fly.tracking_data.flytrack.objects[0].dataset["frame"]

        dataset["adjusted_time"] = fly.f1_metrics["adjusted_time"]

        # Assign training_ball and test_ball distances separately

        if fly.metadata.F1_condition != "control":
            dataset["training_ball"] = fly.f1_metrics["training_ball_distances"][
                "euclidean_distance"
            ]

        dataset["test_ball"] = fly.f1_metrics["test_ball_distances"][
            "euclidean_distance"
        ]

        # Exclude the fly if test_ball_distances has moved > 10 px before adjusted time 0
        NegData = dataset[dataset["adjusted_time"] < 0]

        if NegData["test_ball"].max() > 10:
            print(f"Fly {fly.metadata.name} excluded due to premature ball movements")
            return

        if downsampling_factor:
            dataset = dataset.iloc[:: downsampling_factor * fly.experiment.fps]

        dataset = self._add_metadata(dataset, fly)

        return dataset

    def _prepare_dataset_F1_checkpoints(self, fly):
        if fly.tracking_data.exit_time is None:
            print(f"Fly {fly.metadata.name} never exits the corridor")
            return pd.DataFrame()  # Return an empty DataFrame

        # Create a list of dictionaries for each checkpoint
        data = [
            {
                "fly_exit_time": fly.tracking_data.exit_time,
                "distance": distance,
                "adjusted_time": adjusted_time,
            }
            for distance, adjusted_time in fly.f1_metrics["F1_checkpoints"].items()
        ]

        # Convert the list of dictionaries to a DataFrame
        dataset = pd.DataFrame(data)

        # Add metadata to the dataset
        dataset = self._add_metadata(dataset, fly)

        if fly.metadata.F1_condition == "control":
            dataset["success_direction"] = fly.events_metrics["fly_0_ball_0"][
                "success_direction"
            ]

        else:
            dataset["success_direction"] = fly.events_metrics["fly_0_ball_1"][
                "success_direction"
            ]

        return dataset

    def _prepare_dataset_skeleton_contacts(self, fly):
        """
        Prepares a dataset with the fly's contacts with the ball and associated ball displacement, + metadata
        """

        if not fly.skeleton_metrics:
            print(f"No skeleton metrics found for fly {fly.metadata.name}")
            return pd.DataFrame()

        # Check if there are contacts for this fly
        if not fly.skeleton_metrics.ball_displacements:
            print(f"No contacts found for fly {fly.metadata.name}")
            return pd.DataFrame()

        dataset = pd.DataFrame()

        # Get the ball displacements of the fly

        ball_displacements = fly.skeleton_metrics.ball_displacements

        # Use the list indices + 1 as the contact indices

        contact_indices = [i + 1 for i in range(len(ball_displacements))]

        # Create a DataFrame with the contact indices and the ball displacements

        dataset["contact_index"] = contact_indices
        dataset["ball_displacement"] = ball_displacements

        # Add metadata to the dataset

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
            dataset["fly"] = fly.metadata.name
            dataset["fly"] = dataset["fly"].astype("category")

            # Add a column with the path to the fly's folder
            dataset["flypath"] = fly.metadata.directory.as_posix()
            dataset["flypath"] = dataset["flypath"].astype("category")

            # Add a column with the experiment name as categorical data
            dataset["experiment"] = fly.experiment.directory.name
            dataset["experiment"] = dataset["experiment"].astype("category")

            # Handle missing values for 'Nickname' and 'Brain region'
            dataset["Nickname"] = (
                fly.metadata.nickname
                if fly.metadata.nickname is not None
                else "Unknown"
            )
            dataset["Brain region"] = (
                fly.metadata.brain_region
                if fly.metadata.brain_region is not None
                else "Unknown"
            )
            dataset["Simplified Nickname"] = (
                fly.metadata.simplified_nickname
                if fly.metadata.simplified_nickname is not None
                else "Unknown"
            )
            dataset["Split"] = (
                fly.metadata.split if fly.metadata.split is not None else "Unknown"
            )

            # Add the metadata for the fly's arena as columns
            for var, data in fly.metadata.arena_metadata.items():
                # Handle missing values in arena metadata
                data = data if data is not None else "Unknown"
                dataset[var] = data
                dataset[var] = dataset[var].astype("category")

                # If the variable name is not in the metadata list, add it
                if var not in self.metadata:
                    self.metadata.append(var)

        except Exception as e:
            print(
                f"Error occurred while adding metadata for fly {fly.metadata.name}: {str(e)}"
            )
            print(f"Current dataset:\n{dataset}")

        return dataset

    def _prepare_dataset_standardized_contacts(self, fly):
        """Prepares standardized contact event windows for analysis"""
        if not hasattr(fly, "skeleton_metrics") or fly.skeleton_metrics is None:
            return pd.DataFrame()

        events_df = fly.skeleton_metrics.events_based_contacts

        # Add trial information for learning experiments
        if fly.config.experiment_type == "Learning" and hasattr(fly, "learning_metrics"):
            trials_data = fly.learning_metrics.trials_data

            # Create mapping from frame to trial
            frame_to_trial = dict(zip(trials_data.index, trials_data["trial"]))

            # Add trial column with default value of 0
            events_df["trial"] = 0

            # Update with actual trial values - more efficient with vectorized operations
            common_indices = events_df.index.intersection(trials_data.index)
            if not common_indices.empty:
                events_df.loc[common_indices, "trial"] = trials_data.loc[
                    common_indices, "trial"
                ].values

        # add metadata
        events_df = self._add_metadata(events_df, fly)

        return events_df

    def compute_behavior_map(
        self,
        perplexity=30,
        n_iter=3000,
        pca_components=50,
        hidden_value=None,
        savepath=None,
    ):
        if hidden_value is None:
            hidden_value = self.config.hidden_value

        all_contact_data = []
        metadata = []
        contact_indices = []

        print("Fetching contacts for all flies...")

        for fly_idx, fly in enumerate(self.flies):

            if fly.skeleton_metrics is None:
                print(
                    f"No skeleton metrics found for fly {fly.metadata.name}. Skipping..."
                )
                continue
            else:
                skeleton_data = fly.tracking_data.skeletontrack.objects[0].dataset
                contact_events = fly.skeleton_metrics.find_contact_events()

                for event_idx, event in enumerate(contact_events):
                    start_idx, end_idx = event[0], event[1]
                    contact_data = skeleton_data.iloc[start_idx:end_idx]
                    all_contact_data.append(contact_data)

                    metadata.append(
                        {
                            "name": fly.metadata.name,
                            "flypath": fly.metadata.directory.as_posix(),
                            "experiment": fly.experiment.directory.name,
                            "Nickname": (
                                fly.metadata.nickname
                                if fly.metadata.nickname is not None
                                else "Unknown"
                            ),
                            "Brain region": (
                                fly.metadata.brain_region
                                if fly.metadata.brain_region is not None
                                else "Unknown"
                            ),
                            "Simplified Nickname": (
                                fly.metadata.simplified_nickname
                                if fly.metadata.simplified_nickname is not None
                                else "Unknown"
                            ),
                            "Split": (
                                fly.metadata.split
                                if fly.metadata.split is not None
                                else "Unknown"
                            ),
                            **{
                                var: data if data is not None else "Unknown"
                                for var, data in fly.metadata.arena_metadata.items()
                            },
                        }
                    )

                    contact_indices.append(event_idx)

        combined_data = pd.concat(all_contact_data, ignore_index=True)
        combined_data.fillna(hidden_value, inplace=True)

        print("Applying PCA...")

        features = combined_data.filter(regex="^(x|y)_").values

        # Adjust PCA components if necessary
        n_features = features.shape[1]
        pca_components = min(pca_components, n_features)

        pca = PCA(n_components=pca_components)
        pca_results = pca.fit_transform(features)

        print("PCA completed. Starting t-SNE...")

        # Create the progress callback
        progress_callback = ProgressCallback(n_iter)

        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            n_iter=n_iter,
            random_state=42,
            n_jobs=-1,
            callbacks=[progress_callback],
            callbacks_every_iters=1,
        )
        tsne_results = tsne.fit(pca_results)

        # Close the progress bar
        progress_callback.close()

        tsne_df = pd.DataFrame(
            tsne_results, columns=["t-SNE Component 1", "t-SNE Component 2"]
        )
        metadata_df = pd.DataFrame(metadata)

        result_df = pd.concat(
            [
                tsne_df,
                metadata_df.reset_index(drop=True),
                pd.DataFrame({"contact_index": contact_indices}),
            ],
            axis=1,
        )

        self.behavior_map = result_df

        if savepath:
            result_df.to_csv(savepath, index=False)
            print(f"Behavior map saved to {savepath}")

        return result_df

    def generate_clip(self, fly, event, outpath):
        """
        Make a video clip of a fly's event.
        """

        # Get the fly object
        flies = self.find_flies("metadata.name", fly)
        if not flies:
            raise ValueError(f"Fly with name {fly} not found.")
        fly = flies[0]

        # Get the event start and end frames
        event_start = fly.skeleton_metrics.contacts[event][0]
        event_end = fly.skeleton_metrics.contacts[event][1]

        # Get the video file
        video_file = fly.metadata.video

        # Open the video file
        cap = cv2.VideoCapture(str(video_file))
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {video_file}")

        # Get the frame rate of the video
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        # Set the start and end frames
        cap.set(cv2.CAP_PROP_POS_FRAMES, event_start)
        start_frame = event_start
        end_frame = event_end

        # Create a VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        # Get the video dimensions
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Create the output file
        out = cv2.VideoWriter(outpath, fourcc, fps, (width, height))

        try:
            # Go to the start frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            # Read the video frame by frame and write to the output file
            while cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    out.write(frame)
                    if cap.get(cv2.CAP_PROP_POS_FRAMES) >= end_frame:
                        break
                else:
                    break
        finally:
            # Release the video capture and writer objects
            cap.release()
            out.release()

        return outpath

    def make_events_grid(self, events_dict, clip_path):
        """
        This function will generate a grid video of all the events selected in events_dict, identified by the fly name and the event index.

        Args:
            events_dict (list): List of dictionaries with keys "name" and "event_index".
            clip_path (str): The path where the grid video will be saved.
        """

        clips = []

        for event in events_dict:
            print(event)

            fly_name = event["name"]
            event_index = event["event_index"]

            try:
                # Generate the clip for the event using the new generate_clip method
                event_clip_path = self.generate_clip(
                    fly_name,
                    event_index,
                    outpath=f"{os.path.dirname(clip_path)}/{fly_name}_event_{event_index}.mp4",
                )
                clips.append(event_clip_path)
            except Exception as e:
                print(
                    f"Error generating clip for fly {fly_name}, event {event_index}: {e}"
                )
                continue

        if clips:
            # Concatenate the clips into a grid video
            self.concatenate_clips(clips, clip_path)

            # Remove the individual clip files
            for event_clip_path in clips:
                os.remove(event_clip_path)

            print(f"Finished processing events grid! Saved to {clip_path}")
        else:
            print("No clips were generated.")

    def concatenate_clips(self, clips, output_path):
        """
        Concatenate the clips into a grid video.

        Args:
            clips (list): List of paths to the video clips.
            output_path (str): The path where the grid video will be saved.
        """
        video_clips = [VideoFileClip(clip) for clip in clips]

        # Determine the number of rows and columns for the grid
        num_clips = len(video_clips)
        grid_size = int(np.ceil(np.sqrt(num_clips)))

        # Create a grid of clips
        clip_grid = []
        for i in range(0, num_clips, grid_size):
            clip_row = video_clips[i : i + grid_size]
            # Pad the row with empty clips if necessary
            while len(clip_row) < grid_size:
                empty_clip = ColorClip(
                    size=(video_clips[0].w, video_clips[0].h),
                    color=(0, 0, 0),
                    duration=video_clips[0].duration,
                )
                clip_row.append(empty_clip)
            clip_grid.append(clip_row)

        # Pad the grid with empty rows if necessary
        while len(clip_grid) < grid_size:
            empty_row = [
                ColorClip(
                    size=(video_clips[0].w, video_clips[0].h),
                    color=(0, 0, 0),
                    duration=video_clips[0].duration,
                )
                for _ in range(grid_size)
            ]
            clip_grid.append(empty_row)

        # Stack the clips into a grid
        final_clip = clips_array(clip_grid)

        # Write the final video to the output path
        final_clip.write_videofile(output_path, fps=29)


def detect_boundaries(Fly, threshold1=30, threshold2=100):
    """Detects the start and end of the corridor in the video. This is later used to compute the relative distance of the fly from the start of the corridor.

    Args:
        threshold1 (int, optional): the first threshold for the hysteresis procedure in Canny edge detection. Defaults to 30.
        threshold2 (int, optional): the second threshold for the hysteresis procedure in Canny edge detection. Defaults to 100.

    Returns:
        frame (np.array): the last frame of the video.
        start (int): the start of the corridor.
        end (int): the end of the corridor.
    """

    video_file = Fly.metadata.video

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


def generate_grid(Experiment, preview=False, overwrite=False):

    # Check if the grid image already exists
    if (Experiment.directory / "grid.png").exists() and not overwrite:
        print(f"Grid image already exists for {Experiment.directory.name}")
        return
    else:
        print(f"Generating grid image for {Experiment.directory.name}")

        frames = []
        starts = []
        paths = []

        for fly in Experiment.flies:
            frame, start, end = detect_boundaries(fly)
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
        plt.savefig(Experiment.directory / "grid.png")

        if preview == True:
            plt.show()
