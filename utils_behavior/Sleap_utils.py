import h5py

import pandas as pd

import cv2

# import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import random


from pathlib import Path

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor

import yaml

import subprocess


class Sleap_Tracks:
    """Class for handling SLEAP tracking data. It is a wrapper around the SLEAP H5 file format."""

    class Object:
        """Nested class to represent an object with node properties."""

        def __init__(self, object_data, node_names):
            self.node_names = node_names
            self.object_data = object_data

            for node in node_names:
                setattr(self, node, self._create_node_property(node))

        def _create_node_property(self, node):
            """Creates a property for a node to access its x and y coordinates for each frame.

            Args:
                node (str): The name of the node.

            Returns:
                property: A property for the node coordinates.
            """

            @property
            def node_property(self):
                x_values = self.object_data[f"x_{node}"].values
                y_values = self.object_data[f"y_{node}"].values
                return list(zip(x_values, y_values))

            return node_property

    def __init__(self, filename, object_type="object"):
        """Initialize the Sleap_Track object with the given SLEAP tracking file.

        Args:
            filename (Path): Path to the SLEAP tracking file.
            object_type (str): Type of the object (e.g., "ball", "fly"). Defaults to "object".
        """

        self.path = filename
        self.object_type = object_type

        # Open the SLEAP tracking file
        self.h5file = h5py.File(filename, "r")

        self.node_names = [x.decode("utf-8") for x in self.h5file["node_names"]]
        self.edge_names = [
            [y.decode("utf-8") for y in x] for x in self.h5file["edge_names"]
        ]
        self.edges_idx = self.h5file["edge_inds"]

        self.tracks = self.h5file["tracks"][:]

        self.video = Path(self.h5file["video_path"][()].decode("utf-8"))

        # Try to load the video file to check its accessibility and get fps
        try:
            cap = cv2.VideoCapture(str(self.video))
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {self.video}")
            
            self.fps = cap.get(cv2.CAP_PROP_FPS)
            self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
        except Exception as e:
            print(f"Video file not available: {self.video}. Check path and server access. Error: {e}")
            self.fps = None  # Set fps to None if video is not accessible
            
        self.dataset = self.generate_tracks_data()

        # Create object properties
        self.objects = []
        for i in range(len(self.tracks)):
            object_data = self.dataset[self.dataset["object"] == f"{self.object_type}_{i+1}"]
            self.objects.append(self.Object(object_data, self.node_names))

        print(f"Loaded SLEAP tracking file: {filename}")
        print(f"N° of objects: {len(self.objects)}")
        print(f"Nodes: {self.node_names}")
        print(f"Video FPS: {self.fps}")

    def generate_tracks_data(self):
        """Generates a pandas DataFrame with the tracking data, with the following columns:
        - frame
        for each node:
        - node_x
        - node_y

        The shape of the tracks is instance, x or y, nodes and frame and the order of the nodes is the same as the one in the nodes attribute.

        Returns:
            DataFrame: DataFrame with the tracking data.
        """

        df_list = []

        for i, obj in enumerate(self.tracks):
            
            print(f"Processing {self.object_type} {i+1}/{len(self.tracks)}")

            x_coords = obj[0]
            y_coords = obj[1]

            frames = range(1, len(obj[0][0]) + 1)

            tracking_df = pd.DataFrame(frames, columns=["frame"])
            
            tracking_df["time"] = tracking_df["frame"] / self.fps

            # Give each object some number
            tracking_df["object"] = f"{self.object_type}_{i+1}"

            for k, n in enumerate(self.node_names):
                tracking_df[f"x_{n}"] = x_coords[k]
                tracking_df[f"y_{n}"] = y_coords[k]

            df_list.append(tracking_df)

        df = pd.concat(df_list, ignore_index=True)

        return df

    def generate_annotated_frame(self, frame, nodes=None, labels=False, edges=True, colorby=None):
        """Generates an annotated frame image for a specific frame."""
        frame_data = self.dataset[self.dataset["frame"] == frame]

        cap = cv2.VideoCapture(str(self.video))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame - 1)

        ret, img = cap.read()
        if not ret:
            raise ValueError(f"Could not read frame {frame} from video {self.video}")

        if nodes is None:
            nodes = self.node_names
        elif isinstance(nodes, str):
            nodes = [nodes]

        # Define colors
        if colorby == 'Nodes':
            color_map = {node: (int(np.random.randint(0, 255)), int(np.random.randint(0, 255)), int(np.random.randint(0, 255))) for node in self.node_names}
        else:
            color_map = {node: (0, 255, 255) for node in self.node_names}

        for _, row in frame_data.iterrows():
            for node in nodes:
                x = row[f"x_{node}"]
                y = row[f"y_{node}"]
                if not np.isnan(x) and not np.isnan(y):
                    x, y = int(x), int(y)
                    cv2.circle(img, (x, y), 2, color_map[node], -1)
                    if labels:
                        cv2.putText(
                            img,
                            node,
                            (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 255),
                            1,
                            cv2.LINE_AA,
                        )

        if edges:
            for edge in self.edge_names:
                node1, node2 = edge
                if node1 in nodes and node2 in nodes:
                    x1 = frame_data[f"x_{node1}"].values[0]
                    y1 = frame_data[f"y_{node1}"].values[0]
                    x2 = frame_data[f"x_{node2}"].values[0]
                    y2 = frame_data[f"y_{node2}"].values[0]
                    if not (
                        np.isnan(x1) or np.isnan(y1) or np.isnan(x2) or np.isnan(y2)
                    ):
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        cv2.line(img, (x1, y1), (x2, y2), color_map[node2], 1)

        cap.release()
        return img

    def cpu_generate_annotated_video(
        self,
        save=False,
        output_path=None,
        start=None,
        end=None,
        nodes=None,
        labels=False,
        edges=True,
        colorby=None,
    ):
        cap = cv2.VideoCapture(str(self.video))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if start is None:
            start = 1
        if end is None:
            end = total_frames

        if save:
            if output_path is None:
                output_path = self.video.with_suffix(".annotated.mp4")
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        def process_frame(frame):
            return self.generate_annotated_frame(
                frame, nodes=nodes, labels=labels, edges=edges, colorby=colorby
            )

        with ThreadPoolExecutor() as executor:
            annotated_frames = list(
                tqdm(
                    executor.map(process_frame, range(start, end + 1)),
                    total=end - start + 1,
                    desc="Processing frames",
                )
            )

        for annotated_frame in annotated_frames:
            if save:
                out.write(annotated_frame)
            else:
                cv2.imshow("Annotated Video", annotated_frame)
                if cv2.waitKey(25) & 0xFF == ord("q"):
                    break

        cap.release()
        if save:
            out.release()
        else:
            cv2.destroyAllWindows()

    def gpu_generate_annotated_video(
        self,
        save=False,
        output_path=None,
        start=None,
        end=None,
        nodes=None,
        labels=False,
        edges=True,
        colorby=None,
    ):
        """Generates a video with GPU-accelerated annotated frames."""

        cap = cv2.VideoCapture(str(self.video))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if start is None:
            start = 1
        if end is None:
            end = total_frames

        if save:
            if output_path is None:
                output_path = self.video.with_suffix(".annotated.mp4")
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        # Define colors
        if colorby == 'Nodes':
            color_map = {node: (int(np.random.randint(0, 255)), int(np.random.randint(0, 255)), int(np.random.randint(0, 255))) for node in self.node_names}
        else:
            color_map = {node: (0, 255, 255) for node in self.node_names}

        while cap.isOpened():
            ret, img = cap.read()
            if not ret or cap.get(cv2.CAP_PROP_POS_FRAMES) > end:
                break

            # Upload frame to GPU
            gpu_img = cv2.cuda_GpuMat()
            gpu_img.upload(img)

            frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            frame_data = self.dataset[self.dataset["frame"] == frame]

            if nodes is None:
                nodes = self.node_names

            # Annotate nodes and labels
            for _, row in frame_data.iterrows():
                for node in nodes:
                    x = row[f"x_{node}"]
                    y = row[f"y_{node}"]
                    if not np.isnan(x) and not np.isnan(y):
                        x, y = int(x), int(y)
                        img = gpu_img.download()  # Download to CPU for annotations
                        cv2.circle(img, (x, y), 2, color_map[node], -1)
                        if labels:
                            cv2.putText(
                                img,
                                node,
                                (x, y),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (255, 255, 255),
                                1,
                                cv2.LINE_AA,
                            )
                        gpu_img.upload(img)  # Upload back to GPU

            # Annotate edges if needed
            if edges:
                for edge in self.edge_names:
                    node1, node2 = edge
                    if node1 in nodes and node2 in nodes:
                        x1 = frame_data[f"x_{node1}"].values[0]
                        y1 = frame_data[f"y_{node1}"].values[0]
                        x2 = frame_data[f"x_{node2}"].values[0]
                        y2 = frame_data[f"y_{node2}"].values[0]
                        if not (
                            np.isnan(x1) or np.isnan(y1) or np.isnan(x2) or np.isnan(y2)
                        ):
                            img = gpu_img.download()  # CPU operations for drawing
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                            cv2.line(img, (x1, y1), (x2, y2), color_map[node2], 1)
                            gpu_img.upload(img)  # Back to GPU

            if save:
                out.write(gpu_img.download())
            else:
                cv2.imshow("Annotated Video", gpu_img.download())
                if cv2.waitKey(25) & 0xFF == ord("q"):
                    break

        cap.release()
        if save:
            out.release()
        else:
            cv2.destroyAllWindows()
            
    def generate_annotated_video(self, 
            save=False,
            output_path=None,
            start=None,
            end=None,
            nodes=None,
            labels=False,
            edges=True,
            colorby=None):
        """Generates an annotated video with tracked nodes and edges.

        Args:
            save (bool, optional): Whether to save the video. Defaults to False.
            output_path (str, optional): Path to save the video. Defaults to None.
            start (int, optional): Start frame for the video. Defaults to None.
            end (int, optional): End frame for the video. Defaults to None.
            nodes (list, optional): List of nodes to annotate. Defaults to None.
            labels (bool, optional): Whether to include labels. Defaults to False.
            edges (bool, optional): Whether to include edges. Defaults to True.
            colorby (str, optional): Attribute to color by. Defaults to None.

        Returns:
            None
        """
        # Check if OpenCV has CUDA support
        try:
            if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                print("CUDA is enabled, using GPU for video processing.")
                self.gpu_generate_annotated_video(save=save, output_path=output_path, start=start, end=end, nodes=nodes, labels=labels, edges=edges, colorby=colorby)
            else:
                print("No CUDA devices found, falling back to CPU processing.")
                self.cpu_generate_annotated_video(save=save, output_path=output_path, start=start, end=end, nodes=nodes, labels=labels, edges=edges, colorby=colorby)
        except cv2.error as e:
            print(f"Error while checking for CUDA support: {e}")
            print("Falling back to CPU processing.")
            self.cpu_generate_annotated_video(save=save, output_path=output_path, start=start, end=end, nodes=nodes, labels=labels, edges=edges, colorby=colorby)
            
class CombinedSleapTracks:
    """Class for handling and combining multiple SLEAP Tracks for the same video."""

    def __init__(self, video_path, sleap_tracks_list):
        """Initializes the CombinedSleapTracks object.

        Args:
            video_path (Path): Path to the video file.
            sleap_tracks_list (list): List of Sleap_Tracks objects.
        """
        self.video_path = video_path
        self.sleap_tracks_list = sleap_tracks_list
        self.dataset = self.generate_dataset()
        self.colors = self._assign_colors()

    def _assign_colors(self):
        """Assign random colors to each Sleap_Tracks object for distinct annotation."""
        colors = {}
        for idx, tracks in enumerate(self.sleap_tracks_list):
            # Assign a random color for each Sleap_Tracks object (BGR format)
            color = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255),
            )
            colors[tracks] = color
        return colors
    
    def generate_dataset(self):
        """Generates a combined dataset from multiple Sleap_Tracks objects."""
        combined_dataset = pd.concat([tracks.dataset for tracks in self.sleap_tracks_list], ignore_index=True)
        return combined_dataset

    def cpu_generate_combined_annotated_video(
        self, 
        save=False, 
        output_path=None, 
        start=None, 
        end=None, 
        labels=False, 
        edges=True
    ):
        """Generates a combined annotated video from multiple Sleap_Tracks objects (CPU version).
        
        Args:
            video_path (Path): Path to the video file.
            sleap_tracks_list (list): List of Sleap_Tracks objects.
            save (bool): Whether to save the video. Defaults to False.
            output_path (str): Path to save the video. Defaults to None.
            start (int): Start frame. Defaults to None.
            end (int): End frame. Defaults to None.
            labels (bool): Whether to include labels. Defaults to False.
            edges (bool): Whether to include edges. Defaults to True.
        
        Returns:
            None
        """
        
        video_path = self.video_path
        sleap_tracks_list = self.sleap_tracks_list
        
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if start is None:
            start = 1
        if end is None:
            end = total_frames

        # Define a list of colors for each Sleap_Tracks object
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]  # Add more as needed

        if save:
            if output_path is None:
                output_path = video_path.with_suffix(".combined_annotated.mp4")
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        def process_frame(frame):
            ret, img = cap.read()
            if not ret:
                return None
            
            # Loop through each Sleap_Tracks object and annotate
            for idx, tracks in enumerate(sleap_tracks_list):
                frame_data = tracks.dataset[tracks.dataset["frame"] == frame]
                color = colors[idx % len(colors)]

                for _, row in frame_data.iterrows():
                    for node in tracks.node_names:
                        x = row[f"x_{node}"]
                        y = row[f"y_{node}"]
                        if not np.isnan(x) and not np.isnan(y):
                            x, y = int(x), int(y)
                            cv2.circle(img, (x, y), 2, color, -1)
                            if labels:
                                cv2.putText(
                                    img, node, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA
                                )

                if edges:
                    for edge in tracks.edge_names:
                        node1, node2 = edge
                        x1 = row[f"x_{node1}"]
                        y1 = row[f"y_{node1}"]
                        x2 = row[f"x_{node2}"]
                        y2 = row[f"y_{node2}"]
                        if not np.isnan(x1) and not np.isnan(y1) and not np.isnan(x2) and not np.isnan(y2):
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                            cv2.line(img, (x1, y1), (x2, y2), color, 1)

            return img

        for frame in tqdm(range(start, end + 1), desc="Processing frames"):
            annotated_frame = process_frame(frame)
            if annotated_frame is None:
                break
            if save:
                out.write(annotated_frame)
            else:
                cv2.imshow("Annotated Video", annotated_frame)
                if cv2.waitKey(25) & 0xFF == ord("q"):
                    break

        cap.release()
        if save:
            out.release()
        else:
            cv2.destroyAllWindows()
            
    def gpu_generate_combined_annotated_video(
        self,
        save=False, 
        output_path=None, 
        start=None, 
        end=None, 
        labels=False, 
        edges=True
    ):
        """Generates a combined annotated video from multiple Sleap_Tracks objects (GPU version).
        
        Args:
            video_path (Path): Path to the video file.
            sleap_tracks_list (list): List of Sleap_Tracks objects.
            save (bool): Whether to save the video. Defaults to False.
            output_path (str): Path to save the video. Defaults to None.
            start (int): Start frame. Defaults to None.
            end (int): End frame. Defaults to None.
            labels (bool): Whether to include labels. Defaults to False.
            edges (bool): Whether to include edges. Defaults to True.
        
        Returns:
            None
        """
        
        video_path = self.video_path
        sleap_tracks_list = self.sleap_tracks_list
        
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if start is None:
            start = 1
        if end is None:
            end = total_frames

        # Define colors for each Sleap_Tracks object
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]  # Extend as needed

        if save:
            if output_path is None:
                output_path = video_path.with_suffix(".combined_annotated.mp4")
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        while cap.isOpened():
            ret, img = cap.read()
            if not ret or cap.get(cv2.CAP_PROP_POS_FRAMES) > end:
                break

            # Upload frame to GPU
            gpu_img = cv2.cuda_GpuMat()
            gpu_img.upload(img)

            frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

            for idx, tracks in enumerate(sleap_tracks_list):
                frame_data = tracks.dataset[tracks.dataset["frame"] == frame]
                color = colors[idx % len(colors)]

                # Annotate nodes and labels
                for _, row in frame_data.iterrows():
                    for node in tracks.node_names:
                        x = row[f"x_{node}"]
                        y = row[f"y_{node}"]
                        if not np.isnan(x) and not np.isnan(y):
                            x, y = int(x), int(y)
                            img = gpu_img.download()  # Download to CPU for annotations
                            cv2.circle(img, (x, y), 2, color, -1)
                            if labels:
                                cv2.putText(
                                    img, node, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA
                                )
                            gpu_img.upload(img)  # Upload back to GPU

                # Annotate edges
                if edges:
                    for edge in tracks.edge_names:
                        node1, node2 = edge
                        x1 = row[f"x_{node1}"]
                        y1 = row[f"y_{node1}"]
                        x2 = row[f"x_{node2}"]
                        y2 = row[f"y_{node2}"]
                        if not np.isnan(x1) and not np.isnan(y1) and not np.isnan(x2) and not np.isnan(y2):
                            img = gpu_img.download()  # CPU operations for drawing
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                            cv2.line(img, (x1, y1), (x2, y2), color, 1)
                            gpu_img.upload(img)  # Back to GPU

            if save:
                out.write(gpu_img.download())
            else:
                cv2.imshow("Annotated Video", gpu_img.download())
                if cv2.waitKey(25) & 0xFF == ord("q"):
                    break

        cap.release()
        if save:
            out.release()
        else:
            cv2.destroyAllWindows()
            
    def generate_combined_annotated_video(
        self,
        save=False,
        output_path=None,
        start=None,
        end=None,
        labels=False,
        edges=True,
    ):
        
        try:
            if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                print("CUDA is enabled, using GPU for video processing.")
                self.gpu_generate_combined_annotated_video(save=save, output_path=output_path, start=start, end=end, labels=labels, edges=edges)
            else:
                print("No CUDA devices found, falling back to CPU processing.")
                self.cpu_generate_combined_annotated_video(save=save, output_path=output_path, start=start, end=end, labels=labels, edges=edges)
        except cv2.error as e:
            print(f"Error while checking for CUDA support: {e}")
            print("Falling back to CPU processing.")
            self.cpu_generate_combined_annotated_video(save=save, output_path=output_path, start=start, end=end, labels=labels, edges=edges)
            
class SleapTracker:
    def __init__(self, model_path, data_folder=None, model_centered_instance_path=None, output_folder=None, conda_env='sleap', batch_size=16, max_tracks=None, tracker='simple', video_filter=None, yaml_file=None):
        """
        Initialize the SleapTracker class.

        Args:
            model_path (str or pathlib.Path): Path to the trained SLEAP model for tracking. If multiple models are needed, this one should be the centroid model.
            data_folder (str or pathlib.Path, optional): Directory containing videos to track. This will be ignored if a YAML file is provided.
            model_centered_instance_path (str or pathlib.Path, optional): Path to the centered instance model for tracking.
            output_folder (str or pathlib.Path, optional): Directory to store tracked .slp files. Defaults to data_folder.
            conda_env (str, optional): Conda environment where SLEAP is installed. Defaults to 'sleap'.
            batch_size (int, optional): Number of frames to process at once in sleap-track command. Defaults to 16.
            max_tracks (int, optional): The maximum number of tracks to predict. Set to None for unlimited.
            tracker (str, optional): The type of tracker to use (e.g., 'simple', 'flow', etc.). Defaults to 'simple'.
            video_filter (str, optional): Optional filter for videos to process.
            yaml_file (str or pathlib.Path, optional): Path to a YAML file containing a list of directories to process instead of data_folder.
            
        Example usage:
            tracker = SleapTracker(model_path="path/to/model", data_folder="path/to/videos", batch_size=16)
            tracker.run()

        If you have a YAML file with folders:
            tracker = SleapTracker(model_path="path/to/model", yaml_file="path/to/config.yaml")
            tracker.run()
            
            example YAML file:
            directories:
              - path/to/videos1
              - path/to/videos2
        """
        self.model_path = Path(model_path)
        self.model_centered_instance = Path(model_centered_instance_path) if model_centered_instance_path else None
        self.data_folder = Path(data_folder) if data_folder else None
        self.output_folder = Path(output_folder) if output_folder else self.data_folder
        self.conda_env = conda_env
        self.batch_size = batch_size
        self.max_tracks = max_tracks if max_tracks else None
        self.tracker = tracker
        self.videos_to_process = []
        self.video_filter = video_filter if video_filter else None
        self.yaml_file = Path(yaml_file) if yaml_file else None

    def load_directories_from_yaml(self):
        """
        Load directories from a YAML file if provided.
        """
        if self.yaml_file:
            with open(self.yaml_file, 'r') as file:
                directories = yaml.safe_load(file).get('directories', [])
                if directories:
                    print(f"Loaded {len(directories)} directories from YAML file.")
                    return [Path(d) for d in directories]
        return []

    def collect_videos(self, video_extension=".mp4"):
        """
        Collect all videos from the data folder(s) that need tracking.

        If a YAML file is provided, directories from the YAML file will be used instead of data_folder.
        """
        directories_to_process = self.load_directories_from_yaml() if self.yaml_file else [self.data_folder]
        
        for folder in directories_to_process:
            if folder.exists() and folder.is_dir():
                self.videos_to_process.extend(list(folder.rglob(f'*{video_extension}')))
        print(f"Collected {len(self.videos_to_process)} videos for tracking.")

    def filter_tracked_videos(self):
        """
        Filter out videos that have already been tracked (i.e., videos with corresponding .slp or .h5 files).
        """
        videos_filtered = []
        for video in self.videos_to_process:
            video_name = video.stem
            slp_file = self.output_folder / f"{video_name}_tracked.slp"
            h5_file = self.output_folder / f"{video_name}_tracked.h5"
            if not slp_file.exists() and not h5_file.exists():
                videos_filtered.append(video)

        self.videos_to_process = videos_filtered
        print(f"Filtered to {len(self.videos_to_process)} videos needing tracking.")

    def activate_conda(self):
        """
        Activate the conda environment where SLEAP is installed.
        
        Example usage:
            tracker.activate_conda()
        
        This is necessary to ensure the correct environment is used for tracking.
        """
        subprocess.run(['conda', 'activate', self.conda_env], check=True)

    def process_videos(self):
        """
        Process videos by running the SLEAP tracking command.

        Example usage:
            tracker.process_videos()
        """
        print(f"Processing {len(self.videos_to_process)} videos...")

        for video in self.videos_to_process:
            # Build the sleap-track command for tracking
            sleap_track_cmd = [
                "sleap-track",
                str(video),
                "--model", str(self.model_path),
                "--output", str(self.output_folder / f"{video.stem}_tracked.slp"),
                "--batch_size", str(self.batch_size),
                # Add tracker if a centered instance model is specified
                *(["--tracking.tracker", self.tracker] if self.model_centered_instance else []),
                # Add max_tracks and set max_tracking if specified
                *(["--max_tracking", "1"] if self.max_tracks else []),
                *(["--max_tracks", str(self.max_tracks)] if self.max_tracks else []),
                "--verbosity", "rich"
            ]
            if self.model_centered_instance:
                sleap_track_cmd.extend(["--model", str(self.model_centered_instance)])

            subprocess.run(sleap_track_cmd, check=True)

            # Convert the .slp file to .h5 format for analysis
            sleap_convert_cmd = [
                "sleap-convert",
                str(self.output_folder / f"{video.stem}_tracked.slp"),
                "--format", "analysis"
            ]
            subprocess.run(sleap_convert_cmd, check=True)

        print("Video processing complete.")

    def run(self, video_extension=".mp4", render=False):
        """
        Main method to collect videos, filter tracked ones, and track remaining videos.

        Example usage:
            tracker.run()
        """
        self.collect_videos(video_extension=video_extension)
        self.filter_tracked_videos()
        if self.videos_to_process:
            self.process_videos()
            
            if render:
                sleap_tracks = Sleap_Tracks(self.output_folder / f"{self.videos_to_process[0].stem}_tracked.h5")
                
                sleap_tracks.generate_annotated_video(save=True)
        else:
            print("No new videos to track.")
