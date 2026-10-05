import matplotlib.pyplot as plt
import numpy as np

class PerformanceProfiler:
    def __init__(self):
        self.cycle_times = []
        self.point_counts = []
        self.frame_indices = []
        self.current_frame = 0

    def record_frame(self, cycle_time_s: float, num_points: int):
        """Stores the metrics for the current frame in memory."""
        self.cycle_times.append(cycle_time_s * 1000)  # Convert to ms
        self.point_counts.append(num_points)
        self.frame_indices.append(self.current_frame)
        self.current_frame += 1

    def plot_results(self):
        """Generates a dual-axis plot and prints the average cycle time."""
        skip_frames = 5
        
        if len(self.frame_indices) <= skip_frames:
            print(f"Not enough data recorded to plot (needs more than {skip_frames} frames).")
            return

        # Slice the arrays to ignore the first 5 frames
        valid_frames = self.frame_indices[skip_frames:]
        valid_times = self.cycle_times[skip_frames:]
        valid_points = self.point_counts[skip_frames:]

        # Calculate and print the average reference time
        avg_time = np.mean(valid_times)
        print(f"\n--- Performance Summary ---")
        print(f"Total Frames Processed : {len(self.frame_indices)}")
        print(f"Frames Ignored (Warmup): {skip_frames}")
        print(f"Average Cycle Time     : {avg_time:.2f} ms ({1000 / avg_time:.1f} FPS)")
        print(f"---------------------------\n")

        # Create the plot
        fig, ax1 = plt.subplots(figsize=(10, 5))

        # Axis 1: Cycle Time (Red) - using marker='o' for points
        ax1.set_xlabel('Frame Index')
        ax1.set_ylabel('Cycle Time (ms)', color='tab:red')
        line1, = ax1.plot(valid_frames, valid_times, color='tab:red', marker='o', markersize=4, linestyle='-', label='Cycle Time (ms)', linewidth=1.5)
        ax1.tick_params(axis='y', labelcolor='tab:red')
        
        # Add a horizontal line for the 33ms (30 FPS) target
        ax1.axhline(y=33.3, color='r', linestyle='--', alpha=0.3, label='30 FPS Target')

        # Axis 2: Number of Points (Blue) - using marker='s' (squares) for points
        ax2 = ax1.twinx()  
        ax2.set_ylabel('Number of Points', color='tab:blue')
        line2, = ax2.plot(valid_frames, valid_points, color='tab:blue', marker='s', markersize=4, linestyle='-', label='Point Count', linewidth=1.5, alpha=0.7)
        ax2.tick_params(axis='y', labelcolor='tab:blue')

        # Formatting
        plt.title('Performance Profile: Cycle Time vs. Point Count (Warmup Excluded)')
        fig.tight_layout()
        
        # Combine legends
        lines = [line1, line2]
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left')

        print("Opening performance plot... Close the window to exit.")
        plt.show()