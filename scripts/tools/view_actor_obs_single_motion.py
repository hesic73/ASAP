import numpy as np
import tyro
from rich.console import Console
from rich.text import Text
from dataclasses import dataclass

# Initialize Rich Console for styled output
console = Console()


def print_actor_obs_v2(obs: np.ndarray):
    """
    Prints the components of a single actor observation with improved readability using rich,
    tailored for the new observation format.

    Args:
        obs (np.ndarray): A 1D numpy array representing a single actor observation.
    """
    if not isinstance(obs, np.ndarray) or obs.ndim != 1:
        console.print(
            "[red]Error: Input observation must be a 1D numpy array.[/red]")
        return

    current_idx = 0

    def print_component(name: str, dim: int, prefix: str = "", level: int = 0):
        nonlocal current_idx
        indent = "   " * level
        if current_idx + dim > len(obs):
            console.print(f"[red]Error: Observation data exhausted before parsing {prefix}{name}. "
                          f"Expected dimension {dim}, but only {len(obs) - current_idx} elements remaining.[/red]")
            # Raise to stop further execution
            raise ValueError("Observation parsing error.")

        value_text = Text(
            str(obs[current_idx: current_idx + dim]), style="green")
        console.print(f"{indent}[cyan bold]{name}:[/cyan bold] ", value_text)
        current_idx += dim

    console.print(
        "\n--- [bold magenta]Parsing Actor Observation (v2)[/bold magenta] ---")

    # Top-level components for the new format
    print_component("actions", 23)
    print_component("base_ang_vel", 3)
    print_component("dof_pos", 23)
    print_component("dof_vel", 23)

    # Handle history_actor with its new internal structure
    console.print("[cyan bold]history_actor:[/cyan bold]")
    history_actor_start_idx = current_idx
    # Corrected: Calculate total dimension for history_actor based on num_frames * single_frame_dim
    num_frames = 4
    history_base_ang_vel_dim = 3 * num_frames
    history_projected_gravity_dim = 3 * num_frames
    history_dof_pos_dim = 23 * num_frames
    history_dof_vel_dim = 23 * num_frames
    history_actions_dim = 23 * num_frames
    history_ref_motion_phase_dim = 1 * num_frames

    history_actor_total_dim = (
        history_base_ang_vel_dim +
        history_projected_gravity_dim +
        history_dof_pos_dim +
        history_dof_vel_dim +
        history_actions_dim +
        history_ref_motion_phase_dim
    )

    if current_idx + history_actor_total_dim > len(obs):
        console.print(f"[red]Error: Observation data exhausted before parsing history_actor. "
                      f"Expected total dimension {history_actor_total_dim}, but only {len(obs) - current_idx} elements remaining.[/red]")
        # Raise to stop further execution
        raise ValueError("History_actor parsing error.")

    history_obs_slice = obs[history_actor_start_idx:
                            history_actor_start_idx + history_actor_total_dim]
    temp_current_idx_for_history = 0

    def print_history_sub_component(name: str, dim: int):
        nonlocal temp_current_idx_for_history
        indent = "   " * 1  # One level deeper for sub-components
        if temp_current_idx_for_history + dim > len(history_obs_slice):
            console.print(f"[red]Error: History_actor data exhausted before parsing sub-component {name}. "
                          f"Expected dimension {dim}, but only {len(history_obs_slice) - temp_current_idx_for_history} elements remaining.[/red]")
            raise ValueError("History sub-component parsing error.")

        value_text = Text(str(
            history_obs_slice[temp_current_idx_for_history: temp_current_idx_for_history + dim]), style="yellow")
        console.print(f"{indent}[blue]{name}:[/blue] ", value_text)
        temp_current_idx_for_history += dim

    # New history_actor sub-components with corrected dimensions
    print_history_sub_component("base_ang_vel", history_base_ang_vel_dim)
    print_history_sub_component(
        "projected_gravity", history_projected_gravity_dim)
    print_history_sub_component("dof_pos", history_dof_pos_dim)
    print_history_sub_component("dof_vel", history_dof_vel_dim)
    print_history_sub_component("actions", history_actions_dim)
    print_history_sub_component(
        "ref_motion_phase", history_ref_motion_phase_dim)

    current_idx += history_actor_total_dim  # Advance the main index

    # Remaining top-level components
    print_component("projected_gravity", 3)
    print_component("ref_motion_phase", 1)  # New top-level component

    if current_idx != len(obs):
        console.print(
            f"[orange3]Warning: Not all observation components were parsed. Remaining elements: {len(obs) - current_idx}[/orange3]")
    console.print("--- [bold magenta]End Parsing (v2)[/bold magenta] ---")


@dataclass
class Args:
    """
    Configuration for the observation parsing script.
    """
    filename: str
    """Path to the eval_trajectory.npz file."""


def main():
    args = tyro.cli(Args)  # Parse arguments from the command line

    try:
        data = np.load(args.filename)
        observations = data['observations']
        actions = data['actions']
        console.print(
            f"[bold]Observations shape:[/bold] [green]{observations.shape}[/green]")
        console.print(
            f"[bold]Actions shape:[/bold] [green]{actions.shape}[/green]")

        # Example usage: Print the first observation from the first trajectory
        if observations.shape[0] > 0 and observations.shape[1] > 0:
            console.print(
                "\n[bold]--- Details for the First Actor Observation ---[/bold]")
            first_obs = observations[0, 0, :]
            print_actor_obs_v2(first_obs)  # Use the new parsing function
        else:
            console.print(
                "[orange3]No observations available in the loaded data.[/orange3]")

    except FileNotFoundError:
        console.print(
            f"[red]Error: The file '{args.filename}' was not found. Please check the path.[/red]")
    except Exception as e:
        console.print(f"[red]An unexpected error occurred: {e}[/red]")


if __name__ == "__main__":
    main()
