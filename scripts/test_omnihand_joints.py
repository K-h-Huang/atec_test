"""Cycle through OmniHand Pro joints and return each one to its default pose."""

import argparse
import math

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Test each OmniHand Pro joint in sequence.")
parser.add_argument("--joint_angle_deg", type=float, default=20.0, help="Requested joint rotation in degrees.")
parser.add_argument("--move_time", type=float, default=1.0, help="Seconds used to move to or from the target.")
parser.add_argument("--hold_time", type=float, default=1.0, help="Seconds to hold the rotated joint.")
parser.add_argument("--restore_time", type=float, default=0.5, help="Seconds to hold the restored pose.")
parser.add_argument("--cycles", type=int, default=0, help="Number of full cycles. Zero repeats until the app closes.")
parser.add_argument("--free_base", action="store_true", help="Leave the palm base free instead of fixing it in space.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass

from atec_rl_lab.assets.robots.omnihand_pro import OMNIHAND_ACTION_JOINT_NAMES, OMNIHAND_CFG


@configclass
class OmniHandJointTestSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )
    robot = OMNIHAND_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=OMNIHAND_CFG.spawn.replace(fix_base=not args_cli.free_base),
        init_state=OMNIHAND_CFG.init_state.replace(pos=(0.0, 0.0, 0.01)),
    )


def step_target(robot, scene, sim, target: torch.Tensor, duration: float) -> bool:
    """Hold a position target for a duration while advancing the simulation."""
    step_dt = sim.get_physics_dt()
    num_steps = max(1, round(duration / step_dt))
    for _ in range(num_steps):
        if not simulation_app.is_running():
            return False
        robot.set_joint_position_target(target)
        scene.write_data_to_sim()
        sim.step()
        scene.update(step_dt)
    return True


def move_target(robot, scene, sim, start: torch.Tensor, end: torch.Tensor, duration: float) -> bool:
    """Linearly interpolate between two joint position targets."""
    step_dt = sim.get_physics_dt()
    num_steps = max(1, round(duration / step_dt))
    for step in range(num_steps):
        if not simulation_app.is_running():
            return False
        alpha = (step + 1) / num_steps
        target = torch.lerp(start, end, alpha)
        robot.set_joint_position_target(target)
        scene.write_data_to_sim()
        sim.step()
        scene.update(step_dt)
    return True


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([0.35, 0.35, 0.25], [0.0, 0.0, 0.05])

    scene_cfg = OmniHandJointTestSceneCfg(num_envs=1, env_spacing=1.0, replicate_physics=True)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    scene.reset()

    robot = scene.articulations["robot"]
    joint_ids, joint_names = robot.find_joints(list(OMNIHAND_ACTION_JOINT_NAMES), preserve_order=True)
    default_target = robot.data.default_joint_pos.clone()
    robot.set_joint_position_target(default_target)

    print("-" * 100)
    print("OmniHand Pro joint test")
    print("Controlled joints:", joint_names)
    print("Requested angle:", args_cli.joint_angle_deg, "degrees")
    print("Base mode:", "free" if args_cli.free_base else "fixed")

    if not step_target(robot, scene, sim, default_target, 1.0):
        return

    requested_angle = math.radians(abs(args_cli.joint_angle_deg))
    completed_cycles = 0

    while simulation_app.is_running() and (args_cli.cycles <= 0 or completed_cycles < args_cli.cycles):
        for joint_id, joint_name in zip(joint_ids, joint_names):
            if not simulation_app.is_running():
                return

            lower = robot.data.soft_joint_pos_limits[0, joint_id, 0].item()
            upper = robot.data.soft_joint_pos_limits[0, joint_id, 1].item()
            default_position = default_target[0, joint_id].item()
            positive_range = max(0.0, upper - default_position)
            negative_range = max(0.0, default_position - lower)

            if positive_range >= negative_range:
                direction = 1.0
                available_range = positive_range
            else:
                direction = -1.0
                available_range = negative_range

            actual_angle = min(requested_angle, 0.8 * available_range)
            if actual_angle <= 1.0e-5:
                print(f"Skipping {joint_name}: no available motion from the default position.")
                continue

            rotated_target = default_target.clone()
            rotated_target[:, joint_id] = default_position + direction * actual_angle
            signed_angle_deg = math.degrees(direction * actual_angle)
            print(f"Moving {joint_name}: {default_position:.3f} rad -> {rotated_target[0, joint_id]:.3f} rad "
                  f"({signed_angle_deg:+.1f} deg)")

            if not move_target(robot, scene, sim, default_target, rotated_target, args_cli.move_time):
                return
            if not step_target(robot, scene, sim, rotated_target, args_cli.hold_time):
                return
            if not move_target(robot, scene, sim, rotated_target, default_target, args_cli.move_time):
                return
            if not step_target(robot, scene, sim, default_target, args_cli.restore_time):
                return

        completed_cycles += 1
        print(f"Completed cycle {completed_cycles}.")


if __name__ == "__main__":
    main()
    simulation_app.close()
