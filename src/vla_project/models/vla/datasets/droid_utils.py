"""Episode transforms for DROID dataset."""

# import tensorflow_graphics.geometry.transformation as tfg
import torch
from jaxtyping import Float


def _index_from_letter(letter: str) -> int:
    """Convert axis letter to corresponding index.

    Args:
        letter: Axis letter, must be "X", "Y", or "Z".

    Returns:
        Index corresponding to the axis (0 for X, 1 for Y, 2 for Z).

    Raises:
        ValueError: If letter is not X, Y, or Z.

    """
    if letter == "X":
        return 0
    if letter == "Y":
        return 1
    if letter == "Z":
        return 2
    msg = "letter must be either X, Y or Z."
    raise ValueError(msg)


def _angle_from_tan(axis: str, other_axis: str, data, horizontal: bool, tait_bryan: bool) -> torch.Tensor:
    """Extract the first or third Euler angle from the two members of the matrix which are positive constant times its sine and cosine.

    Args:
        axis: Axis label "X" or "Y or "Z" for the angle we are finding.
        other_axis: Axis label "X" or "Y or "Z" for the middle axis in the
            convention.
        data: Rotation matrices as tensor of shape (..., 3, 3).
        horizontal: Whether we are looking for the angle for the third axis,
            which means the relevant entries are in the same row of the
            rotation matrix. If not, they are in the same column.
        tait_bryan: Whether the first and third axes in the convention differ.

    Returns:
        Euler Angles in radians for each matrix in data as a tensor
        of shape (...).

    """
    i1, i2 = {"X": (2, 1), "Y": (0, 2), "Z": (1, 0)}[axis]
    if horizontal:
        i2, i1 = i1, i2
    even = (axis + other_axis) in ["XY", "YZ", "ZX"]
    if horizontal == even:
        return torch.atan2(data[..., i1], data[..., i2])
    if tait_bryan:
        return torch.atan2(-data[..., i2], data[..., i1])
    return torch.atan2(data[..., i2], -data[..., i1])


def _axis_angle_rotation(axis: str, angle: torch.Tensor) -> torch.Tensor:
    """Return the rotation matrices for one of the rotations about an axis of which Euler angles describe, for each value of the angle given.

    Args:
        axis: Axis label "X" or "Y or "Z".
        angle: any shape tensor of Euler angles in radians

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).

    """
    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    else:
        msg = "letter must be either X, Y or Z."
        raise ValueError(msg)

    return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))


def matrix_to_euler_angles(
    matrix: Float[torch.Tensor, ..., 3, 3], convention: str = "XYZ",
) -> Float[torch.Tensor, ..., 3]:
    """Convert rotations given as rotation matrices to Euler angles in radians.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).
        convention: Convention string of three uppercase letters.

    Returns:
        Euler angles in radians as tensor of shape (..., 3).

    """
    if len(convention) != 3:
        msg = "Convention must have 3 letters."
        raise ValueError(msg)
    if convention[1] in (convention[0], convention[2]):
        msg = f"Invalid convention {convention}."
        raise ValueError(msg)
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            msg = f"Invalid letter {letter} in convention string."
            raise ValueError(msg)
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        msg = f"Invalid rotation matrix shape {matrix.shape}."
        raise ValueError(msg)
    i0 = _index_from_letter(convention[0])
    i2 = _index_from_letter(convention[2])
    tait_bryan = i0 != i2
    if tait_bryan:
        central_angle = torch.asin(torch.clamp(matrix[..., i0, i2], -1.0, 1.0) * (-1.0 if i0 - i2 in [-1, 2] else 1.0))
    else:
        central_angle = torch.acos(torch.clamp(matrix[..., i0, i0], -1.0, 1.0))

    o = (
        _angle_from_tan(convention[0], convention[1], matrix[..., i2], False, tait_bryan),
        central_angle,
        _angle_from_tan(convention[2], convention[1], matrix[..., i0, :], True, tait_bryan),
    )
    return torch.stack(o, -1)


def euler_angles_to_matrix(
    euler_angles: Float[torch.Tensor, ..., 3],
    convention: str = "XYZ",
) -> Float[torch.Tensor, ..., 3, 3]:
    """Convert rotations given as Euler angles in radians to rotation matrices.

    Args:
        euler_angles: Euler angles in radians as tensor of shape (..., 3).
        convention: Convention string of three uppercase letters from
            {"X", "Y", and "Z"}.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).

    """
    if euler_angles.dim() == 0 or euler_angles.shape[-1] != 3:
        msg = "Invalid input euler angles."
        raise ValueError(msg)
    if len(convention) != 3:
        msg = "Convention must have 3 letters."
        raise ValueError(msg)
    if convention[1] in (convention[0], convention[2]):
        msg = f"Invalid convention {convention}."
        raise ValueError(msg)
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            msg = f"Invalid letter {letter} in convention string."
            raise ValueError(msg)
    matrices = [_axis_angle_rotation(c, e) for c, e in zip(convention, torch.unbind(euler_angles, -1), strict=False)]
    # return functools.reduce(torch.matmul, matrices)
    return torch.matmul(torch.matmul(matrices[0], matrices[1]), matrices[2])


def inverse_rotation_matrix(rotation_matrix: Float[torch.Tensor, ..., 3, 3]) -> Float[torch.Tensor, ..., 3, 3]:
    """Compute the inverse of a rotation matrix.

    Args:
        rotation_matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        Inverse rotation matrices as tensor of shape (..., 3, 3).

    """
    if rotation_matrix.size(-1) != 3 or rotation_matrix.size(-2) != 3:
        msg = f"Invalid rotation matrix shape {rotation_matrix.shape}."
        raise ValueError(msg)
    return rotation_matrix.transpose(-1, -2)


# def euler_to_rmat(euler):
#     return tfg.rotation_matrix_3d.from_euler(euler)


# def invert_rmat(rot_mat):
#     return tfg.rotation_matrix_3d.inverse(rot_mat)


def rotmat_to_rot6d(mat: Float[torch.Tensor, ..., 3, 3]) -> Float[torch.Tensor, ..., 6]:
    """Convert rotation matrix to R6 rotation representation (first two rows in rotation matrix).

    Args:
        mat: Rotation matrix tensor of shape (..., 3, 3).

    Returns:
        6D rotation vector containing the first two rows of the rotation matrix,
        with shape (..., 6).

    Note:
        The 6D representation uses the first two rows of the rotation matrix
        as proposed in Zhou et al. "On the Continuity of Rotation Representations
        in Neural Networks".

    """
    r6 = mat[..., :2, :]
    r6_0, r6_1 = r6[..., 0, :], r6[..., 1, :]
    # r6_flat = torch.cat([r6_0, r6_1], axis=-1)
    return torch.concat([r6_0, r6_1], dim=-1)


def velocity_act_to_wrist_frame(velocity: torch.Tensor, wrist_in_robot_frame: torch.Tensor) -> torch.Tensor:
    """Translate velocity actions (translation + rotation) from base frame of the robot to wrist frame.

    Args:
        velocity: 6D velocity action tensor with shape (batch_size, 6).
            First 3 elements are translation, last 3 are rotation.
        wrist_in_robot_frame: 6D pose of the end-effector in robot base frame
            with shape (batch_size, 6). First 3 elements are position,
            last 3 are Euler angles.

    Returns:
        9D velocity action in robot wrist frame with shape (batch_size, 9).
        First 3 elements are translation, last 6 are rotation in R6 representation.

    Note:
        This function transforms velocity commands from the robot's base frame
        to the wrist frame, which is useful for end-effector control.

    """
    R_frame = euler_angles_to_matrix(wrist_in_robot_frame[:, 3:6]) # (bs, 3, 3)
    R_frame_inv = inverse_rotation_matrix(R_frame)  # (bs, 3, 3)

    # world to wrist: dT_pi = R^-1 dT_rbt
    vel_t = (R_frame_inv @ velocity[:, :3][..., None])[..., 0] # (bs, 3)

    # world to wrist: dR_pi = R^-1 dR_rbt R
    dR = euler_angles_to_matrix(velocity[:, 3:6])
    dR = R_frame_inv @ (dR @ R_frame)
    dR_r6 = rotmat_to_rot6d(dR) # (bs, 6)
    return torch.concat([vel_t, dR_r6], dim=-1)


def rand_swap_exterior_images(img1: torch.Tensor, img2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Randomly swaps the two exterior images (for training with single exterior input).

    Args:
        img1: First exterior image tensor.
        img2: Second exterior image tensor.

    Returns:
        Tuple of (img1, img2) or (img2, img1) with 50% probability each.

    Note:
        This augmentation helps train models that can work with either
        exterior camera view when only one is available during inference.

    """
    swap_probability = 0.5
    if torch.rand(1).item() > swap_probability:
        return img1, img2
    return img2, img1


def droid_baseact_transform(trajectory: dict) -> dict:
    """DROID dataset transformation for actions expressed in *base* frame of the robot.

    Args:
        trajectory: Dictionary containing trajectory data with keys:
            - "action_dict": Dictionary with "cartesian_velocity" and "gripper_position"
            - "observation": Dictionary with robot state and sensor data

    Returns:
        Modified trajectory dictionary with transformed actions and observations.
        The "action" key contains 7D actions: [translation(3), rotation(3), gripper(1)].
        The "observation" key is updated with swapped exterior images and concatenated
        proprioceptive state.

    Note:
        This transform prepares DROID data for training with base-frame actions.
        Gripper position is inverted (1 - gripper_position) to match expected format.

    """
    dt = trajectory["action_dict"]["cartesian_velocity"][:, :3]
    dR = trajectory["action_dict"]["cartesian_velocity"][:, 3:6]

    trajectory["action"] = torch.concat(
        (
            dt,
            dR,
            1 - trajectory["action_dict"]["gripper_position"],
        ),
        dim=-1,
    )
    trajectory["observation"]["exterior_image_1_left"], trajectory["observation"]["exterior_image_2_left"] = (
        rand_swap_exterior_images(
            trajectory["observation"]["exterior_image_1_left"],
            trajectory["observation"]["exterior_image_2_left"],
        )
    )
    trajectory["observation"]["proprio"] = torch.concat(
        (
            trajectory["observation"]["cartesian_position"],
            trajectory["observation"]["gripper_position"],
        ),
        dim=-1,
    )
    return trajectory


def droid_wristact_transform(trajectory: dict) -> dict:
    """DROID dataset transformation for actions expressed in *wrist* frame of the robot.

    Args:
        trajectory: Dictionary containing trajectory data with keys:
            - "action_dict": Dictionary with "cartesian_velocity" and "gripper_position"
            - "observation": Dictionary with robot state and sensor data

    Returns:
        Modified trajectory dictionary with transformed actions and observations.
        The "action" key contains 10D actions: [translation(3), rotation_r6(6), gripper(1)].
        The "observation" key is updated with swapped exterior images and concatenated
        proprioceptive state.

    Note:
        This transform converts velocity actions to the wrist frame using R6 rotation
        representation, which can be more stable for training than Euler angles.

    """
    wrist_act = velocity_act_to_wrist_frame(
        trajectory["action_dict"]["cartesian_velocity"],
        trajectory["observation"]["cartesian_position"],
    )
    trajectory["action"] = torch.concat(
        (
            wrist_act,
            trajectory["action_dict"]["gripper_position"],
        ),
        dim=-1,
    )
    trajectory["observation"]["exterior_image_1_left"], trajectory["observation"]["exterior_image_2_left"] = (
        rand_swap_exterior_images(
            trajectory["observation"]["exterior_image_1_left"],
            trajectory["observation"]["exterior_image_2_left"],
        )
    )
    trajectory["observation"]["proprio"] = torch.concat(
        (
            trajectory["observation"]["cartesian_position"],
            trajectory["observation"]["gripper_position"],
        ),
        dim=-1,
    )
    return trajectory


def droid_finetuning_transform(trajectory: dict) -> dict:
    """DROID dataset transformation for actions expressed in *base* frame of the robot.

    Args:
        trajectory: Dictionary containing trajectory data with keys:
            - "action_dict": Dictionary with "cartesian_velocity" and "gripper_position"
            - "observation": Dictionary with robot state and sensor data

    Returns:
        Modified trajectory dictionary with transformed actions and observations.
        The "action" key contains 7D actions: [translation(3), rotation(3), gripper(1)].
        The "observation" key is updated with concatenated proprioceptive state.

    Note:
        Similar to droid_baseact_transform but without exterior image swapping,
        designed for fine-tuning scenarios where data augmentation may not be desired.

    """
    dt = trajectory["action_dict"]["cartesian_velocity"][:, :3]
    dR = trajectory["action_dict"]["cartesian_velocity"][:, 3:6]
    trajectory["action"] = torch.concat(
        (
            dt,
            dR,
            1 - trajectory["action_dict"]["gripper_position"],
        ),
        dim=-1,
    )
    trajectory["observation"]["proprio"] = torch.concat(
        (
            trajectory["observation"]["cartesian_position"],
            trajectory["observation"]["gripper_position"],
        ),
        dim=-1,
    )
    return trajectory


def zero_action_filter(traj: dict) -> bool:
    """Filter transitions whose actions are all-0 (only relative actions, no gripper action).

    Args:
        traj: Trajectory dictionary containing an "action" key with action tensor
            of shape (..., action_dim) where the first 6 dimensions are the
            relative actions to be checked.

    Returns:
        True if the trajectory contains non-zero actions, False if all actions
        are effectively zero (within tolerance).

    Note:
        This filter is applied *after* action normalization, so it compares against
        the normalized zero action computed using DROID dataset statistics.
        Uses a tolerance of 1e-5 to account for floating point precision.

    Examples:
        >>> traj = {"action": torch.zeros(10, 7)}  # All zero actions
        >>> zero_action_filter(traj)
        False
        >>> traj = {"action": torch.randn(10, 7)}  # Non-zero actions
        >>> zero_action_filter(traj)
        True

    """
    DROID_Q01 = torch.tensor(
        [
            -0.7776297926902771,
            -0.5803514122962952,
            -0.5795090794563293,
            -0.6464047729969025,
            -0.7041108310222626,
            -0.8895104378461838,
        ],
    )
    DROID_Q99 = torch.tensor(
        [
            0.7597932070493698,
            0.5726242214441299,
            0.7351000607013702,
            0.6705610305070877,
            0.6464948207139969,
            0.8897542208433151,
        ],
    )
    DROID_NORM_0_ACT = 2 * (torch.zeros_like(traj["action"][:, :6]) - DROID_Q01) / (DROID_Q99 - DROID_Q01 + 1e-8) - 1

    return bool(torch.any(torch.abs(traj["action"][:, :6] - DROID_NORM_0_ACT) > 1e-5).item())
