import torch
import torch.nn.functional as F
from pytorch3d.renderer.mesh import rasterize_meshes
from pytorch3d.structures import Meshes
from pytorch3d.ops import interpolate_face_attributes


class RGBDRenderer:
    """RGBD renderer for novel view synthesis based on PyTorch3D.

    Constructs a triangular mesh from input RGBD images and renders novel views
    using random camera extrinsics. Supports batch rendering, depth normalization,
    and Sobel edge-aware masking.
    """

    def __init__(self, device):
        """Initialize the renderer.

        Args:
            device: compute device (e.g., "cuda:3")
        """
        self.device = device
        self.eps = 0.1
        self.near_z = 1e-4
        self.far_z = 1e4

    def render_mesh(self, mesh_dict, cam_int, cam_ext):
        """Render a 3D mesh onto a 2D image plane.

        Pipeline:
        1. Transform vertices from camera space to world space
        2. Map vertices to NDC space via perspective projection
        3. Rasterize using PyTorch3D rasterizer
        4. Interpolate color, mask, and depth for each pixel

        Args:
            mesh_dict: mesh dictionary containing vertice, faces, attributes, size
            cam_int: camera intrinsics (b, 3, 3)
            cam_ext: camera extrinsics (b, 3, 4)

        Returns:
            render: rendered RGB image (b, 3, h, w)
            disparity: rendered disparity map (b, 1, h, w)
            mask: valid region mask (b, 1, h, w)
            object_mask: object mask (b, 1, h, w)
        """
        vertice = mesh_dict["vertice"]  # [b, h*w, 3]
        faces = mesh_dict["faces"]  # [b, nface, 3]
        attributes = mesh_dict["attributes"]  # [b, h*w, 4]
        h, w = mesh_dict["size"]

        # Transform vertices to world space
        vertice_homo = self.lift_to_homo(vertice)  # [b, h*w, 4]
        vertice_world = torch.matmul(
            cam_ext.unsqueeze(1), vertice_homo[..., None]
        ).squeeze(-1)  # [b, h*w, 3]
        vertice_depth = vertice_world[..., -1:]  # [b, h*w, 1]
        attributes = torch.cat([attributes, vertice_depth], dim=-1)  # [b, h*w, 5]

        # Perspective projection to NDC space
        vertice_world_homo = self.lift_to_homo(vertice_world)
        persp = self.get_perspective_from_intrinsic(cam_int)  # [b, 4, 4]
        vertice_ndc = torch.matmul(
            persp.unsqueeze(1), vertice_world_homo[..., None]
        ).squeeze(-1)  # [b, h*w, 4]
        vertice_ndc = vertice_ndc[..., :-1] / vertice_ndc[..., -1:]
        vertice_ndc[..., :-1] *= -1
        vertice_ndc[..., 0] *= w / h

        # Rasterization
        mesh = Meshes(vertice_ndc, faces)
        pix_to_face, _, bary_coords, _ = rasterize_meshes(
            mesh, (h, w), faces_per_pixel=1, blur_radius=1e-6
        )

        b, nf, _ = faces.size()
        faces = faces.reshape(b, nf * 3, 1).repeat(1, 1, 6)
        face_attributes = torch.gather(attributes, dim=1, index=faces)
        face_attributes = face_attributes.reshape(b * nf, 3, 6)
        output = interpolate_face_attributes(pix_to_face, bary_coords, face_attributes)
        output = output.squeeze(-2).permute(0, 3, 1, 2)

        render = output[:, :3]
        mask = output[:, 3:4]
        object_mask = output[:, 4:5]
        disparity = torch.reciprocal(output[:, 5:] + 1e-4)

        return render * mask, disparity * mask, mask, object_mask

    def construct_mesh(self, rgbd, cam_int, obj_mask, normalize_depth=False):
        """Construct a triangular mesh from an RGBD image.

        Pipeline:
        1. Get normalized screen pixel coordinates
        2. Back-project to 3D space using inverse intrinsics and depth
        3. Build triangular faces
        4. Compute vertex attributes (color, visibility mask, object mask)

        Args:
            rgbd: RGBD tensor (b, 4, h, w), channel order [R, G, B, Disparity]
            cam_int: camera intrinsics (b, 3, 3)
            obj_mask: object mask (b, 1, h, w)
            normalize_depth: whether to normalize depth

        Returns:
            mesh_dict: dictionary containing vertice, faces, attributes, size
        """
        b, _, h, w = rgbd.size()

        # Get pixel coordinates and back-project to 3D
        pixel_2d = self.get_screen_pixel_coord(h, w)  # [1, h, w, 2]
        pixel_2d_homo = self.lift_to_homo(pixel_2d)  # [1, h, w, 3]

        rgbd = rgbd.permute(0, 2, 3, 1)  # [b, h, w, 4]
        disparity = rgbd[..., -1:]  # [b, h, w, 1]
        depth = torch.reciprocal(disparity + 1e-2)  # [b, h, w, 1]
        obj_mask = obj_mask.permute(0, 2, 3, 1).to(depth.device)

        cam_int_inv = torch.inverse(cam_int)
        pixel_3d = torch.matmul(
            cam_int_inv[:, None, None, :, :], pixel_2d_homo[..., None]
        ).squeeze(-1)  # [b, h, w, 3]
        pixel_3d = pixel_3d * depth  # [b, h, w, 3]
        vertice = pixel_3d.reshape(b, h * w, 3)

        # Build faces
        faces = self.get_faces(h, w).repeat(b, 1, 1).long()

        # Compute vertex attributes
        attr_color = rgbd[..., :-1].reshape(b, h * w, 3)
        attr_object = obj_mask.reshape(b, h * w, 1).to(attr_color.device)
        attr_mask = self.get_visible_mask(disparity, alpha_threshold=0.1).reshape(
            b, h * w, 1
        )
        attr = torch.cat([attr_color, attr_mask, attr_object], dim=-1)

        mesh_dict = {
            "vertice": vertice,
            "faces": faces,
            "attributes": attr,
            "size": [h, w],
        }
        return mesh_dict

    def get_screen_pixel_coord(self, h, w):
        """Get normalized screen pixel coordinates.

        Origin at top-left, x right, y down, range [0, 1].

        Args:
            h: image height
            w: image width

        Returns:
            pixel_coord: normalized pixel coordinates (1, h, w, 2)
        """
        x = torch.arange(w).to(self.device)
        y = torch.arange(h).to(self.device)
        x = (x + 0.5) / w
        y = (y + 0.5) / h
        x = x[None, None, ..., None].repeat(1, h, 1, 1)
        y = y[None, ..., None, None].repeat(1, 1, w, 1)
        pixel_coord = torch.cat([x, y], dim=-1)
        return pixel_coord

    def lift_to_homo(self, coord):
        """Lift coordinates to homogeneous coordinates.

        Args:
            coord: input coordinates (..., k)

        Returns:
            homo_coord: homogeneous coordinates (..., k+1), last channel filled with 1
        """
        ones = torch.ones_like(coord[..., -1:])
        return torch.cat([coord, ones], dim=-1)

    def get_faces(self, h, w):
        """Build triangular face indices for an h x w grid.

        Each 2x2 grid is split into two triangles:
        - Upper-left triangle: [tl, bl, br]
        - Lower-right triangle: [br, tr, tl]

        Args:
            h: grid height
            w: grid width

        Returns:
            faces: face indices (1, (h-1)*(w-1)*2, 3)
        """
        x = torch.arange(w - 1).to(self.device)
        y = torch.arange(h - 1).to(self.device)
        x = x[None, None, ..., None].repeat(1, h - 1, 1, 1)
        y = y[None, ..., None, None].repeat(1, 1, w - 1, 1)

        tl = y * w + x
        tr = y * w + x + 1
        bl = (y + 1) * w + x
        br = (y + 1) * w + x + 1

        faces_l = torch.cat([tl, bl, br], dim=-1).reshape(1, -1, 3)
        faces_r = torch.cat([br, tr, tl], dim=-1).reshape(1, -1, 3)

        return torch.cat([faces_l, faces_r], dim=1)

    def get_visible_mask(self, disparity, beta=10, alpha_threshold=0.3):
        """Compute visibility mask based on disparity gradient.

        Uses Sobel operator to compute gradient magnitude of the disparity map.
        Regions with large gradients (depth edges) are marked as invisible.

        Args:
            disparity: disparity map (b, h, w, 1)
            beta: gradient decay factor, default 10
            alpha_threshold: visibility threshold, default 0.3

        Returns:
            vis_mask: visibility mask (b, h, w, 1), 1 for visible, 0 for invisible
        """
        b, h, w, _ = disparity.size()
        disparity = disparity.reshape(b, 1, h, w)

        kernel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        ).unsqueeze(0).unsqueeze(0).float().to(self.device)
        kernel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
        ).unsqueeze(0).unsqueeze(0).float().to(self.device)

        sobel_x = F.conv2d(disparity, kernel_x, padding=(1, 1))
        sobel_y = F.conv2d(disparity, kernel_y, padding=(1, 1))
        sobel_mag = torch.sqrt(sobel_x ** 2 + sobel_y ** 2).reshape(b, h, w, 1)

        alpha = torch.exp(-1.0 * beta * sobel_mag)
        vis_mask = torch.greater(alpha, alpha_threshold).float()
        return vis_mask

    def get_perspective_from_intrinsic(self, cam_int):
        """Build perspective projection matrix from camera intrinsics.

        Uses near_z and far_z to compute z-buffer mapping parameters.

        Args:
            cam_int: camera intrinsics matrix (b, 3, 3)

        Returns:
            persp: perspective projection matrix (b, 4, 4)
        """
        fx, fy = cam_int[:, 0, 0], cam_int[:, 1, 1]
        cx, cy = cam_int[:, 0, 2], cam_int[:, 1, 2]
        one = torch.ones_like(cx)
        zero = torch.zeros_like(cx)

        near_z, far_z = self.near_z * one, self.far_z * one
        a = (near_z + far_z) / (far_z - near_z)
        b = -2.0 * near_z * far_z / (far_z - near_z)

        matrix = [
            [2.0 * fx, zero, 2.0 * cx - 1.0, zero],
            [zero, 2.0 * fy, 2.0 * cy - 1.0, zero],
            [zero, zero, a, b],
            [zero, zero, one, zero],
        ]
        persp = torch.stack(
            [torch.stack(row, dim=-1) for row in matrix], dim=-2
        )
        return persp


def transformation_from_parameters(axisangle, translation, invert=False):
    """Build a 4x4 transformation matrix from axis-angle rotation and translation.

    Reference: https://github.com/mattpoggi/depthstillation

    Args:
        axisangle: axis-angle representation of rotation (b, 3)
        translation: translation vector (b, 3)
        invert: whether to return the inverse transformation, default False

    Returns:
        M: 4x4 transformation matrix (b, 4, 4)
    """
    R = rot_from_axisangle(axisangle)
    t = translation.clone()

    if invert:
        R = R.transpose(1, 2)
        t *= -1

    T = get_translation_matrix(t)
    M = torch.matmul(T, R) if not invert else torch.matmul(R, T)
    return M


def get_translation_matrix(translation_vector):
    """Build a homogeneous translation matrix from a translation vector.

    Args:
        translation_vector: translation vector (b, 3)

    Returns:
        T: translation matrix (b, 4, 4)
    """
    T = torch.zeros(
        translation_vector.shape[0], 4, 4, device=translation_vector.device
    )
    t = translation_vector.contiguous().view(-1, 3, 1)
    T[:, 0, 0] = 1
    T[:, 1, 1] = 1
    T[:, 2, 2] = 1
    T[:, 3, 3] = 1
    T[:, :3, 3, None] = t
    return T


def rot_from_axisangle(vec):
    """Build a rotation matrix from an axis-angle vector.

    Computed using Rodrigues' rotation formula.

    Args:
        vec: axis-angle vector (b, 3)

    Returns:
        rot: 4x4 rotation matrix (b, 4, 4)
    """
    angle = torch.norm(vec, 2, 2, True)
    axis = vec / (angle + 1e-7)

    ca = torch.cos(angle)
    sa = torch.sin(angle)
    C = 1 - ca

    x = axis[..., 0].unsqueeze(1)
    y = axis[..., 1].unsqueeze(1)
    z = axis[..., 2].unsqueeze(1)

    xs = x * sa
    ys = y * sa
    zs = z * sa
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC

    rot = torch.zeros((vec.shape[0], 4, 4), device=vec.device)
    rot[:, 0, 0] = torch.squeeze(x * xC + ca)
    rot[:, 0, 1] = torch.squeeze(xyC - zs)
    rot[:, 0, 2] = torch.squeeze(zxC + ys)
    rot[:, 1, 0] = torch.squeeze(xyC + zs)
    rot[:, 1, 1] = torch.squeeze(y * yC + ca)
    rot[:, 1, 2] = torch.squeeze(yzC - xs)
    rot[:, 2, 0] = torch.squeeze(zxC - ys)
    rot[:, 2, 1] = torch.squeeze(yzC + xs)
    rot[:, 2, 2] = torch.squeeze(z * zC + ca)
    rot[:, 3, 3] = 1

    return rot