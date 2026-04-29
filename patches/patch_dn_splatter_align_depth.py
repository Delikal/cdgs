from pathlib import Path


path = Path("/opt/dn-splatter/dn_splatter/scripts/align_depth.py")
text = path.read_text()

old_camera_block = """    # Only support first camera
    CAMERA_ID = 1
    W = cam_id_to_camera[CAMERA_ID].width
    H = cam_id_to_camera[CAMERA_ID].height

    iter_images = iter(im_id_to_image.items())
"""

new_camera_block = """    iter_images = iter(im_id_to_image.items())
"""

old_points_block = """        # TODO(1480) BEGIN delete when abandoning colmap_parsing_utils
        pids = [pid for pid in im_data.point3D_ids if pid != -1]
        xyz_world = np.array([ptid_to_info[pid].xyz for pid in pids])
        # delete
        # xyz_world = np.array([p.xyz for p in ptid_to_info.values()])
        rotation = qvec2rotmat(im_data.qvec)
        z = (rotation @ xyz_world.T)[-1] + im_data.tvec[-1]
        errors = np.array([ptid_to_info[pid].error for pid in pids])
        n_visible = np.array([len(ptid_to_info[pid].image_ids) for pid in pids])

        uv = np.array(
            [
                im_data.xys[i]
                for i in range(len(im_data.xys))
                if im_data.point3D_ids[i] != -1
            ]
        )

        idx = np.where(
            (z >= min_depth)
            & (z <= max_depth)
            & (errors <= max_repoj_err)
            & (n_visible >= min_n_visible)
            & (uv[:, 0] >= 0)
            & (uv[:, 0] < W)
            & (uv[:, 1] >= 0)
            & (uv[:, 1] < H)
        )
        z = z[idx]
        uv = uv[idx]

        uu, vv = uv[:, 0].astype(int), uv[:, 1].astype(int)

        depth = np.zeros((H, W), dtype=np.float32)
        depth[vv, uu] = z
"""

new_points_block = """        camera = cam_id_to_camera[im_data.camera_id]
        W = camera.width
        H = camera.height

        pids = [pid for pid in im_data.point3D_ids if pid != -1 and pid in ptid_to_info]
        depth = np.zeros((H, W), dtype=np.float32)

        if pids:
            xyz_world = np.array([ptid_to_info[pid].xyz for pid in pids]).reshape(-1, 3)
            rotation = qvec2rotmat(im_data.qvec)
            z = (rotation @ xyz_world.T)[-1] + im_data.tvec[-1]
            errors = np.array([ptid_to_info[pid].error for pid in pids])
            n_visible = np.array([len(ptid_to_info[pid].image_ids) for pid in pids])

            uv = np.array(
                [
                    im_data.xys[i]
                    for i in range(len(im_data.xys))
                    if im_data.point3D_ids[i] != -1
                    and im_data.point3D_ids[i] in ptid_to_info
                ]
            )

            idx = np.where(
                (z >= min_depth)
                & (z <= max_depth)
                & (errors <= max_repoj_err)
                & (n_visible >= min_n_visible)
                & (uv[:, 0] >= 0)
                & (uv[:, 0] < W)
                & (uv[:, 1] >= 0)
                & (uv[:, 1] < H)
            )
            z = z[idx]
            uv = uv[idx]

            uu, vv = uv[:, 0].astype(int), uv[:, 1].astype(int)
            depth[vv, uu] = z
"""

for old, new, label in [
    (old_camera_block, new_camera_block, "camera block"),
    (old_points_block, new_points_block, "points block"),
]:
    if old not in text:
        raise RuntimeError(f"Could not patch dn-splatter align_depth.py: missing {label}")
    text = text.replace(old, new)

path.write_text(text)
