from pathlib import Path


path = Path("/opt/dn-splatter/dn_splatter/scripts/align_depth.py")
text = path.read_text()

start = text.index("def colmap_sfm_points_to_depths(")
end = text.index("\ndef sdfstudio_grad_descent(", start)

replacement = r'''def colmap_sfm_points_to_depths(
    recon_dir: Path,
    output_dir: Path,
    min_depth: float = 0.001,
    max_depth: float = 1000,
    max_repoj_err: float = 2.5,
    min_n_visible: int = 5,
    include_depth_debug: bool = True,
    input_images_dir: Optional[Path] = Path(),
) -> Dict[int, Path]:
    """Converts COLMAP's points3d.bin to sparse depth maps."""
    depth_scale_to_integer_factor = 1
    output_dir.mkdir(parents=True, exist_ok=True)

    if (recon_dir / "points3D.bin").exists():
        ptid_to_info = read_points3D_binary(recon_dir / "points3D.bin")
        cam_id_to_camera = read_cameras_binary(recon_dir / "cameras.bin")
        im_id_to_image = read_images_binary(recon_dir / "images.bin")
    elif (recon_dir / "points3D.txt").exists():
        ptid_to_info = read_points3D_text(recon_dir / "points3D.txt")
        cam_id_to_camera = read_cameras_text(recon_dir / "cameras.txt")
        im_id_to_image = read_images_text(recon_dir / "images.txt")
    else:
        raise FileNotFoundError(f"No COLMAP points3D model found in {recon_dir}")

    image_id_to_depth_path = {}
    iter_images = iter(im_id_to_image.items())

    for im_id, im_data in track(iter_images, description="..."):
        camera = cam_id_to_camera[im_data.camera_id]
        W = camera.width
        H = camera.height

        valid_pairs = [
            (idx, pid)
            for idx, pid in enumerate(im_data.point3D_ids)
            if pid != -1 and pid in ptid_to_info
        ]

        depth = np.zeros((H, W), dtype=np.float32)

        if valid_pairs:
            pids = [pid for _, pid in valid_pairs]
            xyz_world = np.array([ptid_to_info[pid].xyz for pid in pids]).reshape(-1, 3)
            rotation = qvec2rotmat(im_data.qvec)
            z = (rotation @ xyz_world.T)[-1] + im_data.tvec[-1]
            errors = np.array([ptid_to_info[pid].error for pid in pids])
            n_visible = np.array([len(ptid_to_info[pid].image_ids) for pid in pids])
            uv = np.array([im_data.xys[idx] for idx, _ in valid_pairs])

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

            if len(z) > 0:
                uu, vv = uv[:, 0].astype(int), uv[:, 1].astype(int)
                depth[vv, uu] = z

        depth_img = depth_scale_to_integer_factor * depth
        out_name = Path(str(im_data.name)).stem
        depth_path = output_dir / out_name
        save_depth(
            depth=depth_img, depth_path=depth_path, scale_factor=1, verbose=False
        )
        image_id_to_depth_path[im_id] = depth_path

        if include_depth_debug:
            assert (
                input_images_dir is not None
            ), "Need explicit input_images_dir for debug images"
            assert input_images_dir.exists(), input_images_dir
            depth_flat = depth.flatten()[:, None]
            overlay = (
                255.0
                * colormaps.apply_depth_colormap(torch.from_numpy(depth_flat)).numpy()
            )
            overlay = overlay.reshape([H, W, 3])
            input_image_path = input_images_dir / im_data.name
            input_image = cv2.imread(str(input_image_path))  # type: ignore
            if input_image is None:
                continue
            if input_image.shape[:2] != overlay.shape[:2]:
                print("images are not the right size!")
                continue
            debug = 0.3 * input_image + 0.7 + overlay
            out_name_debug = out_name + ".debug.jpg"
            output_path = output_dir / "debug_depth" / out_name_debug
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), debug.astype(np.uint8))  # type: ignore

    return image_id_to_depth_path
'''

path.write_text(text[:start] + replacement + text[end:])
