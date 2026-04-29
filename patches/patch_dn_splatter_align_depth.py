from pathlib import Path


path = Path("/opt/dn-splatter/dn_splatter/scripts/align_depth.py")
text = path.read_text()

old_alignment_block = '''            # filter out aligned depth and frames not have pose
            sfm_name = [item.name for item in sfm_depth_filenames]
            mono_depth_filenames = [
                item
                for item in mono_depth_filenames
                if "_aligned.npy" not in item.name and str(item.stem) in str(sfm_name)
            ]
            assert len(sfm_depth_filenames) == len(mono_depth_filenames)
'''

new_alignment_block = '''            # Filter out aligned depths and frames that do not have a COLMAP pose.
            sfm_depth_filenames = sorted((self.data / Path("sfm_depths")).rglob("*.npy"))
            mono_depth_filenames = sorted((self.data / Path("mono_depth")).rglob("*.npy"))

            def _depth_key(path: Path, root_name: str) -> str:
                rel = path
                if root_name in path.parts:
                    rel = Path(*path.parts[path.parts.index(root_name) + 1 :])
                stem = str(rel.with_suffix(""))
                if stem.endswith("_aligned"):
                    stem = stem[: -len("_aligned")]
                return stem

            sfm_by_key = {
                _depth_key(path, "sfm_depths"): path for path in sfm_depth_filenames
            }
            mono_by_key = {
                _depth_key(path, "mono_depth"): path
                for path in mono_depth_filenames
                if "_aligned.npy" not in path.name
            }
            common_keys = sorted(set(sfm_by_key) & set(mono_by_key))
            if not common_keys:
                raise RuntimeError(
                    "No matching SfM and mono depth maps found. "
                    "Check that mono_depth and sfm_depths preserve the same relative image paths."
                )
            sfm_depth_filenames = [sfm_by_key[key] for key in common_keys]
            mono_depth_filenames = [mono_by_key[key] for key in common_keys]
'''

if old_alignment_block in text:
    text = text.replace(old_alignment_block, new_alignment_block)
else:
    print("Warning: did not patch align_depth main filename pairing block")

old_closed_form_block = '''                    mask = (sparse_depths > 0.1) & (sparse_depths < 10.0)
                    scale, shift = compute_scale_and_shift(
                        mono_depth_tensors, sparse_depths, mask=mask
                    )
                    scale = scale.unsqueeze(1).unsqueeze(2)
                    shift = shift.unsqueeze(1).unsqueeze(2)
                    depth_aligned = scale * mono_depth_tensors + shift
                    mse_loss = torch.nn.MSELoss()
                    avg = mse_loss(depth_aligned[mask], sparse_depths[mask])
                    CONSOLE.print(
                        f"[bold yellow]Average depth alignment error for batch depths is: {avg:3f} which is {'good' if avg<0.2 else 'bad'}"
                    )
'''

new_closed_form_block = '''                    mask = (sparse_depths > 0.1) & (sparse_depths < 10.0)
                    scale, shift = compute_scale_and_shift(
                        mono_depth_tensors, sparse_depths, mask=mask
                    )
                    valid_frames = mask.flatten(1).any(dim=1)
                    scale = torch.where(valid_frames, scale, torch.ones_like(scale))
                    shift = torch.where(valid_frames, shift, torch.zeros_like(shift))
                    scale = scale.unsqueeze(1).unsqueeze(2)
                    shift = shift.unsqueeze(1).unsqueeze(2)
                    depth_aligned = scale * mono_depth_tensors + shift
                    if mask.any():
                        mse_loss = torch.nn.MSELoss()
                        avg = mse_loss(depth_aligned[mask], sparse_depths[mask])
                        CONSOLE.print(
                            f"[bold yellow]Average depth alignment error for batch depths is: {avg:3f} which is {'good' if avg<0.2 else 'bad'}"
                        )
                    else:
                        CONSOLE.print(
                            "[bold yellow]No sparse depth pixels in this batch; keeping mono depths unaligned."
                        )
'''

if old_closed_form_block in text:
    text = text.replace(old_closed_form_block, new_closed_form_block)
else:
    print("Warning: did not patch align_depth closed-form empty-mask handling")

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
        out_name = Path(str(im_data.name)).with_suffix("")
        depth_path = output_dir / out_name
        depth_path.parent.mkdir(parents=True, exist_ok=True)
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
            out_name_debug = Path(str(im_data.name)).with_suffix(".debug.jpg")
            output_path = output_dir / "debug_depth" / out_name_debug
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), debug.astype(np.uint8))  # type: ignore

    return image_id_to_depth_path
'''

path.write_text(text[:start] + replacement + text[end:])


dataparser_path = Path("/opt/dn-splatter/dn_splatter/data/coolermap_dataparser.py")
dataparser_text = dataparser_path.read_text()
dp_start = dataparser_text.index("    def get_depth_filepaths(self):")
dp_end = dataparser_text.index("\n    def get_normal_filepaths(self):", dp_start)

dp_replacement = r'''    def get_depth_filepaths(self):
        # Rig datasets keep depth files in the same relative tree as images.
        mono_depth_dir = self.config.data / "mono_depth"
        depth_paths = natsorted([str(path) for path in mono_depth_dir.rglob("*_aligned.npy")])
        if not depth_paths:
            CONSOLE.log("Could not find _aligned.npy depths, trying *.npy")
            depth_paths = natsorted(
                [
                    str(path)
                    for path in mono_depth_dir.rglob("*.npy")
                    if not path.name.endswith("_aligned.npy")
                ]
            )
        if depth_paths:
            CONSOLE.log("Found depths ending in *.npy")
        else:
            CONSOLE.log("Could not find depths, quitting.")
            quit()
        return depth_paths
'''

dataparser_path.write_text(
    dataparser_text[:dp_start] + dp_replacement + dataparser_text[dp_end:]
)
